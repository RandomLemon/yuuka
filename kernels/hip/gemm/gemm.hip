#include "../yuuka_kernels.h"

#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float load_f16(const void * ptr) {
    return __half2float(*reinterpret_cast<const __half *>(ptr));
}

// One thread computes one output column for one batch row. This is a
// correctness baseline, not the Phase 3 tiled GEMM.
__global__ void gemm_f32_kernel(float * out, const float * x, const float * w,
                                int64_t n_in, int64_t n_out, int64_t batch) {
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= n_out || row >= batch) {
        return;
    }
    const float * x_row = x + row * n_in;
    const float * w_row = w + col * n_in;
    float sum = 0.0f;
    for (int64_t i = 0; i < n_in; ++i) {
        sum += x_row[i] * w_row[i];
    }
    out[row * n_out + col] = sum;
}

// gemm_f32_router_tile_kernel is the measured 2048x256x512 MoE router cell.
// One block covers 64 tokens by 64 output channels with a 16x16 thread grid;
// each thread accumulates a 4x4 output tile. Both operands are staged in LDS
// 32 K values at a time, and every output keeps the scalar kernel's single
// FP32 accumulator in strictly ascending K order. The padded row stride keeps
// the two lane groups of a warp on different banks.
constexpr int kRouterRows = 64;
constexpr int kRouterCols = 64;
constexpr int kRouterStep = 32;
constexpr int kRouterStride = kRouterStep + 1;

__global__ void gemm_f32_router_tile_kernel(float * out, const float * x,
                                            const float * w) {
    __shared__ float xs[kRouterRows * kRouterStride];
    __shared__ float ws[kRouterCols * kRouterStride];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * kRouterRows;
    const int col0 = blockIdx.x * kRouterCols;
    const int tid = ty * 16 + tx;
    float sum[4][4] = {};
    for (int k0 = 0; k0 < 2048; k0 += kRouterStep) {
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int index = tid * 4 + part * 1024;
            const int row = index / kRouterStep;
            const int kk = index % kRouterStep;
            *reinterpret_cast<float4 *>(xs + row * kRouterStride + kk) =
                *reinterpret_cast<const float4 *>(
                    x + int64_t(row0 + row) * 2048 + k0 + kk);
            *reinterpret_cast<float4 *>(ws + row * kRouterStride + kk) =
                *reinterpret_cast<const float4 *>(
                    w + int64_t(col0 + row) * 2048 + k0 + kk);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < kRouterStep; ++kk) {
            float xv[4];
            float wv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                xv[i] = xs[(ty + 16 * i) * kRouterStride + kk];
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wv[j] = ws[(tx + 16 * j) * kRouterStride + kk];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sum[i][j] += xv[i] * wv[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[int64_t(row0 + ty + 16 * i) * 256 + col0 + tx + 16 * j] =
                sum[i][j];
        }
    }
}

__global__ void gemm_f16_kernel(float * out, const float * x, const void * w,
                                int64_t n_in, int64_t n_out, int64_t batch) {
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= n_out || row >= batch) {
        return;
    }
    const float * x_row = x + row * n_in;
    const uint8_t * w_row = static_cast<const uint8_t *>(w) + col * n_in * 2;
    float sum = 0.0f;
    for (int64_t i = 0; i < n_in; ++i) {
        sum += x_row[i] * load_f16(w_row + i * 2);
    }
    out[row * n_out + col] = sum;
}

} // namespace

namespace yuuka {

void launch_gemm_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream) {
    // The router projection is the only measured cell; compact and unmeasured
    // shapes keep the scalar contract, including its exact accumulation order.
    if (n_in == 2048 && n_out == 256 && batch == 512 &&
        reinterpret_cast<uintptr_t>(x) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(w) % alignof(float4) == 0) {
        const dim3 grid(4, 8);
        const dim3 block(16, 16);
        gemm_f32_router_tile_kernel<<<grid, block, 0, stream>>>(out, x, w);
        return;
    }
    const dim3 grid(static_cast<unsigned>((n_out + kThreads - 1) / kThreads),
                    static_cast<unsigned>(batch));
    gemm_f32_kernel<<<grid, kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out),
        static_cast<int64_t>(batch));
}

void launch_gemm_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n_out + kThreads - 1) / kThreads),
                    static_cast<unsigned>(batch));
    gemm_f16_kernel<<<grid, kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out),
        static_cast<int64_t>(batch));
}

} // namespace yuuka
