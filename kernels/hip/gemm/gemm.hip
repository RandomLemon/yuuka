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
