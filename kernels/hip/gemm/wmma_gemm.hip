#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kBlockM = 32;
constexpr int kBlockN = 64;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;

// segment_value decodes the 32 values that start at a 32-aligned k offset. The
// K-quant superblock layouts split into 32-value sub-blocks, so one segment
// never crosses a scale boundary.
template <uint32_t DType>
__device__ __forceinline__ float segment_value(const uint8_t * row, int64_t k0,
                                               int lane, int64_t n_in) {
    const int64_t k = k0 + lane;
    if (k >= n_in) {
        return 0.0f;
    }
    if (DType == YUUKA_DTYPE_F32) {
        return *reinterpret_cast<const float *>(row + k * 4);
    } else if (DType == YUUKA_DTYPE_F16) {
        return load_f16_device(row + k * 2);
    } else if (DType == YUUKA_DTYPE_Q4_0) {
        const int64_t block = k0 / 32;
        const uint8_t * blk = row + block * 18;
        const uint8_t packed = blk[2 + (lane & 15)];
        const int q = (lane < 16) ? (packed & 0xF) : (packed >> 4);
        return load_f16_device(blk) * static_cast<float>(q - 8);
    } else if (DType == YUUKA_DTYPE_Q5_0) {
        const int64_t block = k0 / 32;
        const uint8_t * blk = row + block * 22;
        const uint8_t packed = blk[6 + (lane & 15)];
        const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                            (static_cast<uint32_t>(blk[3]) << 8) |
                            (static_cast<uint32_t>(blk[4]) << 16) |
                            (static_cast<uint32_t>(blk[5]) << 24);
        const int high = (lane < 16)
                             ? static_cast<int>(((qh >> lane) << 4) & 0x10)
                             : static_cast<int>((qh >> ((lane & 15) + 12)) & 0x10);
        const int q = (lane < 16 ? (packed & 0xF) : (packed >> 4)) | high;
        return load_f16_device(blk) * static_cast<float>(q - 16);
    } else if (DType == YUUKA_DTYPE_Q8_0) {
        const int64_t block = k0 / 32;
        return q8_value(row + block * 34, lane);
    } else if (DType == YUUKA_DTYPE_Q4_K) {
        const int64_t block = k0 / 256;
        const int sub = static_cast<int>((k0 % 256) / 32);
        const uint8_t * blk = row + block * 144;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const uint8_t * qs = blk + 16 + (sub / 2) * 32;
        const uint8_t q = (sub & 1) ? static_cast<uint8_t>(qs[lane] >> 4)
                                    : static_cast<uint8_t>(qs[lane] & 0xF);
        return d * sc * q - dm * m;
    } else if (DType == YUUKA_DTYPE_Q5_K) {
        const int64_t block = k0 / 256;
        const int sub = static_cast<int>((k0 % 256) / 32);
        const uint8_t * blk = row + block * 176;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const uint8_t * qh = blk + 16;
        const uint8_t * ql = blk + 48 + (sub / 2) * 32;
        int q = (sub & 1) ? (ql[lane] >> 4) : (ql[lane] & 0xF);
        if (qh[lane] & static_cast<uint8_t>(1u << sub)) {
            q += 16;
        }
        return d * sc * q - dm * m;
    } else if (DType == YUUKA_DTYPE_Q6_K) {
        const int64_t block = k0 / 256;
        const int sub = static_cast<int>((k0 % 256) / 32);
        const int half = sub / 4;
        const int seg = sub % 4;
        const uint8_t * blk = row + block * 210;
        const uint8_t * ql = blk + half * 64;
        const uint8_t * qh = blk + 128 + half * 32;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + half * 8;
        const float d = load_f16_device(blk + 208);
        int q = 0;
        if (seg == 0) {
            q = (ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4);
        } else if (seg == 1) {
            q = (ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4);
        } else if (seg == 2) {
            q = (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
        } else {
            q = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
        }
        const int scale_index = seg * 2 + lane / 16;
        return d * sc[scale_index] * static_cast<float>(q - 32);
    }
    return 0.0f;
}

template <uint32_t DType>
__global__ void gemm_wmma_kernel(float * out, const float * x, const uint8_t * w,
                                 int64_t n_in, int64_t n_out, int64_t rows,
                                 uint64_t row_bytes) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp / (kBlockN / 16);
    const int wn = warp % (kBlockN / 16);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kBlockM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __half a_tile[kBlockM * kBlockK];
    __shared__ __half b_tile[kBlockN * kBlockK];
    __shared__ float c_tile[kBlockM * kBlockN];

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (int64_t k0 = 0; k0 < n_in; k0 += kBlockK) {
        for (int idx = tid; idx < kBlockM * kBlockK; idx += kThreads) {
            const int r = idx / kBlockK;
            const int c = idx % kBlockK;
            const int64_t gr = m0 + r;
            const int64_t gk = k0 + c;
            const float value = (gr < rows && gk < n_in) ? x[gr * n_in + gk] : 0.0f;
            a_tile[r * kBlockK + c] = __float2half(value);
        }
        for (int s = warp; s < kBlockN; s += kWaves) {
            const int64_t gn = n0 + s;
            float value = 0.0f;
            if (gn < n_out) {
                value = segment_value<DType>(w + gn * row_bytes, k0, lane, n_in);
            }
            b_tile[s * kBlockK + lane] = __float2half(value);
        }
        __syncthreads();
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag;
        for (int kk = 0; kk < kBlockK; kk += 16) {
            if (k0 + kk >= n_in) {
                break;
            }
            rocwmma::load_matrix_sync(a_frag, a_tile + (wm * 16) * kBlockK + kk, kBlockK);
            rocwmma::load_matrix_sync(b_frag, b_tile + (wn * 16) * kBlockK + kk, kBlockK);
            rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + wn * 16, c_frag, kBlockN,
                               rocwmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < kBlockM * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

uint64_t packed_row_bytes(uint32_t dtype, uint64_t n_in) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return n_in * 4;
        case YUUKA_DTYPE_F16:
            return n_in * 2;
        case YUUKA_DTYPE_Q4_0:
            return (n_in / 32) * 18;
        case YUUKA_DTYPE_Q5_0:
            return (n_in / 32) * 22;
        case YUUKA_DTYPE_Q8_0:
            return (n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K:
            return (n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K:
            return (n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K:
            return (n_in / 256) * 210;
        default:
            return 0;
    }
}

} // namespace

namespace yuuka {

void launch_gemm_wmma_f32out(float * out, const float * x, const void * w,
                             uint64_t n_in, uint64_t n_out, uint64_t batch,
                             uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0) {
        return;
    }
    const uint64_t row_bytes = packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((batch + kBlockM - 1) / kBlockM));
    const auto * weights = static_cast<const uint8_t *>(w);
    const int64_t m = static_cast<int64_t>(batch);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            gemm_wmma_kernel<YUUKA_DTYPE_F32><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_F16:
            gemm_wmma_kernel<YUUKA_DTYPE_F16><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q4_0:
            gemm_wmma_kernel<YUUKA_DTYPE_Q4_0><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q5_0:
            gemm_wmma_kernel<YUUKA_DTYPE_Q5_0><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q8_0:
            gemm_wmma_kernel<YUUKA_DTYPE_Q8_0><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q4_K:
            gemm_wmma_kernel<YUUKA_DTYPE_Q4_K><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q5_K:
            gemm_wmma_kernel<YUUKA_DTYPE_Q5_K><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        case YUUKA_DTYPE_Q6_K:
            gemm_wmma_kernel<YUUKA_DTYPE_Q6_K><<<grid, kThreads, 0, stream>>>(
                out, x, weights, in, on, m, row_bytes);
            break;
        default:
            break;
    }
}

} // namespace yuuka
