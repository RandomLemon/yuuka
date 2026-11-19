#include "../common.cuh"
#include "../iq_decode.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;
constexpr int kSpan = 8;

// row_stride returns the packed byte stride of one weight row: the per-block
// size times the number of 256-value blocks in the row. Only the K-quant and
// IQ2/IQ3 dtypes this kernel supports are listed; every other dtype writes
// nothing.
__device__ __forceinline__ int64_t row_stride(uint32_t dtype, int64_t n_in) {
    const int64_t blocks = n_in / 256;
    switch (dtype) {
        case YUUKA_DTYPE_Q4_K:
            return blocks * 144;
        case YUUKA_DTYPE_Q5_K:
            return blocks * 176;
        case YUUKA_DTYPE_Q6_K:
            return blocks * 210;
        case YUUKA_DTYPE_IQ2_XXS:
            return blocks * 66;
        case YUUKA_DTYPE_IQ2_S:
            return blocks * 82;
        case YUUKA_DTYPE_IQ3_S:
            return blocks * 110;
        default:
            return 0;
    }
}

// dequant_span_kernel expands a packed K-quant weight matrix into FP16. One
// thread decodes eight consecutive values, which stays inside one 32-value
// sub-block, so the Q4_K/Q5_K header, scale, and min are hoisted once per span.
//
// The Q4_K and Q5_K arithmetic mirrors decode_span<DType, 8> in
// gemm/wmma_gemm.hip exactly, including the 8-byte vector loads of the packed
// quants. The dequantized element is therefore the same FP16 value the rocWMMA
// path stages, so a rocBLAS run differs from a WMMA run only in accumulation.
// Q6_K reuses q6k_value for the same reason; one call per value is acceptable
// because this kernel is bandwidth bound.
__global__ void dequant_span_kernel(__half * dst, const uint8_t * w, int64_t n_in,
                                    int64_t n_out, uint32_t dtype) {
    const int64_t stride = row_stride(dtype, n_in);
    if (stride == 0) {
        return;
    }
    const int64_t spans_per_row = n_in / kSpan;
    const int64_t total = n_out * spans_per_row;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = i / spans_per_row;
        const int64_t col = (i % spans_per_row) * kSpan;
        const uint8_t * rowp = w + row * stride;
        __half * out = dst + row * n_in + col;
        if (dtype == YUUKA_DTYPE_IQ2_XXS || dtype == YUUKA_DTYPE_IQ2_S ||
            dtype == YUUKA_DTYPE_IQ3_S) {
            const int block_bytes = dtype == YUUKA_DTYPE_IQ2_XXS ? 66
                                   : dtype == YUUKA_DTYPE_IQ2_S  ? 82
                                                                 : 110;
            const uint8_t * blk = rowp + (col / 256) * block_bytes;
            const int j0 = static_cast<int>(col % 256);
            for (int j = 0; j < kSpan; ++j) {
                const int jj = j0 + j;
                if (dtype == YUUKA_DTYPE_IQ2_XXS) {
                    out[j] = __float2half(iq2xxs_value(blk, jj));
                } else if (dtype == YUUKA_DTYPE_IQ2_S) {
                    out[j] = __float2half(iq2s_value(blk, jj));
                } else {
                    out[j] = __float2half(iq3s_value(blk, jj));
                }
            }
            continue;
        }
        if (dtype == YUUKA_DTYPE_Q6_K) {
            const uint8_t * blk = rowp + (col / 256) * 210;
            for (int j = 0; j < kSpan; ++j) {
                out[j] = __float2half(q6k_value(blk, static_cast<int>(col % 256) + j));
            }
            continue;
        }
        const int sub = static_cast<int>((col % 256) / 32);
        const int within = static_cast<int>(col & 31);
        const bool high_nibble = (sub & 1) != 0;
        if (dtype == YUUKA_DTYPE_Q4_K) {
            const uint8_t * blk = rowp + (col / 256) * 144;
            const float d = load_f16_device(blk);
            const float dm = load_f16_device(blk + 2);
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(sub, blk + 4, sc, m);
            const float scale = d * static_cast<float>(sc);
            const float min = dm * static_cast<float>(m);
            const uint2 packed =
                *reinterpret_cast<const uint2 *>(blk + 16 + (sub / 2) * 32 + within);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int shift = 8 * v + (high_nibble ? 4 : 0);
                out[v] = __float2half(
                    scale * static_cast<float>((packed.x >> shift) & 0xF) - min);
                out[v + 4] = __float2half(
                    scale * static_cast<float>((packed.y >> shift) & 0xF) - min);
            }
            continue;
        }
        const uint8_t * blk = rowp + (col / 256) * 176;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float scale = d * static_cast<float>(sc);
        const float min = dm * static_cast<float>(m);
        const uint2 ql =
            *reinterpret_cast<const uint2 *>(blk + 48 + (sub / 2) * 32 + within);
        const uint2 qh = *reinterpret_cast<const uint2 *>(blk + 16 + within);
        const uint32_t bit = static_cast<uint32_t>(1u << sub);
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int shift = 8 * v + (high_nibble ? 4 : 0);
            int q0 = static_cast<int>((ql.x >> shift) & 0xF);
            int q1 = static_cast<int>((ql.y >> shift) & 0xF);
            if ((qh.x >> (8 * v)) & bit) {
                q0 += 16;
            }
            if ((qh.y >> (8 * v)) & bit) {
                q1 += 16;
            }
            out[v] = __float2half(scale * static_cast<float>(q0) - min);
            out[v + 4] = __float2half(scale * static_cast<float>(q1) - min);
        }
    }
}

} // namespace

namespace yuuka {

void launch_dequant_f16(void * dst, const void * w, uint64_t n_in, uint64_t n_out,
                        uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || n_in % 8 != 0) {
        return;
    }
    const int64_t spans =
        static_cast<int64_t>(n_out) * static_cast<int64_t>(n_in / 8);
    const int64_t blocks = (spans + kThreads - 1) / kThreads;
    dequant_span_kernel<<<static_cast<int>(blocks < 65535 ? blocks : 65535),
                          kThreads, 0, stream>>>(
        static_cast<__half *>(dst), static_cast<const uint8_t *>(w),
        static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), dtype);
}

} // namespace yuuka
