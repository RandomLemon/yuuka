#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kBlockN = 64;
// The plain GEMM stages a 64-row tile with a 32-wide K step: 16 KiB of shared
// memory, so four CTAs stay resident per 64 KiB LDS CU. The fused gate/up
// kernel stages three tiles and also uses a 32-wide K step.
constexpr int kFusedBlockK = 32;
constexpr int kThreads = 256;

// write_q4k_word decodes four Q4_K values from one packed 32-bit word. Even
// sub-blocks occupy the low nibble and odd sub-blocks the high nibble of the
// same byte, so four consecutive bytes hold eight values.
__device__ __forceinline__ void write_q4k_word(uint32_t word, bool high_nibble,
                                               float scale, float min,
                                               __half * out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int shift = 8 * i + (high_nibble ? 4 : 0);
        const int q = static_cast<int>((word >> shift) & 0xF);
        out[i] = __float2half(scale * static_cast<float>(q) - min);
    }
}

// write_q5k_word decodes four Q5_K values from one packed ql word and the four
// matching qh bytes. Bit sub of each qh byte carries the fifth value bit.
__device__ __forceinline__ void write_q5k_word(uint32_t ql, uint32_t qh,
                                               bool high_nibble, int sub,
                                               float scale, float min,
                                               __half * out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int shift = 8 * i + (high_nibble ? 4 : 0);
        int q = static_cast<int>((ql >> shift) & 0xF);
        if ((qh >> (8 * i)) & static_cast<uint32_t>(1u << sub)) {
            q += 16;
        }
        out[i] = __float2half(scale * static_cast<float>(q) - min);
    }
}

// decode_span decodes Span consecutive weight values at absolute k offset. The
// helper writes exactly out[0..valid-1] and the caller zeroes the rest of the
// span. Span is 8, 16, or 32 within a 32-value quantization segment, so
// each block header, scale, and min is hoisted once per span. Q4_K and Q5_K use
// vector global loads: their row byte sizes are multiples of 16 and the span
// offsets are 8 or 16-byte aligned by construction. Q4_0, Q5_0, and Q8_0 decode
// with scalar loads.
template <uint32_t DType, int Span>
__device__ __forceinline__ void decode_span(const uint8_t * row, int64_t k,
                                            int valid, __half * out) {
    if (DType == YUUKA_DTYPE_Q4_0) {
        const uint8_t * blk = row + (k / 32) * 18;
        const float d = load_f16_device(blk);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int lane = lane0 + j;
            const uint8_t packed = blk[2 + (lane & 15)];
            const int q = (lane < 16) ? (packed & 0xF) : (packed >> 4);
            out[j] = __float2half(d * static_cast<float>(q - 8));
        }
    } else if (DType == YUUKA_DTYPE_Q5_0) {
        const uint8_t * blk = row + (k / 32) * 22;
        const float d = load_f16_device(blk);
        const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                            (static_cast<uint32_t>(blk[3]) << 8) |
                            (static_cast<uint32_t>(blk[4]) << 16) |
                            (static_cast<uint32_t>(blk[5]) << 24);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int lane = lane0 + j;
            const uint8_t packed = blk[6 + (lane & 15)];
            const int high =
                (lane < 16) ? static_cast<int>(((qh >> lane) << 4) & 0x10)
                            : static_cast<int>((qh >> ((lane & 15) + 12)) & 0x10);
            const int q = (lane < 16 ? (packed & 0xF) : (packed >> 4)) | high;
            out[j] = __float2half(d * static_cast<float>(q - 16));
        }
    } else if (DType == YUUKA_DTYPE_Q8_0) {
        const uint8_t * blk = row + (k / 32) * 34;
        const float d = load_f16_device(blk);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int8_t q = static_cast<int8_t>(blk[2 + lane0 + j]);
            out[j] = __float2half(d * static_cast<float>(q));
        }
    } else if (DType == YUUKA_DTYPE_Q4_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t * blk = row + (k / 256) * 144;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float scale = d * static_cast<float>(sc);
        const float min = dm * static_cast<float>(m);
        const uint8_t * qs = blk + 16 + (sub / 2) * 32 + within;
        const bool high_nibble = (sub & 1) != 0;
        if (valid == Span) {
            if constexpr (Span >= 16) {
                #pragma unroll
                for (int offset = 0; offset < Span; offset += 16) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(qs + offset);
                    write_q4k_word(packed.x, high_nibble, scale, min, out + offset);
                    write_q4k_word(packed.y, high_nibble, scale, min, out + offset + 4);
                    write_q4k_word(packed.z, high_nibble, scale, min, out + offset + 8);
                    write_q4k_word(packed.w, high_nibble, scale, min, out + offset + 12);
                }
            } else {
                const uint2 packed = *reinterpret_cast<const uint2 *>(qs);
                write_q4k_word(packed.x, high_nibble, scale, min, out);
                write_q4k_word(packed.y, high_nibble, scale, min, out + 4);
            }
            return;
        }
        for (int j = 0; j < valid; ++j) {
            const int q = high_nibble ? static_cast<int>(qs[j] >> 4)
                                      : static_cast<int>(qs[j] & 0xF);
            out[j] = __float2half(scale * static_cast<float>(q) - min);
        }
    } else if (DType == YUUKA_DTYPE_Q5_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t * blk = row + (k / 256) * 176;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float scale = d * static_cast<float>(sc);
        const float min = dm * static_cast<float>(m);
        const uint8_t * ql = blk + 48 + (sub / 2) * 32 + within;
        const uint8_t * qh = blk + 16 + within;
        const bool high_nibble = (sub & 1) != 0;
        if (valid == Span) {
            if constexpr (Span >= 16) {
                #pragma unroll
                for (int offset = 0; offset < Span; offset += 16) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(ql + offset);
                    const uint4 high = *reinterpret_cast<const uint4 *>(qh + offset);
                    write_q5k_word(packed.x, high.x, high_nibble, sub, scale, min, out + offset);
                    write_q5k_word(packed.y, high.y, high_nibble, sub, scale, min, out + offset + 4);
                    write_q5k_word(packed.z, high.z, high_nibble, sub, scale, min, out + offset + 8);
                    write_q5k_word(packed.w, high.w, high_nibble, sub, scale, min, out + offset + 12);
                }
            } else {
                const uint2 packed = *reinterpret_cast<const uint2 *>(ql);
                const uint2 high = *reinterpret_cast<const uint2 *>(qh);
                write_q5k_word(packed.x, high.x, high_nibble, sub, scale, min, out);
                write_q5k_word(packed.y, high.y, high_nibble, sub, scale, min, out + 4);
            }
            return;
        }
        for (int j = 0; j < valid; ++j) {
            int q = high_nibble ? static_cast<int>(ql[j] >> 4)
                                : static_cast<int>(ql[j] & 0xF);
            if (qh[j] & static_cast<uint8_t>(1u << sub)) {
                q += 16;
            }
            out[j] = __float2half(scale * static_cast<float>(q) - min);
        }
    }
}

// Each span stays inside one Q6_K scale group. Block starts are only
// two-byte aligned, so the packed words are assembled from 16-bit halves.
template <int Span>
__device__ __forceinline__ void decode_q6k_span(const uint8_t * blk, int j,
                                                  __half * out) {
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const int half_block = j / 128;
    const int l = j & 127;
    const uint8_t * ql = blk + half_block * 64 + (l & 63);
    const uint8_t * qh = blk + 128 + half_block * 32 + (l & 31);
    const int ql_shift = (l / 64) * 4;
    const int qh_shift = (l / 32) * 2;
    const float d = load_f16_device(blk + 208) *
                    static_cast<float>(sc[half_block * 8 + l / 16]);
    #pragma unroll
    for (int offset = 0; offset < Span; offset += 4) {
        const auto * low = reinterpret_cast<const uint16_t *>(ql + offset);
        const auto * high = reinterpret_cast<const uint16_t *>(qh + offset);
        const uint32_t ql_word = uint32_t(low[0]) | (uint32_t(low[1]) << 16);
        const uint32_t qh_word = uint32_t(high[0]) | (uint32_t(high[1]) << 16);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = static_cast<int>((ql_word >> (8 * i + ql_shift)) & 0xF) |
                          (static_cast<int>((qh_word >> (8 * i + qh_shift)) & 3) << 4);
            out[offset + i] = __float2half(d * static_cast<float>(q - 32));
        }
    }
}

template <int BlockM, int BlockK>
__device__ __forceinline__ void load_a_tile(__half * tile, const float * x,
                                            int64_t m0, int64_t k0, int64_t rows,
                                            int64_t n_in) {
    for (int idx = threadIdx.x; idx < BlockM * BlockK; idx += kThreads) {
        const int r = idx / BlockK;
        const int c = idx % BlockK;
        const int64_t gr = m0 + r;
        const int64_t gk = k0 + c;
        const float value = (gr < rows && gk < n_in) ? x[gr * n_in + gk] : 0.0f;
        tile[r * BlockK + c] = __float2half(value);
    }
}

// load_b_tile stages a kBlockN x BlockK weight tile. Block-quantized layouts
// decode one contiguous span per thread: the block header is amortized, Q4_K and
// Q5_K use vector global loads, and no decoded span is staged in registers or
// scratch. Q6_K decodes four values per thread from one ql word and one qh word;
// the dense layouts keep the element mapping where consecutive lanes read
// consecutive row bytes, because they have no header worth hoisting above the
// per-value cost.
template <uint32_t DType, int BlockK>
__device__ __forceinline__ void load_b_tile(__half * tile, const uint8_t * w,
                                            int64_t n0, int64_t k0, int64_t n_in,
                                            int64_t n_out, uint64_t row_bytes) {
    constexpr int kThreadsPerRow = kThreads / kBlockN;
    constexpr int kSpan = BlockK / kThreadsPerRow;
    constexpr int kRowsPerPass = kThreads / kThreadsPerRow;
    static_assert(BlockK % kThreadsPerRow == 0, "span must divide the K step");
    static_assert(kSpan == 8 || kSpan == 16, "span decoder supports 8 or 16 values");
    if (DType == YUUKA_DTYPE_F32 || DType == YUUKA_DTYPE_F16) {
        for (int idx = threadIdx.x; idx < kBlockN * BlockK; idx += kThreads) {
            const int s = idx / BlockK;
            const int c = idx % BlockK;
            const int64_t gn = n0 + s;
            const int64_t gk = k0 + c;
            float value = 0.0f;
            if (gn < n_out && gk < n_in) {
                const uint8_t * row = w + gn * row_bytes;
                if (DType == YUUKA_DTYPE_F32) {
                    value = reinterpret_cast<const float *>(row)[gk];
                } else {
                    value = load_f16_device(row + gk * 2);
                }
            }
            tile[s * BlockK + c] = __float2half(value);
        }
        return;
    }
    if (DType == YUUKA_DTYPE_Q6_K) {
        static_assert(BlockK == 32, "the Q6_K span covers 32 K values per row");
        // Eight lanes cover one row and four values per lane, so a pass stages
        // 32 rows.
        const int row_in_group = threadIdx.x / 8;
        const int col0 = (threadIdx.x % 8) * 4;
        for (int r = row_in_group; r < kBlockN; r += kThreads / 8) {
            __half * dst = tile + r * BlockK + col0;
            const int64_t gn = n0 + r;
            const int64_t gk = k0 + col0;
            const bool in_range = gn < n_out && gk < n_in;
            const int64_t remaining = in_range ? n_in - gk : 0;
            const int valid =
                in_range ? static_cast<int>(remaining < 4 ? remaining : 4) : 0;
            const uint8_t * row = w + gn * row_bytes;
            if (valid == 4) {
                decode_q6k_span<4>(row + (gk / 256) * 210, static_cast<int>(gk % 256), dst);
                continue;
            }
            for (int j = 0; j < valid; ++j) {
                const int64_t col = gk + j;
                dst[j] = __float2half(
                    q6k_value(row + (col / 256) * 210, static_cast<int>(col % 256)));
            }
            for (int j = valid; j < 4; ++j) {
                dst[j] = __float2half(0.0f);
            }
        }
        return;
    }
    const int row_in_group = threadIdx.x / kThreadsPerRow;
    const int col0 = (threadIdx.x % kThreadsPerRow) * kSpan;
    for (int r = row_in_group; r < kBlockN; r += kRowsPerPass) {
        __half * dst = tile + r * BlockK + col0;
        const int64_t gn = n0 + r;
        const int64_t gk = k0 + col0;
        const bool in_range = gn < n_out && gk < n_in;
        const int64_t remaining = in_range ? n_in - gk : 0;
        const int valid =
            in_range ? static_cast<int>(remaining < kSpan ? remaining : kSpan) : 0;
        if (valid == kSpan) {
            decode_span<DType, kSpan>(w + gn * row_bytes, gk, kSpan, dst);
            continue;
        }
        if (valid > 0) {
            decode_span<DType, kSpan>(w + gn * row_bytes, gk, valid, dst);
        }
        for (int j = valid; j < kSpan; ++j) {
            dst[j] = __float2half(0.0f);
        }
    }
}

// gemm_wmma_kernel computes a BlockM x 64 output tile with eight waves. Each
// wave owns one A fragment and NFragments B fragments; the 32-row variant keeps
// one fragment per wave, and the 64-row variant reuses one decoded B tile
// across two accumulator fragments. A and B staging is double buffered over a
// BlockK-wide K step: the next slice loads while the current slice feeds WMMA.
// The C staging tile aliases the A/B staging memory after the K loop.
template <uint32_t DType, int BlockM, int NFragments, int BlockK>
__global__ void gemm_wmma_kernel(float * out, const float * x, const uint8_t * w,
                                 int64_t n_in, int64_t n_out, int64_t rows,
                                 uint64_t row_bytes) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    constexpr int kWaveCols = kBlockN / 16 / NFragments;
    const int wm = warp / kWaveCols;
    const int wn = warp % kWaveCols;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BlockM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) unsigned char staging[
        2 * (BlockM + kBlockN) * BlockK * sizeof(__half)];
    __half * a_tile = reinterpret_cast<__half *>(staging);
    __half * b_tile = a_tile + 2 * BlockM * BlockK;
    float * c_tile = reinterpret_cast<float *>(staging);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag1;
    rocwmma::fill_fragment(c_frag0, 0.0f);
    if constexpr (NFragments == 2) {
        rocwmma::fill_fragment(c_frag1, 0.0f);
    }

    load_a_tile<BlockM, BlockK>(a_tile, x, m0, 0, rows, n_in);
    load_b_tile<DType, BlockK>(b_tile, w, n0, 0, n_in, n_out, row_bytes);
    __syncthreads();

    int current = 0;
    for (int64_t k0 = 0; k0 < n_in; k0 += BlockK) {
        const int next = current ^ 1;
        if (k0 + BlockK < n_in) {
            load_a_tile<BlockM, BlockK>(a_tile + next * BlockM * BlockK, x, m0,
                                        k0 + BlockK, rows, n_in);
            load_b_tile<DType, BlockK>(b_tile + next * kBlockN * BlockK, w, n0,
                                       k0 + BlockK, n_in, n_out, row_bytes);
        }
        const __half * a_cur = a_tile + current * BlockM * BlockK;
        const __half * b_cur = b_tile + current * kBlockN * BlockK;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag1;
        for (int kk = 0; kk < BlockK; kk += 16) {
            rocwmma::load_matrix_sync(a_frag, a_cur + (wm * 16) * BlockK + kk,
                                      BlockK);
            rocwmma::load_matrix_sync(
                b_frag0, b_cur + (wn * NFragments * 16) * BlockK + kk, BlockK);
            rocwmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
            if constexpr (NFragments == 2) {
                rocwmma::load_matrix_sync(
                    b_frag1, b_cur + (wn * NFragments * 16 + 16) * BlockK + kk,
                    BlockK);
                rocwmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
            }
        }
        __syncthreads();
        current = next;
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16),
                               c_frag0, kBlockN, rocwmma::mem_row_major);
    if constexpr (NFragments == 2) {
        rocwmma::store_matrix_sync(
            c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16 + 16), c_frag1,
            kBlockN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BlockM * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

// gemm_wmma_k32_kernel is the 32x64 tile with a 32-wide K step and one
// accumulator fragment per wave. It stays the measured choice for small row
// counts, where the 64-row tile wastes rows on partial M tiles.
template <uint32_t DType>
__global__ void gemm_wmma_k32_kernel(float * out, const float * x,
                                     const uint8_t * w, int64_t n_in,
                                     int64_t n_out, int64_t rows,
                                     uint64_t row_bytes) {
    constexpr int kM32 = 32;
    constexpr int kK32 = 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp / (kBlockN / 16);
    const int wn = warp % (kBlockN / 16);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kM32;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) __half a_tile[kM32 * kK32];
    __shared__ __align__(16) __half b_tile[kBlockN * kK32];
    __shared__ float c_tile[kM32 * kBlockN];

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (int64_t k0 = 0; k0 < n_in; k0 += kK32) {
        load_a_tile<kM32, kK32>(a_tile, x, m0, k0, rows, n_in);
        load_b_tile<DType, kK32>(b_tile, w, n0, k0, n_in, n_out, row_bytes);
        __syncthreads();
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag;
        for (int kk = 0; kk < kK32; kk += 16) {
            if (k0 + kk >= n_in) {
                break;
            }
            rocwmma::load_matrix_sync(a_frag, a_tile + (wm * 16) * kK32 + kk, kK32);
            rocwmma::load_matrix_sync(b_frag, b_tile + (wn * 16) * kK32 + kk, kK32);
            rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + wn * 16, c_frag,
                               kBlockN, rocwmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < kM32 * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

// gemm_wmma_swiglu_kernel computes two GEMMs that share the same activation
// tile and writes silu(gate) * up in one epilogue. Both weight matrices have
// the same shape and dtype. The C staging tile aliases the A/B staging memory
// after the K loop, and no gate or up intermediate is written to global memory.
template <uint32_t DType, int BlockM, int NFragments>
__global__ void gemm_wmma_swiglu_kernel(float * out, const float * x,
                                        const uint8_t * w_gate,
                                        const uint8_t * w_up, int64_t n_in,
                                        int64_t n_out, int64_t rows,
                                        uint64_t row_bytes) {
    constexpr int kK = kFusedBlockK;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    constexpr int kWaveCols = kBlockN / 16 / NFragments;
    const int wm = warp / kWaveCols;
    const int wn = warp % kWaveCols;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BlockM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) unsigned char staging[
        2 * (BlockM + 2 * kBlockN) * kK * sizeof(__half)];
    __half * a_tile = reinterpret_cast<__half *>(staging);
    __half * gate_tile = a_tile + 2 * BlockM * kK;
    __half * up_tile = gate_tile + 2 * kBlockN * kK;
    float * c_tile = reinterpret_cast<float *>(staging);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> gate0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> gate1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> up0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> up1;
    rocwmma::fill_fragment(gate0, 0.0f);
    rocwmma::fill_fragment(up0, 0.0f);
    if constexpr (NFragments == 2) {
        rocwmma::fill_fragment(gate1, 0.0f);
        rocwmma::fill_fragment(up1, 0.0f);
    }

    load_a_tile<BlockM, kK>(a_tile, x, m0, 0, rows, n_in);
    load_b_tile<DType, kK>(gate_tile, w_gate, n0, 0, n_in, n_out, row_bytes);
    load_b_tile<DType, kK>(up_tile, w_up, n0, 0, n_in, n_out, row_bytes);
    __syncthreads();

    int current = 0;
    for (int64_t k0 = 0; k0 < n_in; k0 += kK) {
        const int next = current ^ 1;
        if (k0 + kK < n_in) {
            load_a_tile<BlockM, kK>(a_tile + next * BlockM * kK, x, m0,
                                    k0 + kK, rows, n_in);
            load_b_tile<DType, kK>(gate_tile + next * kBlockN * kK, w_gate, n0,
                                   k0 + kK, n_in, n_out, row_bytes);
            load_b_tile<DType, kK>(up_tile + next * kBlockN * kK, w_up, n0,
                                   k0 + kK, n_in, n_out, row_bytes);
        }
        const __half * a_cur = a_tile + current * BlockM * kK;
        const __half * g_cur = gate_tile + current * kBlockN * kK;
        const __half * u_cur = up_tile + current * kBlockN * kK;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> g_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> u_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> g_frag1;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> u_frag1;
        for (int kk = 0; kk < kK; kk += 16) {
            rocwmma::load_matrix_sync(a_frag, a_cur + (wm * 16) * kK + kk, kK);
            rocwmma::load_matrix_sync(
                g_frag0, g_cur + (wn * NFragments * 16) * kK + kk, kK);
            rocwmma::load_matrix_sync(
                u_frag0, u_cur + (wn * NFragments * 16) * kK + kk, kK);
            rocwmma::mma_sync(gate0, a_frag, g_frag0, gate0);
            rocwmma::mma_sync(up0, a_frag, u_frag0, up0);
            if constexpr (NFragments == 2) {
                rocwmma::load_matrix_sync(
                    g_frag1, g_cur + (wn * NFragments * 16 + 16) * kK + kk, kK);
                rocwmma::load_matrix_sync(
                    u_frag1, u_cur + (wn * NFragments * 16 + 16) * kK + kk, kK);
                rocwmma::mma_sync(gate1, a_frag, g_frag1, gate1);
                rocwmma::mma_sync(up1, a_frag, u_frag1, up1);
            }
        }
        __syncthreads();
        current = next;
    }

    for (uint32_t i = 0; i < gate0.num_elements; ++i) {
        const float g = gate0.x[i];
        gate0.x[i] = (g / (1.0f + expf(-g))) * up0.x[i];
    }
    if constexpr (NFragments == 2) {
        for (uint32_t i = 0; i < gate1.num_elements; ++i) {
            const float g = gate1.x[i];
            gate1.x[i] = (g / (1.0f + expf(-g))) * up1.x[i];
        }
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16),
                               gate0, kBlockN, rocwmma::mem_row_major);
    if constexpr (NFragments == 2) {
        rocwmma::store_matrix_sync(
            c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16 + 16), gate1,
            kBlockN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BlockM * kBlockN; idx += kThreads) {
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

template <uint32_t DType, int BlockM, int NFragments, int BlockK>
void launch_wmma_tile(float * out, const float * x, const uint8_t * w,
                      int64_t n_in, int64_t n_out, int64_t rows,
                      uint64_t row_bytes, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + BlockM - 1) / BlockM));
    gemm_wmma_kernel<DType, BlockM, NFragments, BlockK>
        <<<grid, kThreads, 0, stream>>>(out, x, w, n_in, n_out, rows, row_bytes);
}

#if defined(YUUKA_DIRECT_FP16_WMMA)
constexpr int kDirectMinRows = 17;
constexpr int kDirectMaxRows = 127;

template <uint32_t DType>
bool direct_fp16_shape(const float * x, const uint8_t * w, int64_t n_in,
                       int64_t n_out, int64_t rows) {
    constexpr size_t weight_alignment = DType == YUUKA_DTYPE_Q6_K ? 2 : 16;
    return rows >= kDirectMinRows && rows <= kDirectMaxRows &&
           n_in % 256 == 0 && n_out % 128 == 0 &&
           reinterpret_cast<uintptr_t>(x) % alignof(float4) == 0 &&
           reinterpret_cast<uintptr_t>(w) % weight_alignment == 0;
}

#if defined(__gfx1151__)
using DirectOperand = _Float16 __attribute__((ext_vector_type(16)));
using DirectAccum = float __attribute__((ext_vector_type(8)));
using DirectWords = uint32_t __attribute__((ext_vector_type(8)));

// Each 16-value fragment has two planes of eight values. Each plane
// advances one 16-byte bank group per row. XOR the K span into the row
// index to spread adjacent K stores across distinct bank groups.
template <int TileRows>
__device__ __forceinline__ int direct_half8_offset(int row, int k8) {
    return k8 * TileRows + (row ^ k8);
}

template <int TileRows>
__device__ __forceinline__ DirectOperand direct_operand(const uint4 * tile,
                                                         int row, int k16, int half_wave) {
    // Match rocWMMA's GFX11 half-wave ordering. The upper lanes start
    // with the second eight K values before reading the lower half.
    const uint4 first = tile[direct_half8_offset<TileRows>(row, 2 * k16 + half_wave)];
    const uint4 second = tile[direct_half8_offset<TileRows>(row, 2 * k16 + (half_wave ^ 1))];
    const DirectWords words{first.x, first.y, first.z, first.w,
                            second.x, second.y, second.z, second.w};
    return __builtin_bit_cast(DirectOperand, words);
}

template <int TileM, int StepK>
__device__ __forceinline__ void stage_direct_fp16_inputs(
    uint4 * tile, const float * x, int64_t m0, int64_t k0, int64_t rows, int64_t n_in) {
    constexpr int kSpans = StepK / 8;
    for (int idx = threadIdx.x; idx < TileM * kSpans; idx += kThreads) {
        const int r = idx / kSpans;
        const int k8 = idx % kSpans;
        uint4 packed{};
        if (m0 + r < rows) {
            const auto * src = reinterpret_cast<const float4 *>(x + (m0 + r) * n_in + k0 + 8 * k8);
            const float4 first = src[0];
            const float4 second = src[1];
            const __half2 pair0 = __floats2half2_rn(first.x, first.y);
            const __half2 pair1 = __floats2half2_rn(first.z, first.w);
            const __half2 pair2 = __floats2half2_rn(second.x, second.y);
            const __half2 pair3 = __floats2half2_rn(second.z, second.w);
            packed = make_uint4(__builtin_bit_cast(uint32_t, pair0),
                                __builtin_bit_cast(uint32_t, pair1),
                                __builtin_bit_cast(uint32_t, pair2),
                                __builtin_bit_cast(uint32_t, pair3));
        }
        tile[direct_half8_offset<TileM>(r, k8)] = packed;
    }
}

template <uint32_t DType, int TileN, int StepK, bool Fused>
__device__ __forceinline__ void stage_direct_fp16_weights(
    uint4 * gate_tile, uint4 * up_tile, const uint8_t * gate, const uint8_t * up,
    int64_t n0, int64_t k0, uint64_t row_bytes) {
    constexpr int kSpanSize = DType == YUUKA_DTYPE_Q6_K ? 16 : 32;
    constexpr int kSpans = StepK / kSpanSize;
    constexpr int kGateSpans = TileN * kSpans;
    for (int idx = threadIdx.x; idx < (Fused ? 2 : 1) * kGateSpans; idx += kThreads) {
        const bool second = Fused && idx >= kGateSpans;
        const int span = idx % kGateSpans;
        const int col = span / kSpans;
        const int slot = span % kSpans;
        const int64_t k = k0 + kSpanSize * slot;
        uint4 * dst = second ? up_tile : gate_tile;
        const uint8_t * row = (second ? up : gate) + (n0 + col) * row_bytes;
        alignas(16) __half values[kSpanSize];
        if constexpr (DType == YUUKA_DTYPE_Q6_K) {
            decode_q6k_span<kSpanSize>(row + (k / 256) * 210, static_cast<int>(k % 256), values);
        } else {
            decode_span<DType, kSpanSize>(row, k, kSpanSize, values);
        }
        constexpr int kHalfVectors = kSpanSize / 8;
        #pragma unroll
        for (int block = 0; block < kHalfVectors; ++block) {
            uint4 packed;
            __builtin_memcpy(&packed, values + 8 * block, sizeof(packed));
            dst[direct_half8_offset<TileN>(col, slot * kHalfVectors + block)] = packed;
        }
    }
}
#endif

template <uint32_t DType, int TileM, int TileN, int StepK, bool Fused>
__global__ void gemm_wmma_direct_kernel(
    float * out, const float * x, const uint8_t * gate, const uint8_t * up,
    int64_t n_in, int64_t n_out, int64_t rows, uint64_t row_bytes) {
#if defined(__gfx1151__)
    constexpr int kChannelWaves = TileN / 16;
    constexpr int kTokenWaves = (kThreads / 32) / kChannelWaves;
    constexpr int kTokenFrags = TileM / (16 * kTokenWaves);
    constexpr int kWeightVectors = TileN * StepK / 8;
    const int wave = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int m_wave = (wave / kChannelWaves) * 16 * kTokenFrags;
    const int n_wave = (wave % kChannelWaves) * 16;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * TileM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * TileN;

    __shared__ uint4 weights[(Fused ? 2 : 1) * kWeightVectors];
    __shared__ uint4 inputs[TileM * StepK / 8];
    uint4 * up_tile = Fused ? weights + kWeightVectors : nullptr;
    DirectAccum gate_acc[kTokenFrags]{};
    DirectAccum up_acc[kTokenFrags]{};
    for (int64_t k = 0; k < n_in; k += StepK) {
        stage_direct_fp16_inputs<TileM, StepK>(inputs, x, m0, k, rows, n_in);
        stage_direct_fp16_weights<DType, TileN, StepK, Fused>(
            weights, up_tile, gate, up, n0, k, row_bytes);
        __syncthreads();
        #pragma unroll
        for (int k16 = 0; k16 < StepK / 16; ++k16) {
            const auto b_gate = direct_operand<TileN>(weights, n_wave + (lane & 15), k16, lane >> 4);
            DirectOperand b_up{};
            if constexpr (Fused) {
                b_up = direct_operand<TileN>(up_tile, n_wave + (lane & 15), k16, lane >> 4);
            }
            #pragma unroll
            for (int t = 0; t < kTokenFrags; ++t) {
                const auto a = direct_operand<TileM>(inputs, m_wave + t * 16 + (lane & 15), k16, lane >> 4);
                gate_acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_gate, gate_acc[t]);
                if constexpr (Fused) {
                    up_acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_up, up_acc[t]);
                }
            }
        }
        __syncthreads();
    }

    const int64_t gn = n0 + n_wave + (lane & 15);
    #pragma unroll
    for (int t = 0; t < kTokenFrags; ++t) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int64_t gm = m0 + m_wave + t * 16 + 2 * e + (lane >> 4);
            if (gm < rows) {
                float value = gate_acc[t][e];
                if constexpr (Fused) {
                    value = (value / (1.0f + expf(-value))) * up_acc[t][e];
                }
                out[gm * n_out + gn] = value;
            }
        }
    }
#else
    (void)out;
    (void)x;
    (void)gate;
    (void)up;
    (void)n_in;
    (void)n_out;
    (void)rows;
    (void)row_bytes;
#endif
}

template <uint32_t DType, int TileM, int TileN, int StepK, bool Fused>
void launch_wmma_direct_tile(float * out, const float * x,
                             const uint8_t * gate, const uint8_t * up,
                             int64_t n_in, int64_t n_out, int64_t rows,
                             uint64_t row_bytes, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>(n_out / TileN),
                    static_cast<unsigned>((rows + TileM - 1) / TileM));
    gemm_wmma_direct_kernel<DType, TileM, TileN, StepK, Fused>
        <<<grid, kThreads, 0, stream>>>(out, x, gate, up, n_in, n_out, rows, row_bytes);
}

template <uint32_t DType, bool Fused>
bool launch_wmma_direct(float * out, const float * x, const uint8_t * gate,
                        const uint8_t * up, int64_t n_in, int64_t n_out, int64_t rows,
                        uint64_t row_bytes, hipStream_t stream) {
    if (!direct_fp16_shape<DType>(x, gate, n_in, n_out, rows) ||
        (Fused && !direct_fp16_shape<DType>(x, up, n_in, n_out, rows))) {
        return false;
    }
    // A narrower fused tile keeps the two accumulator sets and both
    // weight matrices resident while doubling the fused K step.
    constexpr int kTileN = Fused ? 64 : 128;
    if (rows <= 32) {
        launch_wmma_direct_tile<DType, 32, kTileN, 64, Fused>(
            out, x, gate, up, n_in, n_out, rows, row_bytes, stream);
    } else {
        launch_wmma_direct_tile<DType, 64, kTileN, 64, Fused>(
            out, x, gate, up, n_in, n_out, rows, row_bytes, stream);
    }
    return true;
}
#endif

// launch_wmma_selected chooses the measured tile. The 64x64 tile with a 32-wide
// K step reuses a decoded B tile across two accumulator fragments, stages
// 16 KiB so four CTAs stay resident, and is used for wide projections at 64 or
// more rows. Tall-K projections keep the 32x64 tile: halving the output columns
// per CTA doubles the CTA count that hides the long K loop, and F16 and Q6_K
// tall-K shapes measured 40-90% slower with the 64-row tile. The 32x64 tile
// also stays for small batches, where the wasted rows of the 64-row tile
// dominate.
template <uint32_t DType>
void launch_wmma_selected(float * out, const float * x, const uint8_t * w,
                          int64_t n_in, int64_t n_out, int64_t rows,
                          uint64_t row_bytes, hipStream_t stream) {
#if defined(YUUKA_DIRECT_FP16_WMMA)
    if constexpr (DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K ||
                  DType == YUUKA_DTYPE_Q6_K) {
        if (launch_wmma_direct<DType, false>(out, x, w, nullptr, n_in, n_out,
                                             rows, row_bytes, stream)) {
            return;
        }
    }
#endif
    if (rows >= 64 && n_in <= n_out) {
        launch_wmma_tile<DType, 64, 2, 32>(out, x, w, n_in, n_out, rows,
                                           row_bytes, stream);
        return;
    }
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + 31) / 32));
    gemm_wmma_k32_kernel<DType><<<grid, kThreads, 0, stream>>>(
        out, x, w, n_in, n_out, rows, row_bytes);
}

template <uint32_t DType, int BlockM, int NFragments>
void launch_wmma_swiglu_tile(float * out, const float * x, const uint8_t * w_gate,
                             const uint8_t * w_up, int64_t n_in, int64_t n_out,
                             int64_t rows, uint64_t row_bytes,
                             hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + BlockM - 1) / BlockM));
    gemm_wmma_swiglu_kernel<DType, BlockM, NFragments>
        <<<grid, kThreads, 0, stream>>>(out, x, w_gate, w_up, n_in, n_out, rows,
                                        row_bytes);
}

// launch_wmma_swiglu_selected uses the fused 64-row tile for wide projections at
// 64 or more rows, otherwise the 32-row tile. The fused kernel keeps its
// 32-wide K step because it stages gate and up tiles as well as the activation.
template <uint32_t DType>
void launch_wmma_swiglu_selected(float * out, const float * x,
                                 const uint8_t * w_gate, const uint8_t * w_up,
                                 int64_t n_in, int64_t n_out, int64_t rows,
                                 uint64_t row_bytes, hipStream_t stream) {
#if defined(YUUKA_DIRECT_FP16_WMMA)
    if constexpr (DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K ||
                  DType == YUUKA_DTYPE_Q6_K) {
        if (launch_wmma_direct<DType, true>(out, x, w_gate, w_up, n_in, n_out,
                                            rows, row_bytes, stream)) {
            return;
        }
    }
#endif
    if (rows >= 64 && n_in <= n_out) {
        launch_wmma_swiglu_tile<DType, 64, 2>(out, x, w_gate, w_up, n_in, n_out,
                                              rows, row_bytes, stream);
    } else {
        launch_wmma_swiglu_tile<DType, 32, 1>(out, x, w_gate, w_up, n_in, n_out,
                                              rows, row_bytes, stream);
    }
}

} // namespace

namespace yuuka {

// The paired narrow kernel feeds WMMA directly from registers. The upper
// half-wave starts with the second eight K values, as in rocWMMA's GFX11
// operand layout. FP16 conversions and the K16 accumulation order match the
// existing single-matrix path.
#if defined(__gfx1151__)
using NarrowOperand = _Float16 __attribute__((ext_vector_type(16)));
using NarrowAccumulator = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void narrow_half4(NarrowOperand & operand, int group,
                                             float a, float b, float c, float d) {
    using Words = uint32_t __attribute__((ext_vector_type(8)));
    auto words = __builtin_bit_cast(Words, operand);
    words[group * 2] = __builtin_bit_cast(uint32_t, __floats2half2_rn(a, b));
    words[group * 2 + 1] = __builtin_bit_cast(uint32_t, __floats2half2_rn(c, d));
    operand = __builtin_bit_cast(NarrowOperand, words);
}

__device__ __forceinline__ NarrowOperand narrow_q8_operand(const uint8_t * digits,
                                                           float scale, int half_wave) {
    NarrowOperand operand{};
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t word = load_u32_pair(digits + 4 * (g ^ (2 * half_wave)));
        narrow_half4(operand, g,
            scale * int8_t(word), scale * int8_t(word >> 8),
            scale * int8_t(word >> 16), scale * int8_t(word >> 24));
    }
    return operand;
}
#endif

template <int Waves>
__global__ void prefill_q8_0_pair_wmma_kernel(float * out0, float * out1,
                                               const float * x, const uint8_t * w0,
                                               const uint8_t * w1, int64_t width,
                                               int64_t rank, int64_t rows) {
#if defined(__gfx1151__)
    const int lane = threadIdx.x % 32;
    const int half_wave = lane / 16;
    const int64_t row = (int64_t(blockIdx.y) * Waves + threadIdx.x / 32) * 16;
    const int64_t channel = int64_t(blockIdx.x) * 16 + lane % 16;
    const int64_t input_row = row + lane % 16;
    const int64_t weight_stride = width / 32 * 34;
    w0 += channel * weight_stride;
    w1 += channel * weight_stride;
    NarrowAccumulator result0{}, result1{};
    for (int64_t k = 0; k < width; k += 32) {
        const float scale0 = load_f16_device(w0);
        const float scale1 = load_f16_device(w1);
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            NarrowOperand activation{};
            if (input_row < rows) {
                const auto * tile = reinterpret_cast<const float4 *>(x + input_row * width + k + part * 16);
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const float4 value = tile[g ^ (2 * half_wave)];
                    narrow_half4(activation, g, value.x, value.y, value.z, value.w);
                }
            }
            const auto beta = narrow_q8_operand(w0 + 2 + part * 16, scale0, half_wave);
            const auto alpha = narrow_q8_operand(w1 + 2 + part * 16, scale1, half_wave);
            result0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(activation, beta, result0);
            result1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(activation, alpha, result1);
        }
        w0 += 34;
        w1 += 34;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t token = row + 2 * i + half_wave;
        if (token < rows) {
            out0[token * rank + channel] = result0[i];
            out1[token * rank + channel] = result1[i];
        }
    }
#else
    (void)out0;
    (void)out1;
    (void)x;
    (void)w0;
    (void)w1;
    (void)width;
    (void)rank;
    (void)rows;
#endif
}

void launch_prefill_q8_0_pair(float * out0, float * out1, const float * x,
                              const void * w0, const void * w1, uint64_t n_in,
                              uint64_t n_out, uint64_t rows, hipStream_t stream) {
    constexpr int waves = 1;
    const dim3 grid(n_out / 16, (rows + 16 * waves - 1) / (16 * waves));
    prefill_q8_0_pair_wmma_kernel<waves><<<grid, waves * 32, 0, stream>>>(
        out0, out1, x, static_cast<const uint8_t *>(w0), static_cast<const uint8_t *>(w1),
        n_in, n_out, rows);
}

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
    const auto * weights = static_cast<const uint8_t *>(w);
    const int64_t m = static_cast<int64_t>(batch);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_wmma_selected<YUUKA_DTYPE_F32>(out, x, weights, in, on, m,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_wmma_selected<YUUKA_DTYPE_F16>(out, x, weights, in, on, m,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_wmma_selected<YUUKA_DTYPE_Q4_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_wmma_selected<YUUKA_DTYPE_Q5_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_wmma_selected<YUUKA_DTYPE_Q8_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_wmma_selected<YUUKA_DTYPE_Q4_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_wmma_selected<YUUKA_DTYPE_Q5_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_wmma_selected<YUUKA_DTYPE_Q6_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        default:
            break;
    }
}

void launch_gemm_wmma_swiglu_f32out(float * out, const float * x,
                                    const void * w_gate, const void * w_up,
                                    uint64_t n_in, uint64_t n_out, uint64_t batch,
                                    uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0) {
        return;
    }
    const uint64_t row_bytes = packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * gate = static_cast<const uint8_t *>(w_gate);
    const auto * up = static_cast<const uint8_t *>(w_up);
    const int64_t m = static_cast<int64_t>(batch);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_F32>(out, x, gate, up, in, on,
                                                         m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_F16>(out, x, gate, up, in, on,
                                                         m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q4_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q5_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q8_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q4_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q5_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q6_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
