#pragma once

// Shared operand staging for the int8 MMQ GEMM family.
//
// The dense prefill GEMM (kernels/hip/gemm/mmq_gemm.hip) and the grouped MoE
// expert GEMM (kernels/hip/moe/moe_mmq.hip) stage the same rocWMMA fragment
// layouts and decode the same K-quant weight blocks, so those helpers live
// here instead of in either kernel. The MoE kernel adds its own gather and
// scatter around them, and passes its IQ span decoder through the stage_weights
// hook; the layout itself must not diverge, because both read the staged bytes
// with the same fragment loads.
//
// The layout was measured on gfx1151 against the CPU product:
//   - a 16x16 int8 fragment is 256 bytes, loaded as eight bytes per lane,
//     where lane l holds row l%16 of the 8-value K half l/16;
//   - one 32-value K pair per 16-row group is 512 bytes, and a lane's 16-byte
//     slot holds both 8-value K halves of one row, so one ds_read_b128 feeds
//     two WMMA slices.

#include "../common.cuh"
#include "../quant.cuh"

namespace yuuka_mmq {

constexpr int kFragRows = 16;
constexpr int kLaneBytes = 16;               // two 8-byte slice halves per lane
constexpr int kPairBytes = 32 * kLaneBytes;  // one 16-row group x 32 K values
constexpr int kSpan = 8;                     // weight values decoded per call

// fragment_offset locates the eight staged values that start at k_in_pair (a
// multiple of eight) inside one 32-value K pair of a 16-row fragment group.
// The rocWMMA path packs both slices into a lane's slot. The direct path keeps
// each slice contiguous per row so both wave halves can read it.
template <bool Direct, int kPairsPerStep>
__device__ __forceinline__ int fragment_offset(int group, int pair, int row_in_frag,
                                               int k_in_pair) {
    const int half = (k_in_pair >> 3) & 1;
    const int slice = k_in_pair >> 4;
    if constexpr (Direct) {
        return (group * kPairsPerStep + pair) * kPairBytes
               + slice * (kPairBytes / 2) + row_in_frag * kLaneBytes + half * 8;
    }
    const int lane = row_in_frag + kFragRows * half;
    return (group * kPairsPerStep + pair) * kPairBytes + lane * kLaneBytes + slice * 8;
}

// set_fragment copies the eight staged bytes of one 16x16 operand fragment
// into a rocWMMA fragment, which is exactly what load_matrix_sync would put
// there for the same bytes.
template <typename FragT>
__device__ __forceinline__ void set_fragment(FragT & frag, int lo, int hi) {
    const int2 value = make_int2(lo, hi);
    __builtin_memcpy(&frag.x, &value, sizeof(value));
}

// pack4 packs four values in [-128, 127] into one 32-bit word in little-endian
// byte order, which is the int8 order of the staged operand tile.
__device__ __forceinline__ int32_t pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// q6k_span8 decodes eight consecutive Q6_K values starting at a multiple of 8
// inside one 256-value block. The eight values lie inside one 32-value region,
// so the ql nibble and qh bit positions are hoisted once per span and each
// 16-bit load covers two values. Block starts are only 2-byte aligned, so the
// words are assembled from 16-bit halves.
__device__ __forceinline__ int2 q6k_span8(const uint8_t * blk, int j) {
    const int n = j / 128;
    const int l = j & 127;
    const uint8_t * ql = blk + n * 64;
    const uint8_t * qh = blk + 128 + n * 32;
    const uint8_t * ql_src = ql;
    const uint8_t * qh_src = qh;
    int ql_shift = 0;
    int qh_shift = 0;
    if (l < 32) {
        ql_src = ql + l;
        qh_src = qh + l;
    } else if (l < 64) {
        ql_src = ql + l;
        qh_src = qh + (l - 32);
        qh_shift = 2;
    } else if (l < 96) {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 64);
        qh_shift = 4;
    } else {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 96);
        qh_shift = 6;
    }
    int32_t lo = 0;
    int32_t hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t ql_word =
            *reinterpret_cast<const uint16_t *>(ql_src + 2 * i);
        const uint32_t qh_word =
            *reinterpret_cast<const uint16_t *>(qh_src + 2 * i);
        const int q0 = static_cast<int>((ql_word >> ql_shift) & 0xF) |
                       (static_cast<int>((qh_word >> qh_shift) & 3) << 4);
        const int q1 = static_cast<int>((ql_word >> (8 + ql_shift)) & 0xF) |
                       (static_cast<int>((qh_word >> (8 + qh_shift)) & 3) << 4);
        const int32_t word = pack4(q0 - 32, q1 - 32, 0, 0);
        if (i < 2) {
            lo |= word << (16 * i);
        } else {
            hi |= word << (16 * (i - 2));
        }
    }
    return make_int2(lo, hi);
}

// stage_weight_span decodes eight consecutive weights of one packed row into
// int8 and returns them packed. The A tile keeps the raw quantized values, so
// the integer WMMA sees the same digits the dequantized reference path sees;
// the block scale and the Q4_K/Q5_K minimum are staged separately.
template <uint32_t DType, int kTileI, int kStepK>
__device__ __forceinline__ int2 stage_weight_span(const uint8_t * row, int64_t k,
                                                  float * scale_a, int row_index) {
    if (DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const int block_bytes = (DType == YUUKA_DTYPE_Q4_K) ? 144 : 176;
        const uint8_t * blk = row + (k / 256) * block_bytes;
        const uint8_t * ql_base = blk + (DType == YUUKA_DTYPE_Q4_K ? 16 : 48);
        const uint8_t * ql = ql_base + (sub / 2) * 32 + within;
        const int shift = (sub & 1) ? 4 : 0;
        const uint2 packed_ql = *reinterpret_cast<const uint2 *>(ql);
        // The eight digits of a span are the low (or high) nibbles of eight
        // consecutive ql bytes, so one mask (plus a shift for the high nibbles)
        // expands all eight at once.
        uint32_t lo = (packed_ql.x >> shift) & 0x0F0F0F0Fu;
        uint32_t hi = (packed_ql.y >> shift) & 0x0F0F0F0Fu;
        if (DType == YUUKA_DTYPE_Q5_K) {
            // The fifth bit of each digit is one bit of the eight qh bytes; it
            // moves to bit 4 of the matching output byte.
            const uint2 packed_qh = *reinterpret_cast<const uint2 *>(blk + 16 + within);
            const uint32_t bit = 0x01010101u << sub;
            lo |= sub < 4 ? (packed_qh.x & bit) << (4 - sub)
                          : (packed_qh.x & bit) >> (sub - 4);
            hi |= sub < 4 ? (packed_qh.y & bit) << (4 - sub)
                          : (packed_qh.y & bit) >> (sub - 4);
        }
        if (within == 0) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(sub, blk + 4, sc, m);
            const float d = load_f16_device(blk);
            const float dm = load_f16_device(blk + 2);
            // The scale tiles are indexed per staged step, not per 256-value
            // quant block, because kStepK is smaller than a block.
            const int slot = static_cast<int>((k % kStepK) / 32);
            scale_a[(2 * slot) * kTileI + row_index] = d * static_cast<float>(sc);
            scale_a[(2 * slot + 1) * kTileI + row_index] =
                -(dm * static_cast<float>(m));
        }
        return make_int2(static_cast<int32_t>(lo), static_cast<int32_t>(hi));
    }
    // Q6_K: one scale per 16 values and no minimum.
    const int j = static_cast<int>(k % 256);
    const uint8_t * blk = row + (k / 256) * 210;
    const int2 value = q6k_span8(blk, j);
    if ((j & 15) == 0) {
        const int n = j / 128;
        const int l = j & 127;
        int sidx = 0;
        if (l < 32) {
            sidx = l / 16;
        } else if (l < 64) {
            sidx = (l - 32) / 16 + 2;
        } else if (l < 96) {
            sidx = (l - 64) / 16 + 4;
        } else {
            sidx = (l - 96) / 16 + 6;
        }
        const int8_t sc = reinterpret_cast<const int8_t *>(blk + 192)[n * 8 + sidx];
        const int slot = static_cast<int>((k % kStepK) / 16);
        scale_a[slot * kTileI + row_index] =
            load_f16_device(blk + 208) * static_cast<float>(sc);
    }
    return value;
}

// stage_weights_with decodes the kTileI x kStepK weight tile in fragment order.
// Each warp owns whole 512-byte fragment-pair regions and each lane fills one
// 16-byte slot of a region, so a warp's stores cover one contiguous region and
// never conflict in the LDS banks. Rows past n_out are zeroed so a partially
// filled channel tile contributes nothing.
//
// span(row, k, scale_a, row_index) decodes the eight values that start at a
// multiple of eight inside a packed row and returns them packed as int8, so the
// K-quant and the IQ codebooks share this mapping and differ only in their span
// decoder and scale-slot rule. The routing itself must not be restated per
// decode family; see the header comment.
template <int kTileI, int kStepK, int kThreads, typename SpanFn>
__device__ __forceinline__ void stage_weights_with(int8_t * tile_a, float * scale_a,
                                                   const uint8_t * w,
                                                   uint64_t row_bytes, int64_t n0,
                                                   int64_t n_out, int64_t k0,
                                                   SpanFn span) {
    constexpr int kWarps = kThreads / 32;
    constexpr int kPairsPerStep = kStepK / 32;
    constexpr int kRowGroups = kTileI / kFragRows;
    constexpr int kRegions = kRowGroups * kPairsPerStep;
    constexpr int kRegionsPerWarp = (kRegions + kWarps - 1) / kWarps;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_in_frag = lane & 15;
    const int half = lane >> 4;
    #pragma unroll
    for (int i = 0; i < kRegionsPerWarp; ++i) {
        const int region = warp * kRegionsPerWarp + i;
        if (region >= kRegions) {
            continue;
        }
        const int group = region / kPairsPerStep;
        const int pair = region % kPairsPerStep;
        int8_t * dst = tile_a + fragment_offset<false, kPairsPerStep>(
                                    group, pair, row_in_frag, half * kSpan);
        const int64_t gn = n0 + group * kFragRows + row_in_frag;
        if (gn >= n_out) {
            *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            continue;
        }
        const uint8_t * row = w + gn * row_bytes;
        // The two 16-value slices of the pair fill bytes 0..7 and 8..15.
        const int64_t k = k0 + pair * 32 + half * kSpan;
        const int row_index = group * kFragRows + row_in_frag;
        const int2 low = span(row, k, scale_a, row_index);
        const int2 high = span(row, k + 16, scale_a, row_index);
        *reinterpret_cast<int4 *>(dst) = make_int4(low.x, low.y, high.x, high.y);
    }
}

// stage_weights is the dense K-quant form of stage_weights_with: the span
// decoder is the shared K-quant one, so the dense GEMM keeps a single call.
template <uint32_t DType, int kTileI, int kStepK, int kThreads>
__device__ __forceinline__ void stage_weights(int8_t * tile_a, float * scale_a,
                                              const uint8_t * w, uint64_t row_bytes,
                                              int64_t n0, int64_t n_out, int64_t k0) {
    stage_weights_with<kTileI, kStepK, kThreads>(
        tile_a, scale_a, w, row_bytes, n0, n_out, k0,
        [](const uint8_t * row, int64_t k, float * scales, int row_index) {
            return stage_weight_span<DType, kTileI, kStepK>(row, k, scales, row_index);
        });
}

} // namespace yuuka_mmq
