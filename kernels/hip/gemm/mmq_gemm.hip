#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"
#include "mmq_tile.cuh"

#include <rocwmma/rocwmma.hpp>

// Int8 MMQ prefill GEMM and its Q8_1 activation quantizer.
//
// The design follows llama.cpp's MMQ approach (MIT, commit b1b770d90) at the
// algorithmic level: activations are quantized to int8 with one FP32 scale and
// one FP32 block sum per 32 values, packed K-quant weights stay quantized and
// are decoded to int8, and the 16x16x16 int8 WMMA accumulates exact int32
// products. The 32-value weight scale is applied after each pair of 16-wide
// slices, which is exact because the int32 partial sum is exact and the scale
// is constant across the block. The Q4_K/Q5_K min term becomes a per-row
// weight minimum dotted with the activation block sum.
//
// The implementation here is independent: staging, tile shape, scale layout,
// and epilogue were written for this repository and validated against the CPU
// reference in tests/hip/prefill_test.cpp.
//
// The generic gfx1151 path was tuned on Q4_K 2560 -> 9216 at M=512.
// Its staging and K loop are organized for instruction issue:
//   - both operands are staged in fragment register order, so each fragment
//     pair is one ds_read_b128 and the staging stores are conflict free
//     (LDSBankConflict 0.0% against 35.9% for row-major staging);
//   - the K step is 128 values, which keeps the staged operands at 30 KiB and
//     therefore two blocks resident per CU (occupancy 45.6% against 23.9%);
//   - the epilogue writes the accumulators straight to global memory instead
//     of transposing through a 33 KiB LDS tile, which would break the two-block
//     residency for a store pattern that is already two 64-byte runs per warp.
// See docs/kernels.md for the contract and docs/benchmarks.md for the attribution.

namespace {

constexpr int kTileI = 64;   // output channels (weight rows)
constexpr int kTileJ = 128;  // tokens
constexpr int kStepK = 128;  // K values staged per step
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlock32 = kStepK / 32;
constexpr int64_t kDirectRows = 512;
constexpr int64_t kDirectInput = 2560;
constexpr int64_t kDirectOutput = 9216;
// The same launch grid supplies twice as many channel blocks and half as
// many token blocks as the direct layout. Remap its x half inside the kernel.
constexpr int kDirectTileI = 2 * kTileI;
constexpr int kDirectTileJ = kTileJ / 2;
constexpr int kDirectBlocksI = kDirectOutput / kDirectTileI;
constexpr int kDirectWeightStride = (kDirectInput / 256) * 144;
static_assert(kDirectRows % kTileJ == 0 && kDirectOutput % kDirectTileI == 0);
static_assert(kDirectInput % 256 == 0);

// Both operands are staged in the fragment register order of the int8 WMMA
// instead of plain rows. The layout itself lives in mmq_tile.cuh, which the
// grouped MoE expert GEMM shares; it was measured on gfx1151 against the CPU
// product (see docs/kernels.md).
using yuuka_mmq::fragment_offset;
using yuuka_mmq::kFragRows;
using yuuka_mmq::kLaneBytes;
using yuuka_mmq::kPairBytes;
using yuuka_mmq::kSpan;
using yuuka_mmq::pack4;
using yuuka_mmq::set_fragment;
using yuuka_mmq::stage_weights;
constexpr int kRowGroupsI = kTileI / kFragRows;  // 4
constexpr int kRowGroupsJ = kTileJ / kFragRows;  // 8
constexpr int kPairsPerStep = kStepK / 32;       // 4

// One dynamic shared-memory allocation, sliced by offset. A and B hold the
// staged int8 operands for the current K step and the scale tiles carry the
// weight and activation scales for the same step. The total stays at 30 KiB so
// that two blocks remain resident per CU.
constexpr int kABytes = kRowGroupsI * kPairsPerStep * kPairBytes;
constexpr int kBBytes = kRowGroupsJ * kPairsPerStep * kPairBytes;
constexpr int kOperandBytes = kABytes + kBBytes;
constexpr int kAScaleBytes = (kStepK / 16) * kTileI * sizeof(float);
constexpr int kBScaleBytes = kBlock32 * kTileJ * sizeof(float2);
constexpr int kAScaleOffset = kOperandBytes;
constexpr int kBScaleOffset = kAScaleOffset + kAScaleBytes;
constexpr int kSharedBytes = kBScaleOffset + kBScaleBytes;
constexpr int kDirectABytes = kDirectTileI * kStepK;
constexpr int kDirectBScaleOffset = kAScaleOffset + (kStepK / 16) * kDirectTileI * sizeof(float);
static_assert(kDirectABytes + kDirectTileJ * kStepK == kAScaleOffset);
static_assert(kDirectBScaleOffset + kBlock32 * kDirectTileJ * sizeof(float2) == kSharedBytes);
static_assert(kABytes == 8192 && kBBytes == 16384, "unexpected operand staging size");
static_assert(kSharedBytes == 30720, "unexpected shared memory budget");
static_assert(kSharedBytes <= 64 * 1024, "exceeds the gfx1151 LDS budget");

// stage_activations copies the Q8_1 activation tile and its block scales in the
// same warp-region form. Rows beyond the batch are zeroed, which also zeroes
// their scale and sum, so the epilogue masks them instead of biasing the tile.
__device__ __forceinline__ void stage_activations(int8_t * tile_b, float2 * scale_b,
                                                  const int8_t * xs, const float2 * xds,
                                                  int64_t m0, int64_t rows,
                                                  int64_t n_in, int64_t k0) {
    constexpr int kRegionsPerWarp = (kRowGroupsJ * kPairsPerStep) / kWarps;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_in_frag = lane & 15;
    const int half = lane >> 4;
    #pragma unroll
    for (int i = 0; i < kRegionsPerWarp; ++i) {
        const int region = warp * kRegionsPerWarp + i;
        const int group = region / kPairsPerStep;
        const int pair = region % kPairsPerStep;
        const int token = group * kFragRows + row_in_frag;
        const int64_t gm = m0 + token;
        const int64_t col = k0 + pair * 32 + half * kSpan;
        int8_t * dst = tile_b + fragment_offset<false, kPairsPerStep>(
                                    group, pair, row_in_frag, half * kSpan);
        uint2 low = make_uint2(0, 0);
        uint2 high = make_uint2(0, 0);
        float2 scale = make_float2(0.0f, 0.0f);
        if (gm < rows) {
            low = *reinterpret_cast<const uint2 *>(xs + gm * n_in + col);
            high = *reinterpret_cast<const uint2 *>(xs + gm * n_in + col + 16);
            scale = xds[gm * (n_in / 32) + (col / 32)];
        }
        *reinterpret_cast<int4 *>(dst) =
            make_int4(static_cast<int>(low.x), static_cast<int>(low.y),
                      static_cast<int>(high.x), static_cast<int>(high.y));
        scale_b[pair * kTileJ + token] = scale;
    }
}

template <uint32_t DType>
__device__ __forceinline__ uint32_t direct_digits(uint32_t ql, uint32_t qh,
                                                int nibble, int subblock) {
    uint32_t digits = (ql >> (4 * nibble)) & 0x0f0f0f0fu;
    if constexpr (DType == YUUKA_DTYPE_Q5_K) {
        digits |= ((qh >> subblock) & 0x01010101u) << 4;
    }
    return digits;
}

// Load the low and high nibbles of two neighboring subblocks together.
// Q5_K also shares their packed high-bit bytes.
template <uint32_t DType>
__device__ __forceinline__ void stage_direct_weights(
    int8_t * tile_a, float * scale_a, const uint8_t * w, uint64_t row_bytes,
    int64_t n0, int64_t k0) {
    constexpr bool kQ5 = DType == YUUKA_DTYPE_Q5_K;
    constexpr int kBlockBytes = kQ5 ? 176 : 144;
    constexpr int kLowOffset = kQ5 ? 48 : 16;
    for (int index = threadIdx.x; index < kDirectTileI * (kPairsPerStep / 2);
         index += kThreads) {
        const int channel = index % kDirectTileI;
        const int packed_pair = index / kDirectTileI;
        const uint8_t * row = w + (n0 + channel) * row_bytes;
        const uint8_t * block = row + (k0 / 256) * kBlockBytes;
        const int first_sub = ((k0 % 256) / 32) + packed_pair * 2;
        const auto * input = reinterpret_cast<const uint4 *>(
            block + kLowOffset + (first_sub / 2) * 32);
        const uint4 low = input[0];
        const uint4 high = input[1];
        uint4 high_bits_low{};
        uint4 high_bits_high{};
        if constexpr (kQ5) {
            const auto * bits = reinterpret_cast<const uint4 *>(block + 16);
            high_bits_low = bits[0];
            high_bits_high = bits[1];
        }
        const float scale = load_f16_device(block);
        const float minimum = -load_f16_device(block + 2);
        #pragma unroll
        for (int nibble = 0; nibble < 2; ++nibble) {
            const int pair = 2 * packed_pair + nibble;
            const int subblock = first_sub + nibble;
            auto * dst = reinterpret_cast<uint4 *>(
                tile_a + fragment_offset<true, kPairsPerStep>(channel / 16, pair, channel % 16, 0));
            dst[0] = make_uint4(
                direct_digits<DType>(low.x, high_bits_low.x, nibble, subblock),
                direct_digits<DType>(low.y, high_bits_low.y, nibble, subblock),
                direct_digits<DType>(low.z, high_bits_low.z, nibble, subblock),
                direct_digits<DType>(low.w, high_bits_low.w, nibble, subblock));
            dst[kPairBytes / (2 * sizeof(uint4))] = make_uint4(
                direct_digits<DType>(high.x, high_bits_high.x, nibble, subblock),
                direct_digits<DType>(high.y, high_bits_high.y, nibble, subblock),
                direct_digits<DType>(high.z, high_bits_high.z, nibble, subblock),
                direct_digits<DType>(high.w, high_bits_high.w, nibble, subblock));
            uint8_t sc = 0;
            uint8_t mn = 0;
            get_scale_min_k4(subblock, block + 4, sc, mn);
            scale_a[(2 * pair) * kDirectTileI + channel] = scale * static_cast<float>(sc);
            scale_a[(2 * pair + 1) * kDirectTileI + channel] = minimum * static_cast<float>(mn);
        }
    }
}

// The fixed tile has complete rows. One thread owns both 16-byte slices and
// writes the corresponding scale once. Each slice is contiguous in LDS so
// both halves of the consumer wave can load the same operand.
[[maybe_unused]] __device__ __forceinline__ void stage_direct_activations(
    int8_t * tile_b, float2 * scale_b, const int8_t * xs, const float2 * xds,
    int64_t n_in, int64_t m0, int64_t k0) {
    for (int index = threadIdx.x; index < kDirectTileJ * kPairsPerStep; index += kThreads) {
        const int token = index % kDirectTileJ;
        const int pair = index / kDirectTileJ;
        const int64_t row = m0 + token;
        const int64_t col = k0 + pair * 32;
        const auto * input = reinterpret_cast<const uint4 *>(xs + row * n_in + col);
        auto * dst = reinterpret_cast<uint4 *>(tile_b + fragment_offset<true, kPairsPerStep>(
            token / kFragRows, pair, token % kFragRows, 0));
        dst[0] = input[0];
        dst[kPairBytes / (2 * sizeof(uint4))] = input[1];
        scale_b[pair * kDirectTileJ + token] = xds[row * (n_in / 32) + col / 32];
    }
}

// quantize_q8_1_kernel writes Q8_1 activations: one warp per 128 contiguous
// values, four values per lane. Each 32-value block keeps its amax-derived FP32
// scale and the scaled integer sum, which is what the Q4_K/Q5_K minimum term
// needs. n is a multiple of 128, so a warp never crosses a row.
template <bool SwiGLU>
__global__ void quantize_q8_1_kernel(int8_t * qs, float2 * ds, const float * x,
                                     const float * up, int64_t n, int64_t warps) {
    const int64_t warp =
        static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
    if (warp >= warps) {
        return;
    }
    const int lane = threadIdx.x % 32;
    const int64_t warps_per_row = n / 128;
    const int64_t row = warp / warps_per_row;
    const int64_t col0 = (warp % warps_per_row) * 128 + lane * 4;
    const int64_t index = row * n + col0;
    float4 value = *reinterpret_cast<const float4 *>(x + index);
    if constexpr (SwiGLU) {
        const float4 u = *reinterpret_cast<const float4 *>(up + index);
        value.x = (value.x / (1.0f + expf(-value.x))) * u.x;
        value.y = (value.y / (1.0f + expf(-value.y))) * u.y;
        value.z = (value.z / (1.0f + expf(-value.z))) * u.z;
        value.w = (value.w / (1.0f + expf(-value.w))) * u.w;
    }
    float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                       fmaxf(fabsf(value.z), fabsf(value.w)));
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, offset));
    }
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.x * inv))));
    const int q1 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.y * inv))));
    const int q2 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.z * inv))));
    const int q3 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.w * inv))));
    *reinterpret_cast<int32_t *>(qs + index) = pack4(q0, q1, q2, q3);
    float sum = static_cast<float>(q0 + q1 + q2 + q3);
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        sum += __shfl_xor_sync(0xffffffffffffffffull, sum, offset);
    }
    if ((lane & 7) == 0) {
        ds[row * (n / 32) + col0 / 32] = make_float2(d, d * sum);
    }
}

// Eight waves cover 64 channels by 128 tokens in the generic layout or 128
// channels by 64 tokens in the direct layout. Each wave computes one 16-channel
// weight fragment against four 16-token fragments. The int32
// accumulator mapping was measured on gfx1151 against the CPU product:
// lane l element e holds output row 2*e + l/16 of column l%16 inside the
// 16x16 tile. Consequently one activation scale per lane (its column) and eight
// weight scales per lane (its rows) are applied per block.
template <uint32_t DType, bool Direct = false, bool FixedShape = Direct>
__global__ void gemm_mmq_kernel(float * out, const int8_t * xs, const float2 * xds,
                                const uint8_t * w, int64_t n_in, int64_t n_out,
                                int64_t rows, uint64_t row_bytes) {
#if defined(__gfx1151__)
    constexpr bool kUseDirect = Direct;
#else
    constexpr bool kUseDirect = false;
#endif
    if constexpr (kUseDirect && FixedShape) {
        n_in = kDirectInput;
        n_out = kDirectOutput;
        rows = kDirectRows;
        row_bytes = kDirectWeightStride;
    }
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t * tile_a = reinterpret_cast<int8_t *>(smem);
    int8_t * tile_b = tile_a + (kUseDirect ? kDirectABytes : kABytes);
    float * scale_a = reinterpret_cast<float *>(smem + kAScaleOffset);
    float2 * scale_b = reinterpret_cast<float2 *>(
        smem + (kUseDirect ? kDirectBScaleOffset : kBScaleOffset));

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row_base = (kUseDirect ? warp : (warp & 3)) * 16;
    const int col_base = kUseDirect ? 0 : (warp / 4) * 64;

    const unsigned direct_blocks_i = FixedShape ? kDirectBlocksI : gridDim.x / 2;
    const int64_t n0 = kUseDirect ? static_cast<int64_t>(blockIdx.x % direct_blocks_i) * kDirectTileI
                                  : static_cast<int64_t>(blockIdx.x) * kTileI;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kTileJ
                       + (kUseDirect ? (blockIdx.x / direct_blocks_i) * kDirectTileJ : 0);

    // Q6_K carries one scale per 16 values and Q4_K/Q5_K one per 32 values, so
    // the slice group that shares a scale differs by dtype.
    constexpr bool kScalePer16 = (DType == YUUKA_DTYPE_Q6_K);
    constexpr int kSlicesPerScale = kScalePer16 ? 1 : 2;

    // One A row group and four B token groups. For direct WMMA, lanes
    // separated by 16 read the same complete slice without a lane exchange.
    const int operand_lane = kUseDirect ? lane % kFragRows : lane;
    const int a_base = row_base / kFragRows * kPairsPerStep * kPairBytes
                       + operand_lane * kLaneBytes;
    int b_base[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int group = col_base / kFragRows + t;
        b_base[t] = group * kPairsPerStep * kPairBytes + operand_lane * kLaneBytes;
    }

    float sum[4][8];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            sum[t][e] = 0.0f;
        }
    }

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> c[4];

    // A 128-value K step keeps the staged operands at 30 KiB, so two blocks
    // stay resident per CU: the staging stalls of one block are covered by the
    // MMAs of the other instead of idling the CU at a barrier.
    const int64_t steps = n_in / kStepK;
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k0 = step * kStepK;
        if constexpr (kUseDirect) {
            stage_direct_weights<DType>(tile_a, scale_a, w, row_bytes, n0, k0);
            stage_direct_activations(tile_b, scale_b, xs, xds, n_in, m0, k0);
        } else {
            stage_weights<DType, kTileI, kStepK, kThreads>(tile_a, scale_a, w,
                                                          row_bytes, n0, n_out, k0);
            stage_activations(tile_b, scale_b, xs, xds, m0, rows, n_in, k0);
        }
        __syncthreads();
        #pragma unroll
        for (int pair = 0; pair < kPairsPerStep; ++pair) {
#if defined(__gfx1151__)
            if constexpr (kUseDirect) {
                static_assert(DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K);
                using Operand = int32_t __attribute__((ext_vector_type(4)));
                using Accum = int32_t __attribute__((ext_vector_type(8)));
                const int slice_offset = pair * kPairBytes;
                const Operand a0 = *reinterpret_cast<const Operand *>(
                    tile_a + a_base + slice_offset);
                const Operand a1 = *reinterpret_cast<const Operand *>(
                    tile_a + a_base + slice_offset + kPairBytes / 2);
                float wd[8];
                float wm[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = row_base + 2 * e + (lane >> 4);
                    wd[e] = scale_a[(2 * pair) * kDirectTileI + row];
                    wm[e] = scale_a[(2 * pair + 1) * kDirectTileI + row];
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const auto * b = reinterpret_cast<const Operand *>(
                        tile_b + b_base[t] + slice_offset);
                    Accum accum = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a0, true, b[0], Accum{}, true);
                    accum = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a1, true, b[kPairBytes / (2 * sizeof(Operand))], accum, true);
                    const float2 dsb =
                        scale_b[pair * kDirectTileJ + col_base + t * 16 + (lane & 15)];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sum[t][e] = fmaf(wd[e] * dsb.x, static_cast<float>(accum[e]),
                                         fmaf(wm[e], dsb.y, sum[t][e]));
                    }
                }
                continue;
            }
#endif
            // One 128-bit load per operand fragment pair: .xy carries the 16
            // values of the first slice, .zw the second.
            const int4 av =
                *reinterpret_cast<const int4 *>(tile_a + a_base + pair * kPairBytes);
            int4 bv[4];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                bv[t] = *reinterpret_cast<const int4 *>(tile_b + b_base[t] +
                                                        pair * kPairBytes);
            }
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int slice = 2 * pair + s;
                if (slice % kSlicesPerScale == 0) {
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        rocwmma::fill_fragment(c[t], 0);
                    }
                }
                rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t,
                                  rocwmma::row_major> a;
                set_fragment(a, s == 0 ? av.x : av.z, s == 0 ? av.y : av.w);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t,
                                      rocwmma::col_major> b;
                    set_fragment(b, s == 0 ? bv[t].x : bv[t].z,
                                 s == 0 ? bv[t].y : bv[t].w);
                    rocwmma::mma_sync(c[t], a, b, c[t]);
                }
                if ((slice + 1) % kSlicesPerScale != 0) {
                    continue;
                }
                // The weight scales belong to this lane's eight rows, so they
                // are loaded once per scale block and reused by all four
                // accumulator tiles. Q4_K/Q5_K additionally carry a per-row
                // minimum that the activation block sum multiplies; Q6_K has
                // no minimum.
                const int block = slice / kSlicesPerScale;
                const int b32 = kScalePer16 ? (block >> 1) : block;
                const int slot0 = kScalePer16 ? block : 2 * block;
                float wd[8];
                float wm[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = row_base + 2 * e + (lane >> 4);
                    wd[e] = scale_a[slot0 * kTileI + row];
                    wm[e] = kScalePer16 ? 0.0f : scale_a[(slot0 + 1) * kTileI + row];
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float2 dsb =
                        scale_b[b32 * kTileJ + col_base + t * 16 + (lane & 15)];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sum[t][e] = fmaf(wd[e] * dsb.x, static_cast<float>(c[t].x[e]),
                                         fmaf(wm[e], dsb.y, sum[t][e]));
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write the accumulators straight to global memory. For a fixed fragment
    // the warp's 16 low lanes cover 16 consecutive tokens of one output
    // channel, so each store instruction still covers two 64-byte runs; a
    // shared-memory transpose would cost 33 KiB of the LDS budget that keeps
    // two blocks resident per CU.
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int64_t gm = m0 + col_base + t * 16 + (lane & 15);
        if (!kUseDirect && gm >= rows) {
            continue;
        }
        float * dst = out + gm * n_out;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int64_t gn = n0 + row_base + 2 * e + (lane >> 4);
            if (kUseDirect || gn < n_out) {
                dst[gn] = sum[t][e];
            }
        }
    }
}

template <uint32_t DType, bool Direct = false, bool FixedShape = Direct>
void launch_mmq_tile(float * out, const int8_t * xs, const float2 * xds,
                     const uint8_t * w, int64_t n_in, int64_t n_out, int64_t rows,
                     uint64_t row_bytes, hipStream_t stream) {
    static const hipError_t attribute = hipFuncSetAttribute(
        reinterpret_cast<const void *>(&gemm_mmq_kernel<DType, Direct, FixedShape>),
        hipFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    // A rejected attribute makes the launch below fail with a clear device
    // error through the caller's hipGetLastError check; nothing runs silently.
    (void)attribute;
    const dim3 grid(static_cast<unsigned>((n_out + kTileI - 1) / kTileI),
                    static_cast<unsigned>((rows + kTileJ - 1) / kTileJ));
    gemm_mmq_kernel<DType, Direct, FixedShape><<<grid, kThreads, kSharedBytes, stream>>>(
        out, xs, xds, w, n_in, n_out, rows, row_bytes);
}

// The vectorized direct path was measured for these model projections on
// gfx1151. Every selected cell fills the complete channel and token tiles.
bool measured_direct_shape(uint32_t dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    if (rows != 128 && rows != 512 && rows != 2048) {
        return false;
    }
    if (dtype == YUUKA_DTYPE_Q4_K) {
        if (n_in == 2560) {
            return n_out == 1024 || n_out == 4096 || n_out == 8192 || n_out == 9216;
        }
        return n_out == 2560 && (n_in == 4096 || n_in == 9216);
    }
    if (dtype == YUUKA_DTYPE_Q5_K) {
        // The qwen35moe prompt GEMMs, screened at the release chunk size:
        // Q/QKV and the recurrent gate, K/V plus the shared expert gate/up,
        // and the attention output. Other row counts keep the generic tile.
        if (rows == 512) {
            if (n_in == 2048 && (n_out == 8192 || n_out == 4096 || n_out == 512)) {
                return true;
            }
            if (n_in == 4096 && n_out == 2048) {
                return true;
            }
        }
        return (n_in == 2560 && n_out == 8192) || (n_in == 4096 && n_out == 2560);
    }
    return false;
}

} // namespace

namespace yuuka {

void launch_quantize_q8_1(void * qs, void * ds, const float * x, uint64_t n,
                          uint64_t batch, hipStream_t stream) {
    if (n == 0 || batch == 0 || n % 128 != 0) {
        return;
    }
    const int64_t warps = static_cast<int64_t>((n * batch) / 128);
    const int64_t blocks = (warps + kWarps - 1) / kWarps;
    quantize_q8_1_kernel<false><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<int8_t *>(qs), static_cast<float2 *>(ds), x, nullptr,
        static_cast<int64_t>(n), warps);
}

void launch_silu_mul_quantize_q8_1(void * qs, void * ds, const float * gate,
                                  const float * up, uint64_t n, uint64_t batch,
                                  hipStream_t stream) {
    const int64_t warps = static_cast<int64_t>((n * batch) / 128);
    const int64_t blocks = (warps + kWarps - 1) / kWarps;
    quantize_q8_1_kernel<true><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<int8_t *>(qs), static_cast<float2 *>(ds), gate, up,
        static_cast<int64_t>(n), warps);
}

void launch_gemm_mmq_f32out(float * out, const void * qs, const void * ds,
                            const void * w, uint64_t n_in, uint64_t n_out,
                            uint64_t batch, uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0 || n_in % kStepK != 0) {
        return;
    }
    const uint64_t block_bytes =
        dtype == YUUKA_DTYPE_Q4_K    ? 144u
        : dtype == YUUKA_DTYPE_Q5_K  ? 176u
        : dtype == YUUKA_DTYPE_Q6_K  ? 210u
                                     : 0u;
    if (block_bytes == 0) {
        return;
    }
    const uint64_t row_bytes = (n_in / 256) * block_bytes;
    const auto * activations = static_cast<const int8_t *>(qs);
    const auto * scales = static_cast<const float2 *>(ds);
    const auto * weights = static_cast<const uint8_t *>(w);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    const int64_t m = static_cast<int64_t>(batch);
    const bool direct = measured_direct_shape(dtype, in, on, m) &&
                        reinterpret_cast<uintptr_t>(qs) % alignof(uint4) == 0 &&
                        reinterpret_cast<uintptr_t>(w) % alignof(uint4) == 0;
    switch (dtype) {
        case YUUKA_DTYPE_Q4_K:
            if (direct && in == kDirectInput && on == kDirectOutput && m == kDirectRows) {
                launch_mmq_tile<YUUKA_DTYPE_Q4_K, true>(
                    out, activations, scales, weights, in, on, m, row_bytes, stream);
                break;
            }
            if (direct) {
                launch_mmq_tile<YUUKA_DTYPE_Q4_K, true, false>(
                    out, activations, scales, weights, in, on, m, row_bytes, stream);
                break;
            }
            launch_mmq_tile<YUUKA_DTYPE_Q4_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            if (direct) {
                launch_mmq_tile<YUUKA_DTYPE_Q5_K, true, false>(
                    out, activations, scales, weights, in, on, m, row_bytes, stream);
                break;
            }
            launch_mmq_tile<YUUKA_DTYPE_Q5_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_mmq_tile<YUUKA_DTYPE_Q6_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
