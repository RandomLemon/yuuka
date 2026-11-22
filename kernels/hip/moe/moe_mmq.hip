#include "../common.cuh"
#include "../gemm/mmq_tile.cuh"
#include "../iq_decode.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

// Int8 WMMA (MMQ) grouped expert GEMM for the qwen35moe prefill FFN.
//
// The grouped dp4a route (moe_gate_up_q8 / moe_down_q8) is arithmetic bound at
// the assignment counts a prefill chunk produces: the expert set, 9.06 GB of
// IQ weights, costs 99.4 ms at 128 rows and 363.2 ms at 512 rows against a
// 37 ms read floor. This route keeps the Q8_1 activation tile and the int8
// WMMA of the dense MMQ GEMM (kernels/hip/gemm/mmq_tile.cuh) and replaces only
// the assignment loop: a block owns one expert and one tile of output
// channels, walks that expert's assignment bucket in kTileJ-sized tiles, and
// reuses one decoded weight tile across the whole assignment tile.
//
// The design follows llama.cpp's MMQ mul_mat_id at the algorithmic level:
// int8 WMMA, an assignment tile per weight tile, and a device-side per-expert
// loop over the bucket. The staging, tile shape, scale layout, and scatter are
// this repository's; the codebooks and their scale granularity are documented
// in docs/kernels.md section 8.1. The IQ codebooks decode through the dot8
// forms of kernels/hip/iq_decode.cuh, so the digits are the ones the dp4a route
// and the FP32 reference path see.
//
// Two properties are load bearing:
//   - the bucket counts stay on the device, so the launch is HIP-Graph
//     capturable and never synchronizes;
//   - the activation rows are gathered through the routing arrays, so the
//     kernel needs no compaction pass and no host-side permutation.

namespace {

using yuuka_mmq::fragment_offset;
using yuuka_mmq::kFragRows;
using yuuka_mmq::kLaneBytes;
using yuuka_mmq::kPairBytes;
using yuuka_mmq::kSpan;
using yuuka_mmq::set_fragment;
using yuuka_mmq::stage_weights;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// The int8 WMMA is 16 rows wide, so an assignment tile is a multiple of 16.
constexpr int kMinTileJ = 16;
constexpr int kMaxTileJ = 128;

// is_iq_codebook reports whether DType is one of the IQ codebooks the MoE
// kernels decode. They carry codebook indices instead of plain nibbles, so
// their digits come from the iq*_dot8 decoders in kernels/hip/iq_decode.cuh.
template <uint32_t DType>
constexpr bool is_iq_codebook() {
    return DType == YUUKA_DTYPE_IQ2_XXS || DType == YUUKA_DTYPE_IQ2_S ||
           DType == YUUKA_DTYPE_IQ3_S;
}

// iq_scale_per_16 reports whether a codebook carries one weight scale per 16
// values instead of one per 32. IQ2_S stores the scale of a 32-value group in
// the two nibbles of one byte, the low nibble for the first 16 values and the
// high nibble for the second, so applying its scale per 32 values would be
// wrong. IQ2_XXS and IQ3_S hold one scale per 32-value group.
template <uint32_t DType>
constexpr bool iq_scale_per_16() {
    return DType == YUUKA_DTYPE_IQ2_S;
}

// stage_iq_weights stages the kTileI x kStepK IQ weight tile. One thread owns one
// (row, 32-value group) task: it fetches that group's header once, decodes all
// thirty-two values, and writes both WMMA halves' 16-byte lane slots plus the
// two scale entries of the group. The dense K-quant staging is per lane slot
// instead, which fetches the same group header twice (once per half); here a
// group costs one header fetch per row and one codebook fetch per eight values.
template <uint32_t DType, int kTileI, int kStepK>
__device__ __forceinline__ void stage_iq_weights(int8_t * tile_a, float * scale_a,
                                                 const uint8_t * w, uint64_t row_bytes,
                                                 int64_t n0, int64_t n_out, int64_t k0) {
    constexpr int kPairsPerStep = kStepK / 32;
    constexpr int kTasks = kTileI * kPairsPerStep;
    constexpr int kTasksPerThread = (kTasks + kThreads - 1) / kThreads;
    constexpr uint64_t kBlockBytes = DType == YUUKA_DTYPE_IQ2_XXS ? 66
                                   : DType == YUUKA_DTYPE_IQ2_S   ? 82
                                                                  : 110;
    #pragma unroll
    for (int i = 0; i < kTasksPerThread; ++i) {
        const int task = threadIdx.x + i * kThreads;
        if (task >= kTasks) {
            continue;
        }
        // Threads cover rows first, so consecutive threads of a warp stage
        // consecutive rows and the group index is spread over the block.
        const int row = task % kTileI;
        const int pair = task / kTileI;
        const int group = row / kFragRows;
        const int row_in_frag = row % kFragRows;
        int8_t * low = tile_a + fragment_offset<false, kPairsPerStep>(
                                    group, pair, row_in_frag, 0);
        int8_t * high = tile_a + fragment_offset<false, kPairsPerStep>(
                                    group, pair, row_in_frag, kSpan);
        const int64_t gn = n0 + row;
        if (gn >= n_out) {
            // The planned shapes fill the channel tile, so this only guards a
            // tile the ABI rejects; a zero row contributes nothing.
            *reinterpret_cast<int4 *>(low) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4 *>(high) = make_int4(0, 0, 0, 0);
            continue;
        }
        const uint8_t * row_ptr = w + gn * row_bytes;
        const int64_t k = k0 + static_cast<int64_t>(pair) * 32;
        const int j = static_cast<int>(k % 256);
        const uint8_t * blk = row_ptr + (k / 256) * kBlockBytes;
        int32_t w8[8];
        float scale_lo = 0.0f;
        float scale_hi = 0.0f;
        if constexpr (DType == YUUKA_DTYPE_IQ2_XXS) {
            iq2xxs_dot32(blk, j, w8, scale_lo, scale_hi);
        } else if constexpr (DType == YUUKA_DTYPE_IQ2_S) {
            iq2s_dot32(blk, j, w8, scale_lo, scale_hi);
        } else {
            static_assert(DType == YUUKA_DTYPE_IQ3_S,
                          "the IQ expert staging serves IQ2_XXS, IQ2_S, and IQ3_S");
            iq3s_dot32(blk, j, w8, scale_lo, scale_hi);
        }
        // Runs 0 and 2 fill the first WMMA half's lane slot (values 0..7 and
        // 16..23) and runs 1 and 3 the second (values 8..15 and 24..31).
        *reinterpret_cast<int4 *>(low) = make_int4(w8[0], w8[1], w8[4], w8[5]);
        *reinterpret_cast<int4 *>(high) = make_int4(w8[2], w8[3], w8[6], w8[7]);
        // The scale tiles are indexed per 16-value slice of the staged K step, so
        // the two slices of the group write their own entry; a per-32 codebook
        // carries one scale and writes it to both.
        const int slice = static_cast<int>(k % kStepK) / 16;
        scale_a[slice * kTileI + row] = scale_lo;
        scale_a[(slice + 1) * kTileI + row] = scale_hi;
    }
}

// stage_weight_tile stages the weight tile with the decoder of the dtype's
// family: the shared K-quant span for Q4_K/Q5_K/Q6_K, and the IQ group staging
// above for the codebooks. Both fill the same fragment layout.
template <uint32_t DType, int kTileI, int kStepK>
__device__ __forceinline__ void stage_weight_tile(int8_t * tile_a, float * scale_a,
                                                  const uint8_t * w,
                                                  uint64_t row_bytes, int64_t n0,
                                                  int64_t n_out, int64_t k0) {
    if constexpr (is_iq_codebook<DType>()) {
        stage_iq_weights<DType, kTileI, kStepK>(tile_a, scale_a, w, row_bytes, n0, n_out,
                                                k0);
    } else {
        stage_weights<DType, kTileI, kStepK, kThreads>(tile_a, scale_a, w, row_bytes,
                                                       n0, n_out, k0);
    }
}

// Tile maps the assignment tile width to the warp decomposition. Up to 64
// assignments every warp owns one output-row fragment group of the 128-channel
// tile, so all eight warps are busy on the single token group. At 128
// assignments the block splits into two token halves of 64, halving the
// channel tile to 64 rows so a warp still holds at most four accumulator
// fragments, which is the register budget the dense GEMM measured.
template <int kTileJ>
struct Tile {
    static_assert(kTileJ >= kMinTileJ && kTileJ <= kMaxTileJ && kTileJ % kFragRows == 0,
                  "assignment tile must be a measured multiple of the WMMA width");
    static constexpr int kTokenHalves = kTileJ <= 64 ? 1 : 2;
    static constexpr int kRowGroups = kWarps / kTokenHalves;
    static constexpr int kTileI = kRowGroups * kFragRows;
    static constexpr int kTokensPerWarp = kTileJ / kTokenHalves;
    static constexpr int kTokenTiles = kTokensPerWarp / kFragRows;
    static constexpr int kRowGroupsJ = kTileJ / kFragRows;
    static_assert(kTileI % kFragRows == 0);
    static_assert(kTokensPerWarp % kFragRows == 0);
    static_assert(kTokenTiles >= 1 && kTokenTiles <= 4);
    static_assert(kRowGroups * kTokenHalves == kWarps);
    static_assert(kTokensPerWarp * kTokenHalves == kTileJ);
};

// Shared maps the tile to the dynamic shared-memory slices. The A slice holds
// the decoded weight tile, the B slice the gathered Q8_1 activation tile, and
// the scale slices the per-16/per-32 weight scales and per-32 activation
// scales of the same K step.
template <int kTileJ, int kStepK>
struct Shared {
    using T = Tile<kTileJ>;
    static constexpr int kPairsPerStep = kStepK / 32;
    static constexpr int kABytes = T::kRowGroups * kPairsPerStep * kPairBytes;
    static constexpr int kBBytes = T::kRowGroupsJ * kPairsPerStep * kPairBytes;
    static constexpr int kAScaleBytes = (kStepK / 16) * T::kTileI * sizeof(float);
    static constexpr int kBScaleBytes = kPairsPerStep * kTileJ * sizeof(float2);
    static constexpr int kAScaleOffset = kABytes + kBBytes;
    static constexpr int kBScaleOffset = kAScaleOffset + kAScaleBytes;
    static constexpr int kBytes = kBScaleOffset + kBScaleBytes;
    static_assert(kStepK % 32 == 0);
    static_assert(kBytes <= 64 * 1024, "exceeds the gfx1151 LDS budget");
};

// stage_activations_gather stages the current assignment tile in fragment
// order. The rows are indirect through the expert's bucket, which is the one
// difference from the dense GEMM: the gate/up projections read the activation
// row of assign_rows, and the down projection reads the slot of the
// assignment, because its activation row is the SwiGLU product of that slot.
// Tokens past take are zeroed together with their scale, so the epilogue skips
// them instead of biasing the tile.
template <int kTileJ, int kStepK, bool kActivationBySlot>
__device__ __forceinline__ void stage_activations_gather(
    int8_t * tile_b, float2 * scale_b, const int8_t * xs, const float2 * xds,
    const int32_t * assign_rows, const int32_t * assign_slots, int64_t bucket,
    int a0, int take, int64_t used, int64_t n_in, int64_t k0) {
    using T = Tile<kTileJ>;
    constexpr int kPairsPerStep = kStepK / 32;
    constexpr int kRegions = T::kRowGroupsJ * kPairsPerStep;
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
        const int token = group * kFragRows + row_in_frag;
        const int64_t col = k0 + pair * 32 + half * kSpan;
        int8_t * dst = tile_b + fragment_offset<false, kPairsPerStep>(
                                    group, pair, row_in_frag, half * kSpan);
        uint2 low = make_uint2(0, 0);
        uint2 high = make_uint2(0, 0);
        float2 scale = make_float2(0.0f, 0.0f);
        // token is the position inside this assignment tile, which is what
        // take counts; the bucket index adds the tile's start.
        if (token < take) {
            const int64_t j = bucket + a0 + token;
            const int64_t assigned = assign_rows[j];
            const int64_t row = kActivationBySlot
                                    ? assigned * used + assign_slots[j]
                                    : assigned;
            low = *reinterpret_cast<const uint2 *>(xs + row * n_in + col);
            high = *reinterpret_cast<const uint2 *>(xs + row * n_in + col + 16);
            scale = xds[row * (n_in / 32) + col / 32];
        }
        *reinterpret_cast<int4 *>(dst) =
            make_int4(static_cast<int>(low.x), static_cast<int>(low.y),
                      static_cast<int>(high.x), static_cast<int>(high.y));
        scale_b[pair * kTileJ + token] = scale;
    }
}

// moe_gemm_mmq_kernel computes one tile of output channels for every
// assignment of one expert. The block walks the expert's bucket in
// kTileJ-sized tiles; within a tile it stages one kStepK x channel weight tile
// and runs the int8 WMMA, so the decoded weight bytes are shared by all
// kTileJ assignments.
//
// kActivationBySlot selects the activation layout: false reads the normalized
// tile row of the assignment (gate/up), true reads the SwiGLU slot of the
// assignment (down). The output is always written to the slot row, which is
// what moe_reduce_add and the down projection expect.
template <uint32_t DType, int kTileJ, int kStepK, bool kActivationBySlot>
__global__ void moe_gemm_mmq_kernel(
    float * out, const int8_t * xs, const float2 * xds, const uint8_t * w,
    uint64_t row_bytes, const int32_t * counts, const int32_t * assign_rows,
    const int32_t * assign_slots, int rows, int used, int n_in, int n_out) {
    using T = Tile<kTileJ>;
    using S = Shared<kTileJ, kStepK>;
    constexpr int kPairsPerStep = kStepK / 32;
    // One WMMA slice is 16 K values. Q6_K and IQ2_S carry one weight scale per
    // slice, Q4_K, Q5_K, IQ2_XXS, and IQ3_S one per 32 values, and only Q4_K and
    // Q5_K add a per-row minimum term.
    constexpr bool kScalePer16 = DType == YUUKA_DTYPE_Q6_K || iq_scale_per_16<DType>();
    constexpr int kSlicesPerScale = kScalePer16 ? 1 : 2;
    constexpr bool kHasMinimum = DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K;

    extern __shared__ __align__(16) unsigned char smem[];
    int8_t * tile_a = reinterpret_cast<int8_t *>(smem);
    int8_t * tile_b = tile_a + S::kABytes;
    float * scale_a = reinterpret_cast<float *>(smem + S::kAScaleOffset);
    float2 * scale_b = reinterpret_cast<float2 *>(smem + S::kBScaleOffset);

    const int expert = blockIdx.x;
    const int count = counts[expert];
    if (count <= 0) {
        return;
    }
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t n0 = static_cast<int64_t>(blockIdx.y) * T::kTileI;
    const uint8_t * weights =
        w + static_cast<int64_t>(expert) * n_out * row_bytes;
    const int64_t bucket = static_cast<int64_t>(expert) * rows * used;

    const int row_base = (warp % T::kRowGroups) * kFragRows;
    const int col_base = (warp / T::kRowGroups) * T::kTokensPerWarp;
    const int a_base = (warp % T::kRowGroups) * kPairsPerStep * kPairBytes
                       + lane * kLaneBytes;
    int b_base[T::kTokenTiles];
    #pragma unroll
    for (int t = 0; t < T::kTokenTiles; ++t) {
        const int group = col_base / kFragRows + t;
        b_base[t] = group * kPairsPerStep * kPairBytes + lane * kLaneBytes;
    }

    float sum[T::kTokenTiles][8];
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> c[T::kTokenTiles];

    for (int a0 = 0; a0 < count; a0 += kTileJ) {
        const int take = count - a0 < kTileJ ? count - a0 : kTileJ;
        #pragma unroll
        for (int t = 0; t < T::kTokenTiles; ++t) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                sum[t][e] = 0.0f;
            }
        }
        for (int64_t k0 = 0; k0 < n_in; k0 += kStepK) {
            stage_weight_tile<DType, T::kTileI, kStepK>(
                tile_a, scale_a, weights, row_bytes, n0, n_out, k0);
            stage_activations_gather<kTileJ, kStepK, kActivationBySlot>(
                tile_b, scale_b, xs, xds, assign_rows, assign_slots, bucket, a0,
                take, used, n_in, k0);
            __syncthreads();
            #pragma unroll
            for (int pair = 0; pair < kPairsPerStep; ++pair) {
                const int4 av = *reinterpret_cast<const int4 *>(
                    tile_a + a_base + pair * kPairBytes);
                int4 bv[T::kTokenTiles];
                #pragma unroll
                for (int t = 0; t < T::kTokenTiles; ++t) {
                    bv[t] = *reinterpret_cast<const int4 *>(
                        tile_b + b_base[t] + pair * kPairBytes);
                }
                #pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const int slice = 2 * pair + s;
                    if (slice % kSlicesPerScale == 0) {
                        #pragma unroll
                        for (int t = 0; t < T::kTokenTiles; ++t) {
                            rocwmma::fill_fragment(c[t], 0);
                        }
                    }
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t,
                                      rocwmma::row_major> a;
                    set_fragment(a, s == 0 ? av.x : av.z, s == 0 ? av.y : av.w);
                    #pragma unroll
                    for (int t = 0; t < T::kTokenTiles; ++t) {
                        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t,
                                          rocwmma::col_major> b;
                        set_fragment(b, s == 0 ? bv[t].x : bv[t].z,
                                     s == 0 ? bv[t].y : bv[t].w);
                        rocwmma::mma_sync(c[t], a, b, c[t]);
                    }
                    if ((slice + 1) % kSlicesPerScale != 0) {
                        continue;
                    }
                    // The weight scales belong to this lane's eight rows, so
                    // they load once per scale block and serve every token
                    // fragment. Q4_K and Q5_K also carry a per-row minimum that
                    // the activation block sum multiplies; Q6_K and the IQ
                    // codebooks have no minimum.
                    const int block = slice / kSlicesPerScale;
                    const int b32 = kScalePer16 ? (block >> 1) : block;
                    const int slot0 = kScalePer16 ? block : 2 * block;
                    float wd[8];
                    float wm[8];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int row = row_base + 2 * e + (lane >> 4);
                        wd[e] = scale_a[slot0 * T::kTileI + row];
                        wm[e] = kHasMinimum ? scale_a[(slot0 + 1) * T::kTileI + row]
                                            : 0.0f;
                    }
                    #pragma unroll
                    for (int t = 0; t < T::kTokenTiles; ++t) {
                        const float2 dsb =
                            scale_b[b32 * kTileJ + col_base + t * kFragRows + (lane & 15)];
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            sum[t][e] = fmaf(wd[e] * dsb.x,
                                             static_cast<float>(c[t].x[e]),
                                             fmaf(wm[e], dsb.y, sum[t][e]));
                        }
                    }
                }
            }
            __syncthreads();
        }
        // Scatter the tile to the slot rows. The slot of an assignment is its
        // row in the expert output: row * used + slot.
        #pragma unroll
        for (int t = 0; t < T::kTokenTiles; ++t) {
            // token is the position inside this assignment tile; compare it
            // with take before adding the tile's start to reach the bucket.
            const int token = col_base + t * kFragRows + (lane & 15);
            if (token >= take) {
                continue;
            }
            const int64_t j = bucket + a0 + token;
            const int64_t slot = static_cast<int64_t>(assign_rows[j]) * used +
                                 assign_slots[j];
            float * dst = out + slot * n_out;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int64_t gn = n0 + row_base + 2 * e + (lane >> 4);
                if (gn < n_out) {
                    dst[gn] = sum[t][e];
                }
            }
        }
    }
}

template <uint32_t DType, int kTileJ, int kStepK, bool kActivationBySlot>
void launch_tile(float * out, const int8_t * xs, const float2 * xds,
                 const uint8_t * w, uint64_t row_bytes, const int32_t * counts,
                 const int32_t * assign_rows, const int32_t * assign_slots,
                 int rows, int experts, int used, int n_in, int n_out,
                 hipStream_t stream) {
    using S = Shared<kTileJ, kStepK>;
    static const hipError_t attribute = hipFuncSetAttribute(
        reinterpret_cast<const void *>(
            &moe_gemm_mmq_kernel<DType, kTileJ, kStepK, kActivationBySlot>),
        hipFuncAttributeMaxDynamicSharedMemorySize, S::kBytes);
    // A rejected attribute makes the launch below fail with a clear device
    // error through the caller's hipGetLastError check; nothing runs silently.
    (void)attribute;
    constexpr int kTileI = Tile<kTileJ>::kTileI;
    const dim3 grid(static_cast<unsigned>(experts),
                    static_cast<unsigned>((n_out + kTileI - 1) / kTileI));
    moe_gemm_mmq_kernel<DType, kTileJ, kStepK, kActivationBySlot>
        <<<grid, kThreads, S::kBytes, stream>>>(
            out, xs, xds, w, row_bytes, counts, assign_rows, assign_slots, rows,
            used, n_in, n_out);
}

// assignment_tile_step selects the K step that keeps two blocks resident per
// CU for the tile width. The wide assignment tile stages twice the activation
// bytes, so it halves the K step, which is the same 30 KiB budget the dense
// GEMM measured.
template <int kTileJ>
constexpr int assignment_tile_step() {
    return kTileJ <= 64 ? 128 : 64;
}

constexpr bool supported_tile(int tile_j) {
    return tile_j == 16 || tile_j == 32 || tile_j == 64 || tile_j == 128;
}

template <bool kActivationBySlot>
int launch_moe_mmq(float * out, const int8_t * xs, const float2 * xds,
                   const uint8_t * w, uint64_t row_bytes, const int32_t * counts,
                   const int32_t * assign_rows, const int32_t * assign_slots,
                   int rows, int experts, int used, int n_in, int n_out,
                   uint32_t dtype, int tile_j, hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || n_in <= 0 || n_out <= 0 ||
        !supported_tile(tile_j)) {
        return YUUKA_ERR_INVALID_ARGUMENT;
    }
    #define YUUKA_MOE_MMQ_CASE(DT, J)                                                 \
        if (dtype == (DT) && tile_j == (J)) {                                        \
            launch_tile<DT, J, assignment_tile_step<J>(), kActivationBySlot>(         \
                out, xs, xds, w, row_bytes, counts, assign_rows, assign_slots, rows,  \
                experts, used, n_in, n_out, stream);                                 \
            return YUUKA_OK;                                                         \
        }
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q4_K, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q4_K, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q4_K, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q4_K, 128)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q5_K, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q5_K, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q5_K, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q5_K, 128)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q6_K, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q6_K, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q6_K, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_Q6_K, 128)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_XXS, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_XXS, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_XXS, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_XXS, 128)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_S, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_S, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_S, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ2_S, 128)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ3_S, 16)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ3_S, 32)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ3_S, 64)
    YUUKA_MOE_MMQ_CASE(YUUKA_DTYPE_IQ3_S, 128)
    #undef YUUKA_MOE_MMQ_CASE
    // Unreached for a validated plan: the C ABI rejects an unsupported dtype
    // or tile width before the op is submitted.
    return YUUKA_ERR_UNSUPPORTED;
}

} // namespace

namespace yuuka {

int launch_moe_gate_up_mmq(const int8_t * norm_qs, const float2 * norm_ds,
                          const void * gate_exps, const void * up_exps,
                          const int32_t * counts, const int32_t * assign_rows,
                          const int32_t * assign_slots, float * gate_out,
                          float * up_out, int rows, int experts, int used,
                          int hidden, int ffn, uint64_t hidden_row_bytes,
                          uint32_t dtype, int tile_j, hipStream_t stream) {
    int status = launch_moe_mmq<false>(
        gate_out, norm_qs, norm_ds, static_cast<const uint8_t *>(gate_exps),
        hidden_row_bytes, counts, assign_rows, assign_slots, rows, experts, used,
        hidden, ffn, dtype, tile_j, stream);
    if (status != YUUKA_OK) {
        return status;
    }
    return launch_moe_mmq<false>(
        up_out, norm_qs, norm_ds, static_cast<const uint8_t *>(up_exps),
        hidden_row_bytes, counts, assign_rows, assign_slots, rows, experts, used,
        hidden, ffn, dtype, tile_j, stream);
}

int launch_moe_down_mmq(const int8_t * act_qs, const float2 * act_ds,
                       const void * down_exps, const int32_t * counts,
                       const int32_t * assign_rows, const int32_t * assign_slots,
                       float * out, int rows, int experts, int used, int hidden,
                       int ffn, uint64_t ffn_row_bytes, uint32_t dtype,
                       int tile_j, hipStream_t stream) {
    return launch_moe_mmq<true>(
        out, act_qs, act_ds, static_cast<const uint8_t *>(down_exps),
        ffn_row_bytes, counts, assign_rows, assign_slots, rows, experts, used,
        ffn, hidden, dtype, tile_j, stream);
}

} // namespace yuuka
