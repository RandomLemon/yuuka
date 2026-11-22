#include "../common.cuh"
#include "../iq_decode.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// One warp decodes eight weight values per lane per K step, so a step covers
// 256 values of the packed row and costs one codebook fetch per lane.
constexpr int kLaneValues = 8;
constexpr int kStepK = 32 * kLaneValues;
// Assignments that share one decoded weight span. The tile bounds the decode
// work of one expert row to count/kTile spans and keeps the accumulators of the
// tile in registers. It is the measured lever at these shapes, because the
// expert stage is limited by the per-value codebook decode, but it costs
// registers: the eight-value IQ span decoders afford a tile of eight, while the
// scalar decoder of the general dtypes is larger and measures better at four.
constexpr int kAssignTileIQ = 8;
constexpr int kAssignTileGeneric = 4;
// Output rows one warp owns. Tiling output rows trades activation traffic for
// registers and measured neutral at a tile of four on a 128-row chunk, so the
// selected form keeps one row per warp.
constexpr int kRowsPerWarp = 1;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr float kMasked = -INFINITY;

// moe_packed_value decodes element index of one packed expert weight row. It is
// the scalar path: eight general matvec dtypes plus the IQ2/IQ3 codebooks that
// only the MoE kernels and yuuka_dequant_f16 read. The span decoders below
// cover the IQ codebooks eight values at a time; the scalar form stays for the
// tails and for the dtypes without a span decoder.
__device__ __forceinline__ float moe_packed_value(uint32_t dtype,
                                                  const uint8_t * row,
                                                  int64_t index) {
    switch (dtype) {
    case YUUKA_DTYPE_IQ2_XXS:
        return iq2xxs_value(row + (index / 256) * 66, static_cast<int>(index % 256));
    case YUUKA_DTYPE_IQ2_S:
        return iq2s_value(row + (index / 256) * 82, static_cast<int>(index % 256));
    case YUUKA_DTYPE_IQ3_S:
        return iq3s_value(row + (index / 256) * 110, static_cast<int>(index % 256));
    default:
        return packed_value(dtype, row, index);
    }
}

// MoeCodebook selects the compile-time decode form of a launch. Generic keeps
// the runtime dtype, so the eight general matvec dtypes share one
// instantiation.
enum class MoeCodebook { Generic, IQ2XXS, IQ2S, IQ3S };

// moe_span8 decodes the eight values that start at base, which must be a
// multiple of eight and inside the row. The IQ forms fetch one codebook entry
// (two for IQ3_S) for the whole span instead of one per value.
template <MoeCodebook kCodebook>
__device__ __forceinline__ void moe_span8(uint32_t dtype, const uint8_t * row,
                                          int64_t base, float (&out)[kLaneValues]) {
    if constexpr (kCodebook == MoeCodebook::IQ2XXS) {
        iq2xxs_span8(row + (base / 256) * 66, static_cast<int>(base % 256), out);
    } else if constexpr (kCodebook == MoeCodebook::IQ2S) {
        iq2s_span8(row + (base / 256) * 82, static_cast<int>(base % 256), out);
    } else if constexpr (kCodebook == MoeCodebook::IQ3S) {
        iq3s_span8(row + (base / 256) * 110, static_cast<int>(base % 256), out);
    } else {
        #pragma unroll
        for (int k = 0; k < kLaneValues; ++k) {
            out[k] = packed_value(dtype, row, base + k);
        }
    }
}

// moe_dot_span forms one partial dot product of eight activation values and
// eight decoded weights. The activations are read as two aligned float4 loads,
// which is why base must be a multiple of eight.
__device__ __forceinline__ float moe_dot_span(const float * x,
                                              const float (&w)[kLaneValues]) {
    const float4 x0 = *reinterpret_cast<const float4 *>(x);
    const float4 x1 = *reinterpret_cast<const float4 *>(x + 4);
    float sum = x0.x * w[0] + x0.y * w[1] + x0.z * w[2] + x0.w * w[3];
    sum += x1.x * w[4] + x1.y * w[5] + x1.z * w[6] + x1.w * w[7];
    return sum;
}

// moe_dot_values is the same product for an activation span that is already in
// registers, which is the reuse the output-row tile depends on.
__device__ __forceinline__ float moe_dot_values(const float (&x)[kLaneValues],
                                                const float (&w)[kLaneValues]) {
    float sum = x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + x[3] * w[3];
    sum += x[4] * w[4] + x[5] * w[5] + x[6] * w[6] + x[7] * w[7];
    return sum;
}

// moe_route_kernel selects the top-k experts of every row, normalizes the
// selected softmax weights, and bins assignments by expert. One warp handles
// one row: each lane owns every 32nd expert in registers, a slot is a butterfly
// max with a lowest-index tie-break that matches the ascending scan of the
// reference selection, and the winner is masked out of the register set, so a
// slot costs one pass over the row instead of a scan per slot.
template <int kPerLane>
__global__ void moe_route_kernel(const float * logits, int rows, int experts,
                                 int used, int bucket_capacity, int32_t * ids,
                                 float * weights, int32_t * counts,
                                 int32_t * assign_rows, int32_t * assign_slots,
                                 float * assign_weights) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    const float * row_logits = logits + static_cast<int64_t>(row) * experts;
    int32_t * row_ids = ids + static_cast<int64_t>(row) * used;
    float * row_weights = weights + static_cast<int64_t>(row) * used;

    float mine[kPerLane];
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int expert = lane + 32 * i;
        mine[i] = expert < experts ? row_logits[expert] : kMasked;
    }
    #pragma unroll 1
    for (int slot = 0; slot < used; ++slot) {
        float best = kMasked;
        int best_index = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int expert = lane + 32 * i;
            if (mine[i] > best || (mine[i] == best && expert < best_index)) {
                best = mine[i];
                best_index = expert;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffffffffffull, best, offset);
            const int other_index =
                __shfl_xor_sync(0xffffffffffffffffull, best_index, offset);
            if (other > best || (other == best && other_index < best_index)) {
                best = other;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            row_ids[slot] = best_index;
        }
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_index) {
                mine[i] = kMasked;
            }
        }
    }

    // Weights: lane slot holds the exponent of its slot, the warp sums the
    // selected slots, and the lanes below used bin their assignment.
    float weight = 0.0f;
    if (lane < used) {
        weight = expf(row_logits[row_ids[lane]] - row_logits[row_ids[0]]);
    }
    const float total = warp_reduce_sum(weight);
    if (lane >= used) {
        return;
    }
    const float normalized = total > 0.0f ? weight / total : weight;
    row_weights[lane] = normalized;
    const int expert = row_ids[lane];
    const int index = atomicAdd(counts + expert, 1);
    if (index < bucket_capacity) {
        const int64_t base = static_cast<int64_t>(expert) * bucket_capacity + index;
        assign_rows[base] = row;
        assign_slots[base] = lane;
        assign_weights[base] = normalized;
    }
}

// moe_gate_up_grouped_kernel computes one expert output row for every token
// assigned to that expert. One warp owns kRowsPerWarp output rows, decodes the
// packed gate and up rows once per K step, and reuses that span across a tile of
// assignments, so a chunk costs one weight-row decode per expert row instead of
// one per assignment, and one activation read per out-row tile instead of one
// per output row.
template <MoeCodebook kCodebook, int kTile>
__global__ void moe_gate_up_grouped_kernel(
    const float * norm, const uint8_t * gate_exps, const uint8_t * up_exps,
    const int32_t * counts, const int32_t * assign_rows,
    const int32_t * assign_slots, float * gate_out, float * up_out, int rows,
    int experts, int used, int hidden, int ffn, uint64_t hidden_row_bytes,
    uint32_t dtype) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x;
    const int out_row0 = blockIdx.y * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;
    if (expert >= experts || out_row0 >= ffn) {
        return;
    }
    const int64_t bucket = static_cast<int64_t>(expert) * rows * used;
    const int count = counts[expert];
    for (int a0 = 0; a0 < count; a0 += kTile) {
        const int take = count - a0 < kTile ? count - a0 : kTile;
        int64_t tile_offset[kTile];
        float gate_acc[kRowsPerWarp][kTile];
        float up_acc[kRowsPerWarp][kTile];
        #pragma unroll
        for (int j = 0; j < kTile; ++j) {
            tile_offset[j] = j < take
                ? static_cast<int64_t>(assign_rows[bucket + a0 + j]) * hidden
                : 0;
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                gate_acc[r][j] = 0.0f;
                up_acc[r][j] = 0.0f;
            }
        }
        for (int64_t base = 0; base < hidden; base += kStepK) {
            const int64_t idx = base + kLaneValues * lane;
            const bool full = idx + kLaneValues <= hidden;
            float x[kTile][kLaneValues];
            if (full) {
                #pragma unroll
                for (int j = 0; j < kTile; ++j) {
                    const float * src = norm + tile_offset[j] + idx;
                    if (j < take) {
                        const float4 x0 = *reinterpret_cast<const float4 *>(src);
                        const float4 x1 = *reinterpret_cast<const float4 *>(src + 4);
                        x[j][0] = x0.x; x[j][1] = x0.y; x[j][2] = x0.z; x[j][3] = x0.w;
                        x[j][4] = x1.x; x[j][5] = x1.y; x[j][6] = x1.z; x[j][7] = x1.w;
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int out_row = out_row0 + r;
                if (out_row >= ffn) {
                    break;
                }
                const uint8_t * gate_row = gate_exps +
                    static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
                const uint8_t * up_row = up_exps +
                    static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
                if (full) {
                    float gate_w[kLaneValues];
                    float up_w[kLaneValues];
                    moe_span8<kCodebook>(dtype, gate_row, idx, gate_w);
                    moe_span8<kCodebook>(dtype, up_row, idx, up_w);
                    #pragma unroll
                    for (int j = 0; j < kTile; ++j) {
                        if (j >= take) {
                            continue;
                        }
                        gate_acc[r][j] += moe_dot_values(x[j], gate_w);
                        up_acc[r][j] += moe_dot_values(x[j], up_w);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < kTile; ++j) {
                        if (j >= take) {
                            continue;
                        }
                        const float * src = norm + tile_offset[j];
                        for (int64_t i = idx; i < hidden; ++i) {
                            gate_acc[r][j] += src[i] * moe_packed_value(dtype, gate_row, i);
                            up_acc[r][j] += src[i] * moe_packed_value(dtype, up_row, i);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int out_row = out_row0 + r;
            if (out_row >= ffn) {
                break;
            }
            #pragma unroll
            for (int j = 0; j < kTile; ++j) {
                if (j >= take) {
                    continue;
                }
                const float g = warp_reduce_sum(gate_acc[r][j]);
                const float u = warp_reduce_sum(up_acc[r][j]);
                if (lane == 0) {
                    const int64_t dst =
                        (static_cast<int64_t>(assign_rows[bucket + a0 + j]) * used +
                         assign_slots[bucket + a0 + j]) * ffn + out_row;
                    gate_out[dst] = g;
                    up_out[dst] = u;
                }
            }
        }
    }
}

// moe_down_grouped_kernel computes one expert down-projection row for every
// token assigned to that expert, with the same output-row x assignment tile as
// the gate/up kernel.
template <MoeCodebook kCodebook, int kTile>
__global__ void moe_down_grouped_kernel(
    const float * act, const uint8_t * down_exps, const int32_t * counts,
    const int32_t * assign_rows, const int32_t * assign_slots, float * out,
    int rows, int experts, int used, int hidden, int ffn,
    uint64_t ffn_row_bytes, uint32_t dtype) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x;
    const int out_row0 = blockIdx.y * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;
    if (expert >= experts || out_row0 >= hidden) {
        return;
    }
    const int64_t bucket = static_cast<int64_t>(expert) * rows * used;
    const int count = counts[expert];
    for (int a0 = 0; a0 < count; a0 += kTile) {
        const int take = count - a0 < kTile ? count - a0 : kTile;
        int64_t tile_offset[kTile];
        float acc[kRowsPerWarp][kTile];
        #pragma unroll
        for (int j = 0; j < kTile; ++j) {
            tile_offset[j] = j < take
                ? (static_cast<int64_t>(assign_rows[bucket + a0 + j]) * used +
                   assign_slots[bucket + a0 + j]) * ffn
                : 0;
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                acc[r][j] = 0.0f;
            }
        }
        for (int64_t base = 0; base < ffn; base += kStepK) {
            const int64_t idx = base + kLaneValues * lane;
            const bool full = idx + kLaneValues <= ffn;
            float x[kTile][kLaneValues];
            if (full) {
                #pragma unroll
                for (int j = 0; j < kTile; ++j) {
                    if (j < take) {
                        const float * src = act + tile_offset[j] + idx;
                        const float4 x0 = *reinterpret_cast<const float4 *>(src);
                        const float4 x1 = *reinterpret_cast<const float4 *>(src + 4);
                        x[j][0] = x0.x; x[j][1] = x0.y; x[j][2] = x0.z; x[j][3] = x0.w;
                        x[j][4] = x1.x; x[j][5] = x1.y; x[j][6] = x1.z; x[j][7] = x1.w;
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int out_row = out_row0 + r;
                if (out_row >= hidden) {
                    break;
                }
                const uint8_t * weight_row = down_exps +
                    static_cast<int64_t>(expert * hidden + out_row) * ffn_row_bytes;
                if (full) {
                    float w[kLaneValues];
                    moe_span8<kCodebook>(dtype, weight_row, idx, w);
                    #pragma unroll
                    for (int j = 0; j < kTile; ++j) {
                        if (j >= take) {
                            continue;
                        }
                        acc[r][j] += moe_dot_values(x[j], w);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < kTile; ++j) {
                        if (j >= take) {
                            continue;
                        }
                        const float * src = act + tile_offset[j];
                        for (int64_t i = idx; i < ffn; ++i) {
                            acc[r][j] += src[i] *
                                moe_packed_value(dtype, weight_row, i);
                        }
                    }
                }
            }
        }
        for (int j = 0; j < kTile; ++j) {
            if (j >= take) {
                continue;
            }
            const int64_t dst_base =
                (static_cast<int64_t>(assign_rows[bucket + a0 + j]) * used +
                 assign_slots[bucket + a0 + j]) * hidden;
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int out_row = out_row0 + r;
                if (out_row >= hidden) {
                    break;
                }
                const float sum = warp_reduce_sum(acc[r][j]);
                if (lane == 0) {
                    out[dst_base + out_row] = sum;
                }
            }
        }
    }
}

// Q8_1 expert route.
//
// The FP32 grouped kernels above decode eight weight values into eight FP32
// values and multiply them against an FP32 activation, once per assignment in
// the tile. This route instead quantizes the activation tile to Q8_1 and dots
// the packed int8 digits of eight weights against eight activation digits with
// two dp4a instructions, so one assignment costs two instructions per eight
// values instead of eight converts and eight FMAs, and the activation is read
// from shared memory as one eight-byte word per assignment.
//
// The tile, the K step, and the staging were measured on gfx1151 at the MoE
// prefill shapes (docs/kernels.md): sixteen assignments per tile halve the
// weight decode against the FP32 tile of eight, and a 256-value K step stages
// sixteen assignment rows of int8 activations in 4.2 KiB, which keeps the
// block resident. A row is padded by eight bytes so neighbouring rows do not
// land on the same LDS banks.
constexpr int kQ8Tile = 16;
constexpr int kQ8StepK = 256;
constexpr int kQ8XStride = kQ8StepK + 8;
constexpr int kQ8RowsPerBlock = 8;
constexpr int kQ8Block32 = kQ8StepK / 32;

static_assert(kQ8Tile * 16 == kThreads, "one thread stages one 16-byte row chunk");
static_assert(kQ8Tile * kQ8Block32 <= kThreads, "the scale tile fits one pass");
static_assert(kQ8RowsPerBlock == kWarps, "one warp owns one output row");

// dot4_i8 is the signed int8 dot product of the four digits packed in each
// operand; both digits of an IQ row and of a Q8_1 activation are signed, so both
// negation flags are set (see kernels/hip/gemv/gemv_i8.hip).
__device__ __forceinline__ int dot4_i8(int32_t a, int32_t b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, 0, false);
}

// moe_dot8 decodes the eight values that start at base of one packed IQ expert
// row into packed int8 digits and returns the scale of the span. It is
// instantiated only for the IQ codebooks, so it takes no runtime dtype.
template <MoeCodebook kCodebook>
__device__ __forceinline__ void moe_dot8(const uint8_t * row, int64_t base,
                                         int32_t (&w)[2], float & scale) {
    if constexpr (kCodebook == MoeCodebook::IQ2XXS) {
        iq2xxs_dot8(row + (base / 256) * 66, static_cast<int>(base % 256), w, scale);
    } else if constexpr (kCodebook == MoeCodebook::IQ2S) {
        iq2s_dot8(row + (base / 256) * 82, static_cast<int>(base % 256), w, scale);
    } else {
        static_assert(kCodebook == MoeCodebook::IQ3S,
                      "the Q8_1 expert route serves the IQ codebooks only");
        iq3s_dot8(row + (base / 256) * 110, static_cast<int>(base % 256), w, scale);
    }
}

// moe_gate_up_grouped_q8_kernel computes one expert gate and up output row for
// every assignment binned to that expert, from Q8_1 activations. One warp owns
// one output row, decodes its gate and up weight spans once per 256-value K
// step, and reuses them across a tile of sixteen assignments read from shared
// memory.
template <MoeCodebook kCodebook>
__global__ void moe_gate_up_grouped_q8_kernel(
    const int8_t * norm_qs, const float2 * norm_ds, const uint8_t * gate_exps,
    const uint8_t * up_exps, const int32_t * counts, const int32_t * assign_rows,
    const int32_t * assign_slots, float * gate_out, float * up_out, int rows,
    int used, int hidden, int ffn, uint64_t hidden_row_bytes) {
    __shared__ __align__(16) int8_t xs[kQ8Tile][kQ8XStride];
    __shared__ float2 xds[kQ8Tile][kQ8Block32];
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x;
    const int out_row = blockIdx.y * kQ8RowsPerBlock + (threadIdx.x >> 5);
    // A warp past the output width still stages and synchronizes, so the block
    // barriers stay uniform; only the decode, the accumulation, and the store
    // are skipped. The planned shapes make every warp active.
    const bool active = out_row < ffn;
    const int weight_row = active ? out_row : 0;
    const int64_t bucket = static_cast<int64_t>(expert) * rows * used;
    const int count = counts[expert];
    const uint8_t * gate_row =
        gate_exps + static_cast<int64_t>(expert * ffn + weight_row) * hidden_row_bytes;
    const uint8_t * up_row =
        up_exps + static_cast<int64_t>(expert * ffn + weight_row) * hidden_row_bytes;
    for (int a0 = 0; a0 < count; a0 += kQ8Tile) {
        const int take = count - a0 < kQ8Tile ? count - a0 : kQ8Tile;
        float gate_acc[kQ8Tile];
        float up_acc[kQ8Tile];
        #pragma unroll
        for (int j = 0; j < kQ8Tile; ++j) {
            gate_acc[j] = 0.0f;
            up_acc[j] = 0.0f;
        }
        for (int base = 0; base < hidden; base += kQ8StepK) {
            // Stage the assignment rows' int8 digits and block scales before
            // any warp decodes, so the K step is one LDS read per assignment.
            {
                const int j = threadIdx.x >> 4;
                const int chunk = threadIdx.x & 15;
                int4 value = make_int4(0, 0, 0, 0);
                if (j < take) {
                    const int64_t src =
                        static_cast<int64_t>(assign_rows[bucket + a0 + j]) * hidden;
                    value = *reinterpret_cast<const int4 *>(
                        norm_qs + src + base + chunk * 16);
                }
                // A row is padded by eight bytes, so the staged row base is
                // eight-byte aligned and the store is two eight-byte writes.
                int2 * dst = reinterpret_cast<int2 *>(xs[j] + chunk * 16);
                dst[0] = make_int2(value.x, value.y);
                dst[1] = make_int2(value.z, value.w);
            }
            if (threadIdx.x < kQ8Tile * kQ8Block32) {
                const int j = threadIdx.x >> 3;
                const int b = threadIdx.x & 7;
                float2 value = make_float2(0.0f, 0.0f);
                if (j < take) {
                    const int64_t src =
                        static_cast<int64_t>(assign_rows[bucket + a0 + j]) * (hidden / 32);
                    value = norm_ds[src + base / 32 + b];
                }
                xds[j][b] = value;
            }
            __syncthreads();
            if (active) {
                const int idx = base + lane * 8;
                int32_t gate_w[2];
                int32_t up_w[2];
                float gate_scale = 0.0f;
                float up_scale = 0.0f;
                moe_dot8<kCodebook>(gate_row, idx, gate_w, gate_scale);
                moe_dot8<kCodebook>(up_row, idx, up_w, up_scale);
                #pragma unroll
                for (int j = 0; j < kQ8Tile; ++j) {
                    if (j >= take) {
                        continue;
                    }
                    const int2 xw = *reinterpret_cast<const int2 *>(xs[j] + lane * 8);
                    const float x_scale = xds[j][lane / 4].x;
                    const int gate_sum =
                        dot4_i8(gate_w[0], xw.x) + dot4_i8(gate_w[1], xw.y);
                    const int up_sum =
                        dot4_i8(up_w[0], xw.x) + dot4_i8(up_w[1], xw.y);
                    gate_acc[j] += static_cast<float>(gate_sum) * (gate_scale * x_scale);
                    up_acc[j] += static_cast<float>(up_sum) * (up_scale * x_scale);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < kQ8Tile; ++j) {
            if (j >= take) {
                continue;
            }
            const float g = warp_reduce_sum(gate_acc[j]);
            const float u = warp_reduce_sum(up_acc[j]);
            if (active && lane == 0) {
                const int slot = assign_rows[bucket + a0 + j] * used +
                                 assign_slots[bucket + a0 + j];
                const int64_t dst = static_cast<int64_t>(slot) * ffn + out_row;
                gate_out[dst] = g;
                up_out[dst] = u;
            }
        }
    }
}

// moe_down_grouped_q8_kernel computes one expert down-projection row for every
// assignment binned to that expert, from the Q8_1 SwiGLU activations, with the
// same output-row and assignment tile as the gate/up form.
template <MoeCodebook kCodebook>
__global__ void moe_down_grouped_q8_kernel(
    const int8_t * act_qs, const float2 * act_ds, const uint8_t * down_exps,
    const int32_t * counts, const int32_t * assign_rows,
    const int32_t * assign_slots, float * out, int rows, int used, int hidden,
    int ffn, uint64_t ffn_row_bytes) {
    __shared__ __align__(16) int8_t xs[kQ8Tile][kQ8XStride];
    __shared__ float2 xds[kQ8Tile][kQ8Block32];
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x;
    const int out_row = blockIdx.y * kQ8RowsPerBlock + (threadIdx.x >> 5);
    const bool active = out_row < hidden;
    const int weight_row = active ? out_row : 0;
    const int64_t bucket = static_cast<int64_t>(expert) * rows * used;
    const int count = counts[expert];
    const uint8_t * weight = down_exps +
        static_cast<int64_t>(expert * hidden + weight_row) * ffn_row_bytes;
    for (int a0 = 0; a0 < count; a0 += kQ8Tile) {
        const int take = count - a0 < kQ8Tile ? count - a0 : kQ8Tile;
        float acc[kQ8Tile];
        #pragma unroll
        for (int j = 0; j < kQ8Tile; ++j) {
            acc[j] = 0.0f;
        }
        for (int base = 0; base < ffn; base += kQ8StepK) {
            {
                const int j = threadIdx.x >> 4;
                const int chunk = threadIdx.x & 15;
                int4 value = make_int4(0, 0, 0, 0);
                if (j < take) {
                    const int slot = assign_rows[bucket + a0 + j] * used +
                                     assign_slots[bucket + a0 + j];
                    value = *reinterpret_cast<const int4 *>(
                        act_qs + static_cast<int64_t>(slot) * ffn + base + chunk * 16);
                }
                // A row is padded by eight bytes, so the staged row base is
                // eight-byte aligned and the store is two eight-byte writes.
                int2 * dst = reinterpret_cast<int2 *>(xs[j] + chunk * 16);
                dst[0] = make_int2(value.x, value.y);
                dst[1] = make_int2(value.z, value.w);
            }
            if (threadIdx.x < kQ8Tile * kQ8Block32) {
                const int j = threadIdx.x >> 3;
                const int b = threadIdx.x & 7;
                float2 value = make_float2(0.0f, 0.0f);
                if (j < take) {
                    const int slot = assign_rows[bucket + a0 + j] * used +
                                     assign_slots[bucket + a0 + j];
                    value = act_ds[static_cast<int64_t>(slot) * (ffn / 32) + base / 32 + b];
                }
                xds[j][b] = value;
            }
            __syncthreads();
            if (active) {
                const int idx = base + lane * 8;
                int32_t w[2];
                float scale = 0.0f;
                moe_dot8<kCodebook>(weight, idx, w, scale);
                #pragma unroll
                for (int j = 0; j < kQ8Tile; ++j) {
                    if (j >= take) {
                        continue;
                    }
                    const int2 xw = *reinterpret_cast<const int2 *>(xs[j] + lane * 8);
                    const float x_scale = xds[j][lane / 4].x;
                    const int sum = dot4_i8(w[0], xw.x) + dot4_i8(w[1], xw.y);
                    acc[j] += static_cast<float>(sum) * (scale * x_scale);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < kQ8Tile; ++j) {
            if (j >= take) {
                continue;
            }
            const float sum = warp_reduce_sum(acc[j]);
            if (active && lane == 0) {
                const int slot = assign_rows[bucket + a0 + j] * used +
                                 assign_slots[bucket + a0 + j];
                out[static_cast<int64_t>(slot) * hidden + out_row] = sum;
            }
        }
    }
}

// moe_reduce_add_kernel combines per-slot expert outputs with their routing
// weights and accumulates into the residual stream.
__global__ void moe_reduce_add_kernel(float * x, const float * expert_out,
                                      const float * weights, int rows, int used,
                                      int hidden) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float * out_row = expert_out + static_cast<int64_t>(row) * used * hidden;
    const float * weight_row = weights + static_cast<int64_t>(row) * used;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float sum = 0.0f;
        for (int slot = 0; slot < used; ++slot) {
            sum += weight_row[slot] * out_row[slot * hidden + i];
        }
        x[static_cast<int64_t>(row) * hidden + i] += sum;
    }
}

// moe_gate_up_selected_kernel computes one expert-FFN row for one selected
// expert of one token. One warp owns one output row and decodes the packed rows
// eight values per codebook fetch, so the grid no longer needs a block-wide
// reduction per output element.
template <MoeCodebook kCodebook>
__global__ void moe_gate_up_selected_kernel(
    const float * norm, const int32_t * ids, const uint8_t * gate_exps,
    const uint8_t * up_exps, float * act, int rows, int used, int hidden,
    int ffn, uint64_t hidden_row_bytes, uint32_t dtype) {
    const int lane = threadIdx.x & 31;
    const int out_row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int slot_index = blockIdx.y;
    if (out_row >= ffn || slot_index >= rows * used) {
        return;
    }
    const int expert = ids[slot_index];
    const int row = slot_index / used;
    const uint8_t * gate_row =
        gate_exps + static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
    const uint8_t * up_row =
        up_exps + static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
    const float * x = norm + static_cast<int64_t>(row) * hidden;
    float gate = 0.0f;
    float up = 0.0f;
    for (int64_t base = 0; base < hidden; base += kStepK) {
        const int64_t idx = base + kLaneValues * lane;
        if (idx + kLaneValues <= hidden) {
            float gate_w[kLaneValues];
            float up_w[kLaneValues];
            moe_span8<kCodebook>(dtype, gate_row, idx, gate_w);
            moe_span8<kCodebook>(dtype, up_row, idx, up_w);
            gate += moe_dot_span(x + idx, gate_w);
            up += moe_dot_span(x + idx, up_w);
        } else {
            for (int64_t i = idx; i < hidden; ++i) {
                gate += x[i] * moe_packed_value(dtype, gate_row, i);
                up += x[i] * moe_packed_value(dtype, up_row, i);
            }
        }
    }
    gate = warp_reduce_sum(gate);
    up = warp_reduce_sum(up);
    if (lane == 0) {
        act[static_cast<int64_t>(slot_index) * ffn + out_row] =
            (gate / (1.0f + expf(-gate))) * up;
    }
}

// moe_down_selected_kernel computes one down-projection row for one selected
// expert of one token.
template <MoeCodebook kCodebook>
__global__ void moe_down_selected_kernel(
    const float * act, const int32_t * ids, const uint8_t * down_exps, float * out,
    int rows, int used, int hidden, int ffn, uint64_t ffn_row_bytes,
    uint32_t dtype) {
    const int lane = threadIdx.x & 31;
    const int out_row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int slot_index = blockIdx.y;
    if (out_row >= hidden || slot_index >= rows * used) {
        return;
    }
    const int expert = ids[slot_index];
    const uint8_t * weight_row =
        down_exps + static_cast<int64_t>(expert * hidden + out_row) * ffn_row_bytes;
    const float * x = act + static_cast<int64_t>(slot_index) * ffn;
    float sum = 0.0f;
    for (int64_t base = 0; base < ffn; base += kStepK) {
        const int64_t idx = base + kLaneValues * lane;
        if (idx + kLaneValues <= ffn) {
            float w[kLaneValues];
            moe_span8<kCodebook>(dtype, weight_row, idx, w);
            sum += moe_dot_span(x + idx, w);
        } else {
            for (int64_t i = idx; i < ffn; ++i) {
                sum += x[i] * moe_packed_value(dtype, weight_row, i);
            }
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        out[static_cast<int64_t>(slot_index) * hidden + out_row] = sum;
    }
}

// moe_shared_add_kernel adds the sigmoid-gated shared expert output to the
// residual stream. The gate is one dot product of the per-layer gate vector
// with the normalized row.
__global__ void moe_shared_add_kernel(float * x, const float * src,
                                      const float * gate, const float * norm,
                                      int rows, int hidden) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float * n = norm + static_cast<int64_t>(row) * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += gate[i] * n[i];
    }
    __shared__ float scratch[kThreads];
    sum = block_reduce_sum<kThreads>(sum, scratch);
    const float weight = 1.0f / (1.0f + expf(-sum));
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const int64_t index = static_cast<int64_t>(row) * hidden + i;
        x[index] += weight * src[index];
    }
}

} // namespace

namespace yuuka {

namespace {

// moe_codebook maps a dtype to the compile-time decode form of the expert
// kernels.
MoeCodebook moe_codebook(uint32_t dtype) {
    switch (dtype) {
    case YUUKA_DTYPE_IQ2_XXS:
        return MoeCodebook::IQ2XXS;
    case YUUKA_DTYPE_IQ2_S:
        return MoeCodebook::IQ2S;
    case YUUKA_DTYPE_IQ3_S:
        return MoeCodebook::IQ3S;
    default:
        return MoeCodebook::Generic;
    }
}

// moe_route_grid selects the route instantiation whose per-lane register set
// covers the expert count. The bound is 32 values per lane, so up to 1024
// experts; the launcher rejects a larger count instead of silently truncating
// the candidate set.
int moe_route_per_lane(int experts) {
    const int needed = (experts + 31) / 32;
    if (needed <= 1) {
        return 1;
    }
    if (needed <= 2) {
        return 2;
    }
    if (needed <= 4) {
        return 4;
    }
    if (needed <= 8) {
        return 8;
    }
    if (needed <= 16) {
        return 16;
    }
    if (needed <= 32) {
        return 32;
    }
    return 0;
}

} // namespace

int launch_moe_route(const float * logits, int rows, int experts, int used,
                     int bucket_capacity, int32_t * ids, float * weights,
                     int32_t * counts, int32_t * assign_rows,
                     int32_t * assign_slots, float * assign_weights,
                     hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || used > experts ||
        bucket_capacity < rows * used) {
        return YUUKA_ERR_INVALID_ARGUMENT;
    }
    const int per_lane = moe_route_per_lane(experts);
    if (per_lane == 0) {
        return YUUKA_ERR_INVALID_ARGUMENT;
    }
    const hipError_t err =
        hipMemsetAsync(counts, 0, static_cast<size_t>(experts) * sizeof(int32_t), stream);
    if (err != hipSuccess) {
        return YUUKA_ERR_DEVICE;
    }
    const dim3 grid(static_cast<unsigned>((rows + kWarps - 1) / kWarps));
    #define YUUKA_LAUNCH_MOE_ROUTE(PER_LANE)                                          \
        moe_route_kernel<PER_LANE><<<grid, kThreads, 0, stream>>>(                     \
            logits, rows, experts, used, bucket_capacity, ids, weights, counts,        \
            assign_rows, assign_slots, assign_weights)
    switch (per_lane) {
    case 1:
        YUUKA_LAUNCH_MOE_ROUTE(1);
        break;
    case 2:
        YUUKA_LAUNCH_MOE_ROUTE(2);
        break;
    case 4:
        YUUKA_LAUNCH_MOE_ROUTE(4);
        break;
    case 8:
        YUUKA_LAUNCH_MOE_ROUTE(8);
        break;
    case 16:
        YUUKA_LAUNCH_MOE_ROUTE(16);
        break;
    default:
        YUUKA_LAUNCH_MOE_ROUTE(32);
        break;
    }
    #undef YUUKA_LAUNCH_MOE_ROUTE
    return YUUKA_OK;
}

void launch_moe_gate_up_grouped(const float * norm, const void * gate_exps,
                                const void * up_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * gate_out,
                                float * up_out, int rows, int experts, int used,
                                int hidden, int ffn, uint64_t hidden_row_bytes,
                                uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(experts),
                    static_cast<unsigned>((ffn + kRowsPerBlock - 1) / kRowsPerBlock));
    const auto * gate = static_cast<const uint8_t *>(gate_exps);
    const auto * up = static_cast<const uint8_t *>(up_exps);
    #define YUUKA_LAUNCH_MOE_GATE_UP_GROUPED(CODEBOOK, TILE)                           \
        moe_gate_up_grouped_kernel<CODEBOOK, TILE><<<grid, kThreads, 0, stream>>>(            \
            norm, gate, up, counts, assign_rows, assign_slots, gate_out, up_out,       \
            rows, experts, used, hidden, ffn, hidden_row_bytes, dtype)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED(MoeCodebook::IQ2XXS, kAssignTileIQ);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED(MoeCodebook::IQ2S, kAssignTileIQ);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED(MoeCodebook::IQ3S, kAssignTileIQ);
        break;
    default:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED(MoeCodebook::Generic, kAssignTileGeneric);
        break;
    }
    #undef YUUKA_LAUNCH_MOE_GATE_UP_GROUPED
}

void launch_moe_down_grouped(const float * act, const void * down_exps,
                             const int32_t * counts, const int32_t * assign_rows,
                             const int32_t * assign_slots, float * out, int rows,
                             int experts, int used, int hidden, int ffn,
                             uint64_t ffn_row_bytes, uint32_t dtype,
                             hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(experts),
                    static_cast<unsigned>((hidden + kRowsPerBlock - 1) / kRowsPerBlock));
    const auto * down = static_cast<const uint8_t *>(down_exps);
    #define YUUKA_LAUNCH_MOE_DOWN_GROUPED(CODEBOOK, TILE)                              \
        moe_down_grouped_kernel<CODEBOOK, TILE><<<grid, kThreads, 0, stream>>>(               \
            act, down, counts, assign_rows, assign_slots, out, rows, experts, used,     \
            hidden, ffn, ffn_row_bytes, dtype)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED(MoeCodebook::IQ2XXS, kAssignTileIQ);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED(MoeCodebook::IQ2S, kAssignTileIQ);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED(MoeCodebook::IQ3S, kAssignTileIQ);
        break;
    default:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED(MoeCodebook::Generic, kAssignTileGeneric);
        break;
    }
    #undef YUUKA_LAUNCH_MOE_DOWN_GROUPED
}

void launch_moe_gate_up_grouped_q8(const int8_t * norm_qs,
                                   const float2 * norm_ds, const void * gate_exps,
                                   const void * up_exps, const int32_t * counts,
                                   const int32_t * assign_rows,
                                   const int32_t * assign_slots, float * gate_out,
                                   float * up_out, int rows, int experts, int used,
                                   int hidden, int ffn, uint64_t hidden_row_bytes,
                                   uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || hidden <= 0 || ffn <= 0 ||
        hidden % kQ8StepK != 0 || ffn % kQ8StepK != 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(experts),
                    static_cast<unsigned>((ffn + kQ8RowsPerBlock - 1) / kQ8RowsPerBlock));
    const auto * gate = static_cast<const uint8_t *>(gate_exps);
    const auto * up = static_cast<const uint8_t *>(up_exps);
    #define YUUKA_LAUNCH_MOE_GATE_UP_GROUPED_Q8(CODEBOOK)                              \
        moe_gate_up_grouped_q8_kernel<CODEBOOK><<<grid, kThreads, 0, stream>>>(        \
            norm_qs, norm_ds, gate, up, counts, assign_rows, assign_slots, gate_out,   \
            up_out, rows, used, hidden, ffn, hidden_row_bytes)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED_Q8(MoeCodebook::IQ2XXS);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED_Q8(MoeCodebook::IQ2S);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_GATE_UP_GROUPED_Q8(MoeCodebook::IQ3S);
        break;
    default:
        break;
    }
    #undef YUUKA_LAUNCH_MOE_GATE_UP_GROUPED_Q8
}

void launch_moe_down_grouped_q8(const int8_t * act_qs, const float2 * act_ds,
                                const void * down_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * out, int rows,
                                int experts, int used, int hidden, int ffn,
                                uint64_t ffn_row_bytes, uint32_t dtype,
                                hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || hidden <= 0 || ffn <= 0 ||
        hidden % kQ8StepK != 0 || ffn % kQ8StepK != 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(experts),
                    static_cast<unsigned>((hidden + kQ8RowsPerBlock - 1) / kQ8RowsPerBlock));
    const auto * down = static_cast<const uint8_t *>(down_exps);
    #define YUUKA_LAUNCH_MOE_DOWN_GROUPED_Q8(CODEBOOK)                                 \
        moe_down_grouped_q8_kernel<CODEBOOK><<<grid, kThreads, 0, stream>>>(           \
            act_qs, act_ds, down, counts, assign_rows, assign_slots, out, rows,        \
            used, hidden, ffn, ffn_row_bytes)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED_Q8(MoeCodebook::IQ2XXS);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED_Q8(MoeCodebook::IQ2S);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_DOWN_GROUPED_Q8(MoeCodebook::IQ3S);
        break;
    default:
        break;
    }
    #undef YUUKA_LAUNCH_MOE_DOWN_GROUPED_Q8
}

void launch_moe_reduce_add(float * x, const float * expert_out,
                           const float * weights, int rows, int used, int hidden,
                           hipStream_t stream) {
    if (rows <= 0 || used <= 0 || hidden <= 0) {
        return;
    }
    moe_reduce_add_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        x, expert_out, weights, rows, used, hidden);
}

void launch_moe_gate_up_selected(const float * norm, const int32_t * ids,
                                 const void * gate_exps, const void * up_exps,
                                 float * act, int rows, int used, int hidden,
                                 int ffn, uint64_t hidden_row_bytes,
                                 uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>((ffn + kWarps - 1) / kWarps),
                    static_cast<unsigned>(rows * used));
    const auto * gate = static_cast<const uint8_t *>(gate_exps);
    const auto * up = static_cast<const uint8_t *>(up_exps);
    #define YUUKA_LAUNCH_MOE_GATE_UP_SELECTED(CODEBOOK)                                \
        moe_gate_up_selected_kernel<CODEBOOK><<<grid, kThreads, 0, stream>>>(           \
            norm, ids, gate, up, act, rows, used, hidden, ffn, hidden_row_bytes, dtype)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_GATE_UP_SELECTED(MoeCodebook::IQ2XXS);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_GATE_UP_SELECTED(MoeCodebook::IQ2S);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_GATE_UP_SELECTED(MoeCodebook::IQ3S);
        break;
    default:
        YUUKA_LAUNCH_MOE_GATE_UP_SELECTED(MoeCodebook::Generic);
        break;
    }
    #undef YUUKA_LAUNCH_MOE_GATE_UP_SELECTED
}

void launch_moe_down_selected(const float * act, const int32_t * ids,
                              const void * down_exps, float * out, int rows,
                              int used, int hidden, int ffn,
                              uint64_t ffn_row_bytes, uint32_t dtype,
                              hipStream_t stream) {
    if (rows <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>((hidden + kWarps - 1) / kWarps),
                    static_cast<unsigned>(rows * used));
    const auto * down = static_cast<const uint8_t *>(down_exps);
    #define YUUKA_LAUNCH_MOE_DOWN_SELECTED(CODEBOOK)                                   \
        moe_down_selected_kernel<CODEBOOK><<<grid, kThreads, 0, stream>>>(              \
            act, ids, down, out, rows, used, hidden, ffn, ffn_row_bytes, dtype)
    switch (moe_codebook(dtype)) {
    case MoeCodebook::IQ2XXS:
        YUUKA_LAUNCH_MOE_DOWN_SELECTED(MoeCodebook::IQ2XXS);
        break;
    case MoeCodebook::IQ2S:
        YUUKA_LAUNCH_MOE_DOWN_SELECTED(MoeCodebook::IQ2S);
        break;
    case MoeCodebook::IQ3S:
        YUUKA_LAUNCH_MOE_DOWN_SELECTED(MoeCodebook::IQ3S);
        break;
    default:
        YUUKA_LAUNCH_MOE_DOWN_SELECTED(MoeCodebook::Generic);
        break;
    }
    #undef YUUKA_LAUNCH_MOE_DOWN_SELECTED
}

void launch_moe_shared_add(float * x, const float * src, const float * gate,
                           const float * norm, int rows, int hidden,
                           hipStream_t stream) {
    if (rows <= 0 || hidden <= 0) {
        return;
    }
    moe_shared_add_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        x, src, gate, norm, rows, hidden);
}

} // namespace yuuka
