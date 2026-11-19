#include "../common.cuh"
#include "../iq_decode.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

// moe_packed_value decodes element index of one packed expert weight row. The
// MoE kernels are the only family that decodes the IQ2/IQ3 codebooks; every
// other kernel keeps packed_value, which rejects those dtypes so a GEMV or GEMM
// on an IQ tensor still fails instead of reading it as zero.
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

// moe_route_kernel selects the top-k experts for every row, normalizes the
// selected softmax weights, and bins assignments by expert. Binning makes the
// grouped expert kernels read one contiguous assignment range per expert.
__global__ void moe_route_kernel(const float * logits, int rows, int experts,
                                 int used, int bucket_capacity, int32_t * ids,
                                 float * weights, int32_t * counts,
                                 int32_t * assign_rows, int32_t * assign_slots,
                                 float * assign_weights) {
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }
    const float * row_logits = logits + static_cast<int64_t>(row) * experts;
    int32_t * row_ids = ids + static_cast<int64_t>(row) * used;
    float * row_weights = weights + static_cast<int64_t>(row) * used;
    for (int slot = 0; slot < used; ++slot) {
        int best = -1;
        for (int expert = 0; expert < experts; ++expert) {
            bool taken = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (row_ids[prior] == expert) {
                    taken = true;
                    break;
                }
            }
            if (taken) {
                continue;
            }
            if (best < 0 || row_logits[expert] > row_logits[best]) {
                best = expert;
            }
        }
        row_ids[slot] = best < 0 ? 0 : best;
    }
    const float max_logit = row_logits[row_ids[0]];
    float total = 0.0f;
    for (int slot = 0; slot < used; ++slot) {
        const float weight = expf(row_logits[row_ids[slot]] - max_logit);
        row_weights[slot] = weight;
        total += weight;
    }
    if (total > 0.0f) {
        for (int slot = 0; slot < used; ++slot) {
            row_weights[slot] /= total;
        }
    }
    for (int slot = 0; slot < used; ++slot) {
        const int expert = row_ids[slot];
        const int index = atomicAdd(counts + expert, 1);
        if (index < bucket_capacity) {
            const int64_t base = static_cast<int64_t>(expert) * bucket_capacity + index;
            assign_rows[base] = row;
            assign_slots[base] = slot;
            assign_weights[base] = row_weights[slot];
        }
    }
}

// moe_gate_up_grouped_kernel computes one expert output row for every token
// assigned to that expert. Blocks are indexed by (expert, expert-FFN row), so
// one packed weight row serves every assigned token.
__global__ void moe_gate_up_grouped_kernel(
    const float * norm, const uint8_t * gate_exps, const uint8_t * up_exps,
    const int32_t * counts, const int32_t * assign_rows,
    const int32_t * assign_slots, float * gate_out, float * up_out, int rows,
    int experts, int used, int hidden, int ffn, uint64_t hidden_row_bytes,
    uint32_t dtype) {
    const int expert = blockIdx.x;
    const int out_row = blockIdx.y;
    if (expert >= experts || out_row >= ffn) {
        return;
    }
    const uint8_t * gate_row =
        gate_exps + static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
    const uint8_t * up_row =
        up_exps + static_cast<int64_t>(expert * ffn + out_row) * hidden_row_bytes;
    __shared__ float scratch[kThreads];
    const int count = counts[expert];
    for (int a = 0; a < count; ++a) {
        const int64_t base = static_cast<int64_t>(expert) * rows * used + a;
        const int row = assign_rows[base];
        const int slot = assign_slots[base];
        const float * x = norm + static_cast<int64_t>(row) * hidden;
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float value = x[i];
            gate_sum += value * moe_packed_value(dtype, gate_row, i);
            up_sum += value * moe_packed_value(dtype, up_row, i);
        }
        gate_sum = block_reduce_sum<kThreads>(gate_sum, scratch);
        up_sum = block_reduce_sum<kThreads>(up_sum, scratch);
        if (threadIdx.x == 0) {
            const int64_t dst =
                (static_cast<int64_t>(row) * used + slot) * ffn + out_row;
            gate_out[dst] = gate_sum;
            up_out[dst] = up_sum;
        }
        __syncthreads();
    }
}

// moe_down_grouped_kernel computes one expert down-projection row for every
// token assigned to that expert.
__global__ void moe_down_grouped_kernel(
    const float * act, const uint8_t * down_exps, const int32_t * counts,
    const int32_t * assign_rows, const int32_t * assign_slots, float * out,
    int rows, int experts, int used, int hidden, int ffn,
    uint64_t ffn_row_bytes, uint32_t dtype) {
    const int expert = blockIdx.x;
    const int out_row = blockIdx.y;
    if (expert >= experts || out_row >= hidden) {
        return;
    }
    const uint8_t * weight_row =
        down_exps + static_cast<int64_t>(expert * hidden + out_row) * ffn_row_bytes;
    __shared__ float scratch[kThreads];
    const int count = counts[expert];
    for (int a = 0; a < count; ++a) {
        const int64_t base = static_cast<int64_t>(expert) * rows * used + a;
        const int row = assign_rows[base];
        const int slot = assign_slots[base];
        const float * x =
            act + (static_cast<int64_t>(row) * used + slot) * ffn;
        float sum = 0.0f;
        for (int i = threadIdx.x; i < ffn; i += blockDim.x) {
            sum += x[i] * moe_packed_value(dtype, weight_row, i);
        }
        sum = block_reduce_sum<kThreads>(sum, scratch);
        if (threadIdx.x == 0) {
            const int64_t dst =
                (static_cast<int64_t>(row) * used + slot) * hidden + out_row;
            out[dst] = sum;
        }
        __syncthreads();
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
// expert of one token. Blocks are indexed by (expert-FFN row, token slot), so
// the decode path launches one grid over the used experts only instead of the
// grouped kernel's experts-by-FFN grid, which would schedule every unselected
// expert row.
__global__ void moe_gate_up_selected_kernel(
    const float * norm, const int32_t * ids, const uint8_t * gate_exps,
    const uint8_t * up_exps, float * act, int rows, int used, int hidden,
    int ffn, uint64_t hidden_row_bytes, uint32_t dtype) {
    const int out_row = blockIdx.x;
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
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float value = x[i];
        gate_sum += value * moe_packed_value(dtype, gate_row, i);
        up_sum += value * moe_packed_value(dtype, up_row, i);
    }
    __shared__ float scratch[kThreads];
    gate_sum = block_reduce_sum<kThreads>(gate_sum, scratch);
    up_sum = block_reduce_sum<kThreads>(up_sum, scratch);
    if (threadIdx.x == 0) {
        act[static_cast<int64_t>(slot_index) * ffn + out_row] =
            (gate_sum / (1.0f + expf(-gate_sum))) * up_sum;
    }
}

// moe_down_selected_kernel computes one down-projection row for one selected
// expert of one token.
__global__ void moe_down_selected_kernel(
    const float * act, const int32_t * ids, const uint8_t * down_exps, float * out,
    int rows, int used, int hidden, int ffn, uint64_t ffn_row_bytes,
    uint32_t dtype) {
    const int out_row = blockIdx.x;
    const int slot_index = blockIdx.y;
    if (out_row >= hidden || slot_index >= rows * used) {
        return;
    }
    const int expert = ids[slot_index];
    const uint8_t * weight_row =
        down_exps + static_cast<int64_t>(expert * hidden + out_row) * ffn_row_bytes;
    const float * x = act + static_cast<int64_t>(slot_index) * ffn;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < ffn; i += blockDim.x) {
        sum += x[i] * moe_packed_value(dtype, weight_row, i);
    }
    __shared__ float scratch[kThreads];
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
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

int launch_moe_route(const float * logits, int rows, int experts, int used,
                     int bucket_capacity, int32_t * ids, float * weights,
                     int32_t * counts, int32_t * assign_rows,
                     int32_t * assign_slots, float * assign_weights,
                     hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || used > experts ||
        bucket_capacity < rows * used) {
        return YUUKA_ERR_INVALID_ARGUMENT;
    }
    const hipError_t err =
        hipMemsetAsync(counts, 0, static_cast<size_t>(experts) * sizeof(int32_t), stream);
    if (err != hipSuccess) {
        return YUUKA_ERR_DEVICE;
    }
    moe_route_kernel<<<rows, kThreads, 0, stream>>>(
        logits, rows, experts, used, bucket_capacity, ids, weights, counts,
        assign_rows, assign_slots, assign_weights);
    return YUUKA_OK;
}

void launch_moe_gate_up_grouped(const float * norm, const void * gate_exps,
                                const void * up_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * gate_out,
                                float * up_out, int rows, int experts, int used,
                                int hidden, int ffn, uint64_t hidden_row_bytes,
                                uint32_t dtype,
                                hipStream_t stream) {
    if (rows <= 0 || experts <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(experts), static_cast<unsigned>(ffn));
    moe_gate_up_grouped_kernel<<<grid, kThreads, 0, stream>>>(
        norm, static_cast<const uint8_t *>(gate_exps),
        static_cast<const uint8_t *>(up_exps), counts, assign_rows, assign_slots,
        gate_out, up_out, rows, experts, used, hidden, ffn, hidden_row_bytes,
        dtype);
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
    const dim3 grid(static_cast<unsigned>(experts), static_cast<unsigned>(hidden));
    moe_down_grouped_kernel<<<grid, kThreads, 0, stream>>>(
        act, static_cast<const uint8_t *>(down_exps), counts, assign_rows,
        assign_slots, out, rows, experts, used, hidden, ffn, ffn_row_bytes,
        dtype);
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
    const dim3 grid(static_cast<unsigned>(ffn),
                    static_cast<unsigned>(rows * used));
    moe_gate_up_selected_kernel<<<grid, kThreads, 0, stream>>>(
        norm, ids, static_cast<const uint8_t *>(gate_exps),
        static_cast<const uint8_t *>(up_exps), act, rows, used, hidden, ffn,
        hidden_row_bytes, dtype);
}

void launch_moe_down_selected(const float * act, const int32_t * ids,
                              const void * down_exps, float * out, int rows,
                              int used, int hidden, int ffn,
                              uint64_t ffn_row_bytes, uint32_t dtype,
                              hipStream_t stream) {
    if (rows <= 0 || used <= 0 || hidden <= 0 || ffn <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(hidden),
                    static_cast<unsigned>(rows * used));
    moe_down_selected_kernel<<<grid, kThreads, 0, stream>>>(
        act, ids, static_cast<const uint8_t *>(down_exps), out, rows, used,
        hidden, ffn, ffn_row_bytes, dtype);
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
