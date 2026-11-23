#include "../common.cuh"
#include "../kv.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

__global__ void attention_f32_kernel(float * out, const float * q, const float * k_cache,
                                     const float * v_cache, float * scores,
                                     int kv_heads, int heads, int head_dim, int kv_dim,
                                     int position, int capacity, float scale) {
    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float q_shared[kThreads];
    __shared__ float scratch[kThreads];
    const int tid = threadIdx.x;
    const bool active = tid < head_dim;
    q_shared[tid] = active ? q[head * 2 * head_dim + tid] : 0.0f;
    const int kvh = head / (heads / kv_heads);
    float * head_scores = scores + head * capacity;
    const int length = position + 1;
    __syncthreads();
    for (int p = 0; p < length; ++p) {
        const float dot = active ? q_shared[tid] * k_cache[p * kv_dim + kvh * head_dim + tid] : 0.0f;
        const float sum = block_reduce_sum<kThreads>(dot, scratch);
        if (tid == 0) {
            head_scores[p] = sum * scale;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float max = head_scores[0];
        for (int p = 1; p < length; ++p) {
            max = fmaxf(max, head_scores[p]);
        }
        float total = 0.0f;
        for (int p = 0; p < length; ++p) {
            const float e = expf(head_scores[p] - max);
            head_scores[p] = e;
            total += e;
        }
        const float inv = 1.0f / total;
        for (int p = 0; p < length; ++p) {
            head_scores[p] *= inv;
        }
    }
    __syncthreads();
    if (!active) {
        return;
    }
    float value = 0.0f;
    for (int p = 0; p < length; ++p) {
        value += head_scores[p] * v_cache[p * kv_dim + kvh * head_dim + tid];
    }
    const float gate = q[head * 2 * head_dim + head_dim + tid];
    out[head * head_dim + tid] = value / (1.0f + expf(-gate));
}


__device__ __forceinline__ int64_t paged_row(const int32_t * block_table, int block_size,
                                             int pos) {
    const int block = block_table[pos / block_size];
    return static_cast<int64_t>(block) * block_size + (pos % block_size);
}

template <bool Dense>
__global__ void attention_paged_f32_kernel(float * out, const float * q,
                                           const void * k_pool,
                                           const void * v_pool, float * scores,
                                           const int32_t * block_table, int block_size,
                                           int kv_heads, int heads, int head_dim, int kv_dim,
                                           int position, int score_stride, float scale,
                                           uint32_t dtype) {
    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float q_shared[kThreads];
    __shared__ float scratch[kThreads];
    const int tid = threadIdx.x;
    const bool active = tid < head_dim;
    q_shared[tid] = active ? q[head * (Dense ? head_dim : 2 * head_dim) + tid] : 0.0f;
    const int kvh = head / (heads / kv_heads);
    float * head_scores = scores + head * score_stride;
    const int length = position + 1;
    __syncthreads();
    for (int p = 0; p < length; ++p) {
        const int64_t row = paged_row(block_table, block_size, p);
        const float dot = active
                              ? q_shared[tid] * kv_load(k_pool, dtype, row,
                                                        kvh * head_dim + tid, kv_dim)
                              : 0.0f;
        const float sum = block_reduce_sum<kThreads>(dot, scratch);
        if (tid == 0) {
            head_scores[p] = sum * scale;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float max = head_scores[0];
        for (int p = 1; p < length; ++p) {
            max = fmaxf(max, head_scores[p]);
        }
        float total = 0.0f;
        for (int p = 0; p < length; ++p) {
            const float e = expf(head_scores[p] - max);
            head_scores[p] = e;
            total += e;
        }
        const float inv = 1.0f / total;
        for (int p = 0; p < length; ++p) {
            head_scores[p] *= inv;
        }
    }
    __syncthreads();
    if (!active) {
        return;
    }
    float value = 0.0f;
    for (int p = 0; p < length; ++p) {
        const int64_t row = paged_row(block_table, block_size, p);
        value += head_scores[p] *
                 kv_load(v_pool, dtype, row, kvh * head_dim + tid, kv_dim);
    }
    if constexpr (Dense) {
        out[head * head_dim + tid] = value;
    } else {
        const float gate = q[head * 2 * head_dim + head_dim + tid];
        out[head * head_dim + tid] = value / (1.0f + expf(-gate));
    }
}

template <bool Dense>
__global__ void attention_paged_dynamic_f32_kernel(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    float * scores, const int32_t * block_table, const int32_t * dynamic,
    int block_size, int kv_heads, int heads, int head_dim, int kv_dim,
    int score_stride, float scale, uint32_t dtype) {
    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float q_shared[kThreads];
    __shared__ float scratch[kThreads];
    const int tid = threadIdx.x;
    const bool active = tid < head_dim;
    q_shared[tid] = active ? q[head * (Dense ? head_dim : 2 * head_dim) + tid] : 0.0f;
    const int kvh = head / (heads / kv_heads);
    float * head_scores = scores + head * score_stride;
    const int length = dynamic[1] + 1;
    __syncthreads();
    for (int p = 0; p < length; ++p) {
        const int64_t row = paged_row(block_table, block_size, p);
        const float dot = active
                              ? q_shared[tid] * kv_load(k_pool, dtype, row,
                                                        kvh * head_dim + tid, kv_dim)
                              : 0.0f;
        const float sum = block_reduce_sum<kThreads>(dot, scratch);
        if (tid == 0) {
            head_scores[p] = sum * scale;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float max = head_scores[0];
        for (int p = 1; p < length; ++p) {
            max = fmaxf(max, head_scores[p]);
        }
        float total = 0.0f;
        for (int p = 0; p < length; ++p) {
            const float e = expf(head_scores[p] - max);
            head_scores[p] = e;
            total += e;
        }
        const float inv = 1.0f / total;
        for (int p = 0; p < length; ++p) {
            head_scores[p] *= inv;
        }
    }
    __syncthreads();
    if (!active) {
        return;
    }
    float value = 0.0f;
    for (int p = 0; p < length; ++p) {
        const int64_t row = paged_row(block_table, block_size, p);
        value += head_scores[p] *
                 kv_load(v_pool, dtype, row, kvh * head_dim + tid, kv_dim);
    }
    if constexpr (Dense) {
        out[head * head_dim + tid] = value;
    } else {
        const float gate = q[head * 2 * head_dim + head_dim + tid];
        out[head * head_dim + tid] = value / (1.0f + expf(-gate));
    }
}

// Flash-decoding constants. A block owns one tile of key positions; the tile
// length is 32 below kFlashWideContext and 64 above it, because more blocks win
// the short context and fewer partials win the long one. kFlashMaxTile bounds the
// shared-memory arrays, and the scratch layout the launcher and the Go planner
// share is heads * ntiles * (head_dim + 2) floats with
// ntiles = ceil(capacity / tile): the per-tile outputs first, then two floats
// (running max, running normalizer) per head and tile.
constexpr int kFlashMaxTile = 64;
constexpr int kFlashWideContext = 1024;

// flash_tile returns the tile length for a capacity. It must match
// decodeAttentionTile in engine/gpu/model.go.
__host__ __device__ __forceinline__ int flash_tile(int capacity) {
    return capacity > kFlashWideContext ? kFlashMaxTile : kFlashMaxTile / 2;
}

// load_group reads one score or weight row of a tile: kGroup consecutive floats
// in shared memory, one per query head that shares the block's KV head.
template <int kGroup>
__device__ __forceinline__ void load_group(const float * ptr, float (&out)[kGroup]) {
    if constexpr (kGroup == 1) {
        out[0] = ptr[0];
    } else if constexpr (kGroup == 2) {
        const float2 value = *reinterpret_cast<const float2 *>(ptr);
        out[0] = value.x;
        out[1] = value.y;
    } else if constexpr (kGroup == 4) {
        const float4 value = *reinterpret_cast<const float4 *>(ptr);
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = value.w;
    } else {
        const float4 lo = *reinterpret_cast<const float4 *>(ptr);
        const float4 hi = *reinterpret_cast<const float4 *>(ptr + 4);
        out[0] = lo.x;
        out[1] = lo.y;
        out[2] = lo.z;
        out[3] = lo.w;
        out[4] = hi.x;
        out[5] = hi.y;
        out[6] = hi.z;
        out[7] = hi.w;
    }
}

// load_key_span reads the head_dim/32 dims this lane owns for one key row into
// kreg. Lane l covers four consecutive dims per 128-dim block, which is 16 bytes
// of an F32 pool, 8 bytes of an F16 pool, and one 32-value Q8_0 group, so each
// dtype is read with one vector load plus one scale. A pool value is read once
// and reused by every query head of the block.
template <int kVecBlocks>
__device__ __forceinline__ void load_key_span(const void * pool, uint32_t dtype,
                                              int64_t row, int col, int head_dim,
                                              int kv_dim, int lane,
                                              float (&kreg)[4 * kVecBlocks]) {
    #pragma unroll
    for (int block = 0; block < kVecBlocks; ++block) {
        const int dim = block * 128 + lane * 4;
        if (dim >= head_dim) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                kreg[4 * block + i] = 0.0f;
            }
            continue;
        }
        if (dtype == YUUKA_DTYPE_F32) {
            const float4 value = *reinterpret_cast<const float4 *>(
                static_cast<const float *>(pool) + row * kv_dim + col + dim);
            kreg[4 * block + 0] = value.x;
            kreg[4 * block + 1] = value.y;
            kreg[4 * block + 2] = value.z;
            kreg[4 * block + 3] = value.w;
        } else if (dtype == YUUKA_DTYPE_F16) {
            // Four consecutive halves are one 8-byte load: the lane covers dims
            // 4l..4l+3, so the address is 8-byte aligned.
            const uint2 raw = *reinterpret_cast<const uint2 *>(
                static_cast<const uint8_t *>(pool) + row * kv_dim * 2 + (col + dim) * 2);
            const __half2 lo = *reinterpret_cast<const __half2 *>(&raw.x);
            const __half2 hi = *reinterpret_cast<const __half2 *>(&raw.y);
            kreg[4 * block + 0] = __low2float(lo);
            kreg[4 * block + 1] = __high2float(lo);
            kreg[4 * block + 2] = __low2float(hi);
            kreg[4 * block + 3] = __high2float(hi);
        } else {
            // One Q8_0 group is 2 scale bytes plus 32 digit bytes, and the lane
            // covers four digits of one group.
            const int64_t row_bytes = ((kv_dim + 31) / 32) * 34;
            const int column = col + dim;
            const uint8_t * group = static_cast<const uint8_t *>(pool) +
                                    row * row_bytes + (column / 32) * 34;
            const float scale = load_f16_device(group);
            const uint8_t * digits = group + 2 + (column % 32);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                kreg[4 * block + i] =
                    scale * static_cast<float>(static_cast<int8_t>(digits[i]));
            }
        }
    }
}

// load_value_pair reads two consecutive V values of one column pair. The pool is
// read once per pair, which keeps the per-instruction span at 8 bytes (F32),
// 4 bytes (F16), or one Q8_0 group scale plus two digits.
__device__ __forceinline__ void load_value_pair(const void * pool, uint32_t dtype,
                                                int64_t row, int col, int kv_dim,
                                                float (&value)[2]) {
    if (dtype == YUUKA_DTYPE_F32) {
        const float2 raw = *reinterpret_cast<const float2 *>(
            static_cast<const float *>(pool) + row * kv_dim + col);
        value[0] = raw.x;
        value[1] = raw.y;
    } else if (dtype == YUUKA_DTYPE_F16) {
        const uint32_t raw = *reinterpret_cast<const uint32_t *>(
            static_cast<const uint8_t *>(pool) + row * kv_dim * 2 + col * 2);
        const __half2 pair = *reinterpret_cast<const __half2 *>(&raw);
        value[0] = __low2float(pair);
        value[1] = __high2float(pair);
    } else {
        const int64_t row_bytes = ((kv_dim + 31) / 32) * 34;
        const uint8_t * group = static_cast<const uint8_t *>(pool) +
                                row * row_bytes + (col / 32) * 34;
        const float scale = load_f16_device(group);
        const uint8_t * digits = group + 2 + (col % 32);
        value[0] = scale * static_cast<float>(static_cast<int8_t>(digits[0]));
        value[1] = scale * static_cast<float>(static_cast<int8_t>(digits[1]));
    }
}

// attention_paged_flash_kernel is the split-key decode attention: one block owns
// one KV head and one tile of key positions, and the kGroup query heads that
// share the KV head accumulate their online-softmax state in registers. K and V
// rows are read once per block, so a KV row is read once per tile rather than
// once per query head. The partial results are merged by
// attention_flash_combine. kSubgroups defaults to one; the measured two-subgroup
// variant reduced live accumulators but doubled K/V traffic and is not shipped.
template <int kGroup, int kVecBlocks, bool kDense, int kSubgroups = 1>
__global__ void attention_paged_flash_kernel(
    const float * q, const void * k_pool, const void * v_pool, float * partials,
    const int32_t * block_table, const int32_t * dynamic, int block_size,
    int heads, int head_dim, int kv_dim, int position, int ntiles, int tile_length,
    float scale, uint32_t dtype) {
    constexpr int kHeads = kGroup / kSubgroups;
    static_assert(kGroup % kSubgroups == 0, "subgroup must divide the group");
    const int kvh = blockIdx.x / kSubgroups;
    const int sub = blockIdx.x % kSubgroups;
    const int head0 = kvh * kGroup + sub * kHeads;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int nwarps = blockDim.x / 32;
    const int length = (dynamic != nullptr ? dynamic[1] : position) + 1;
    const int start = tile * tile_length;
    // A captured graph launches its original tile extent; a tile that is dead
    // at the replayed position returns uniformly before any barrier or K/V read.
    if (start >= length) {
        return;
    }
    const int valid = min(tile_length, length - start);

    // Dynamic shared memory: two score-sized arrays, then the per-head tile max
    // and tile sum.
    extern __shared__ float smem[];
    float * score_tile = smem;
    float * weight_tile = smem + tile_length * kHeads;
    float * tile_max = smem + 2 * tile_length * kHeads;
    float * tile_sum = tile_max + kHeads;

    const int stride_q = kDense ? head_dim : 2 * head_dim;
    // The query row is per query head; every lane keeps the dims it owns for all
    // heads of the group, so the score phase needs no shared-memory query.
    float qreg[kHeads][4 * kVecBlocks];
    #pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        const float * row = q + (head0 + h) * stride_q;
        #pragma unroll
        for (int block = 0; block < kVecBlocks; ++block) {
            const int dim = block * 128 + lane * 4;
            if (dim < head_dim) {
                const float4 value = *reinterpret_cast<const float4 *>(row + dim);
                qreg[h][4 * block + 0] = value.x;
                qreg[h][4 * block + 1] = value.y;
                qreg[h][4 * block + 2] = value.z;
                qreg[h][4 * block + 3] = value.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    qreg[h][4 * block + i] = 0.0f;
                }
            }
        }
    }

    float acc[kHeads][2];
    float running_max[kHeads];
    float running_sum[kHeads];
    #pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        acc[h][0] = 0.0f;
        acc[h][1] = 0.0f;
        running_max[h] = -INFINITY;
        running_sum[h] = 0.0f;
    }

    // One tile is one online-softmax update: the score phase writes the tile,
    // one thread per head rescales the running state, one warp per head
    // exponentiates the tile, and every head dim accumulates the values. The
    // position range is block-uniform, so the barriers stay aligned for an empty
    // tile too.
    for (int p = start + warp; p < start + valid; p += nwarps) {
        float kreg[4 * kVecBlocks];
        const int64_t row = paged_row(block_table, block_size, p);
        load_key_span<kVecBlocks>(k_pool, dtype, row, kvh * head_dim, head_dim,
                                  kv_dim, lane, kreg);
        #pragma unroll
        for (int h = 0; h < kHeads; ++h) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4 * kVecBlocks; ++i) {
                sum += qreg[h][i] * kreg[i];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                score_tile[(p - start) * kHeads + h] = sum * scale;
            }
        }
    }
    __syncthreads();
    if (tid < kHeads) {
        float max = -INFINITY;
        for (int i = 0; i < valid; ++i) {
            max = fmaxf(max, score_tile[i * kHeads + tid]);
        }
        tile_max[tid] = max;
    }
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        const float previous = running_max[h];
        const float next = fmaxf(previous, tile_max[h]);
        const float rescale = isinf(previous) ? 0.0f : expf(previous - next);
        running_max[h] = next;
        running_sum[h] *= rescale;
        acc[h][0] *= rescale;
        acc[h][1] *= rescale;
    }
    // One warp per head exponentiates the tile and reduces its weight sum.
    if (warp < kHeads) {
        float total = 0.0f;
        for (int i = lane; i < valid; i += 32) {
            const float value = expf(score_tile[i * kHeads + warp] - running_max[warp]);
            weight_tile[i * kHeads + warp] = value;
            total += value;
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            tile_sum[warp] = total;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        running_sum[h] += tile_sum[h];
    }
    // Value phase: one thread per output dim pair, so a warp covers a 32-value
    // span of the V row and the block reads a whole row per position.
    const int pairs = head_dim / 2;
    if (tid < pairs) {
        const int col = kvh * head_dim + tid * 2;
        #pragma unroll 4
        for (int i = 0; i < valid; ++i) {
            const int64_t row = paged_row(block_table, block_size, start + i);
            float value[2];
            load_value_pair(v_pool, dtype, row, col, kv_dim, value);
            float weight[kHeads];
            load_group<kHeads>(weight_tile + i * kHeads, weight);
            #pragma unroll
            for (int h = 0; h < kHeads; ++h) {
                acc[h][0] += weight[h] * value[0];
                acc[h][1] += weight[h] * value[1];
            }
        }
    }
    // The next tile writes its scores only after this barrier, and it writes the
    // weight tile after the barrier that follows the max pass, so no second
    // barrier is needed around the value loop.
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        const int head = head0 + h;
        float * partial = partials + (head * ntiles + tile) * head_dim;
        if (tid < pairs) {
            partial[tid * 2 + 0] = acc[h][0];
            partial[tid * 2 + 1] = acc[h][1];
        }
        if (tid == 0) {
            float * stats = partials + heads * ntiles * head_dim;
            stats[(head * ntiles + tile) * 2 + 0] = running_max[h];
            stats[(head * ntiles + tile) * 2 + 1] = running_sum[h];
        }
    }
}

// attention_flash_combine merges the per-tile online-softmax partials with the
// log-sum-exp rescaling and applies the output gate for a gated query row.
template <bool kDense>
__global__ void attention_flash_combine_kernel(
    float * out, const float * partials, const int32_t * dynamic, const float * q,
    int heads, int head_dim, int ntiles, int position, int tile, int splits) {
    const int head = blockIdx.x / splits;
    const int split = blockIdx.x % splits;
    const int dim = split * 64 + static_cast<int>(threadIdx.x);
    const bool active = head < heads && dim < head_dim;
    // An empty tile is still launched, so the merge stops at the last live tile.
    const int length = (dynamic != nullptr ? dynamic[1] : position) + 1;
    const int live = min(ntiles, (length + tile - 1) / tile);
    const float * stats = partials + heads * ntiles * head_dim + head * ntiles * 2;
    const float * part = partials + head * ntiles * head_dim;
    float max = -INFINITY;
    for (int t = 0; t < live; ++t) {
        max = fmaxf(max, stats[2 * t]);
    }
    // One exp and one stat read per tile per 64-dimension block, staged in
    // bounded shared memory; the numerator and denominator keep tile order.
    __shared__ float weights[64];
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int base = 0; base < live; base += 64) {
        const int chunk = min(64, live - base);
        if (static_cast<int>(threadIdx.x) < chunk) {
            weights[threadIdx.x] = expf(stats[2 * (base + threadIdx.x)] - max);
        }
        __syncthreads();
        if (active) {
            #pragma unroll 4
            for (int i = 0; i < chunk; ++i) {
                const float weight = weights[i];
                numerator += weight * part[(base + i) * head_dim + dim];
                denominator += weight * stats[2 * (base + i) + 1];
            }
        }
        __syncthreads();
    }
    if (!active) {
        return;
    }
    float value = (denominator != 0.0f) ? numerator / denominator : 0.0f;
    if constexpr (!kDense) {
        const float gate = q[head * 2 * head_dim + head_dim + dim];
        value /= (1.0f + expf(-gate));
    }
    out[head * head_dim + dim] = value;
}

} // namespace

namespace yuuka {

namespace {

// flash_group returns the query-heads-per-KV-head count the tiled kernel was
// compiled for, or 0 when the shape keeps the serial kernel.
int flash_group(int heads, int kv_heads) {
    if (kv_heads <= 0 || heads % kv_heads != 0) {
        return 0;
    }
    const int group = heads / kv_heads;
    return (group == 1 || group == 2 || group == 4 || group == 8) ? group : 0;
}

int flash_vec_blocks(int head_dim) {
    if (head_dim % 4 != 0 || head_dim > 256) {
        return 0;
    }
    return head_dim > 128 ? 2 : 1;
}

// launch_flash<KGroup, kVecBlocks, kDense> launches the split-key kernel and its
// merge pass for one shape.
template <int kGroup, int kVecBlocks, bool kDense, int kSubgroups = 1>
void launch_flash(float * out, const float * q, const void * k_pool,
                  const void * v_pool, float * partials,
                  const int32_t * block_table, const int32_t * dynamic,
                  int block_size, int kv_heads, int heads, int head_dim,
                  int kv_dim, int capacity, int position, float scale,
                  uint32_t dtype, hipStream_t stream) {
    const int tile = flash_tile(capacity);
    const int ntiles = (capacity + tile - 1) / tile;
    // A plain submission only schedules tiles that hold a live position. A
    // captured graph keeps its recorded extent and the kernel returns early for
    // the dead tiles from dynamic[1].
    int dispatched = ntiles;
    if (dynamic == nullptr) {
        const int live = (position + tile) / tile;
        dispatched = live < ntiles ? live : ntiles;
    }
    constexpr int kHeads = kGroup / kSubgroups;
    static_assert(kGroup % kSubgroups == 0, "subgroup must divide the group");
    const dim3 grid(static_cast<unsigned int>(kv_heads * kSubgroups),
                    static_cast<unsigned int>(dispatched));
    const size_t shared = static_cast<size_t>(2 * tile * kHeads + 2 * kHeads) *
                          sizeof(float);
    attention_paged_flash_kernel<kGroup, kVecBlocks, kDense, kSubgroups>
        <<<grid, kThreads, shared, stream>>>(q, k_pool, v_pool, partials,
                                             block_table, dynamic, block_size, heads,
                                             head_dim, kv_dim, position, ntiles,
                                             tile, scale, dtype);
    // One 64-dim slice of one head per block: a head-dim-wide block leaves most
    // of the device idle on the partials read.
    const int splits = max(1, (head_dim + 63) / 64);
    attention_flash_combine_kernel<kDense>
        <<<heads * splits, 64, 0, stream>>>(out, partials, dynamic, q, heads,
                                            head_dim, ntiles, position, tile, splits);
}

template <int kGroup, bool kDense>
void launch_flash_vec(float * out, const float * q, const void * k_pool,
                      const void * v_pool, float * partials,
                      const int32_t * block_table, const int32_t * dynamic,
                      int block_size, int kv_heads, int heads, int head_dim,
                      int kv_dim, int capacity, int position, float scale,
                      uint32_t dtype, int vec_blocks, hipStream_t stream) {
    if (vec_blocks == 2) {
        launch_flash<kGroup, 2, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                        dynamic, block_size, kv_heads, heads, head_dim,
                                        kv_dim, capacity, position, scale, dtype, stream);
    } else {
        launch_flash<kGroup, 1, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                        dynamic, block_size, kv_heads, heads, head_dim,
                                        kv_dim, capacity, position, scale, dtype, stream);
    }
}

template <bool kDense>
void launch_flash_shape(float * out, const float * q, const void * k_pool,
                        const void * v_pool, float * partials,
                        const int32_t * block_table, const int32_t * dynamic,
                        int block_size, int kv_heads, int heads, int head_dim,
                        int kv_dim, int capacity, int position, float scale,
                        uint32_t dtype, int group, int vec_blocks,
                        hipStream_t stream) {
    switch (group) {
        case 1:
            launch_flash_vec<1, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                         dynamic, block_size, kv_heads, heads, head_dim,
                                         kv_dim, capacity, position, scale, dtype,
                                         vec_blocks, stream);
            break;
        case 2:
            launch_flash_vec<2, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                         dynamic, block_size, kv_heads, heads, head_dim,
                                         kv_dim, capacity, position, scale, dtype,
                                         vec_blocks, stream);
            break;
        case 4:
            launch_flash_vec<4, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                         dynamic, block_size, kv_heads, heads, head_dim,
                                         kv_dim, capacity, position, scale, dtype,
                                         vec_blocks, stream);
            break;
        default:
            launch_flash_vec<8, kDense>(out, q, k_pool, v_pool, partials, block_table,
                                         dynamic, block_size, kv_heads, heads, head_dim,
                                         kv_dim, capacity, position, scale, dtype,
                                         vec_blocks, stream);
            break;
    }
}

bool flash_applicable(int heads, int kv_heads, int head_dim) {
    return flash_group(heads, kv_heads) != 0 && flash_vec_blocks(head_dim) != 0;
}

} // namespace

void launch_attention_paged_f32(float * out, const float * q, const void * k_pool,
                                const void * v_pool, float * scores,
                                const int32_t * block_table, int block_size,
                                int kv_heads, int heads, int head_dim, int kv_dim,
                                int position, int score_stride, float scale,
                                uint32_t dtype, int dense, hipStream_t stream) {
    if (flash_applicable(heads, kv_heads, head_dim)) {
        if (dense) {
            launch_flash_shape<true>(out, q, k_pool, v_pool, scores, block_table,
                                     nullptr, block_size, kv_heads, heads, head_dim,
                                     kv_dim, score_stride, position, scale, dtype,
                                     flash_group(heads, kv_heads),
                                     flash_vec_blocks(head_dim), stream);
        } else {
            launch_flash_shape<false>(out, q, k_pool, v_pool, scores, block_table,
                                      nullptr, block_size, kv_heads, heads, head_dim,
                                      kv_dim, score_stride, position, scale, dtype,
                                      flash_group(heads, kv_heads),
                                      flash_vec_blocks(head_dim), stream);
        }
        return;
    }
    if (dense) {
        attention_paged_f32_kernel<true><<<heads, kThreads, 0, stream>>>(
            out, q, k_pool, v_pool, scores, block_table, block_size, kv_heads,
            heads, head_dim, kv_dim, position, score_stride, scale, dtype);
    } else {
        attention_paged_f32_kernel<false><<<heads, kThreads, 0, stream>>>(
            out, q, k_pool, v_pool, scores, block_table, block_size, kv_heads,
            heads, head_dim, kv_dim, position, score_stride, scale, dtype);
    }
}

void launch_attention_paged_dynamic_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        float * scores, const int32_t * block_table,
                                        const int32_t * dynamic, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int score_stride, float scale,
                                        uint32_t dtype, int dense, hipStream_t stream) {
    if (flash_applicable(heads, kv_heads, head_dim)) {
        if (dense) {
            launch_flash_shape<true>(out, q, k_pool, v_pool, scores, block_table,
                                     dynamic, block_size, kv_heads, heads, head_dim,
                                     kv_dim, score_stride, 0, scale, dtype,
                                     flash_group(heads, kv_heads),
                                     flash_vec_blocks(head_dim), stream);
        } else {
            launch_flash_shape<false>(out, q, k_pool, v_pool, scores, block_table,
                                      dynamic, block_size, kv_heads, heads, head_dim,
                                      kv_dim, score_stride, 0, scale, dtype,
                                      flash_group(heads, kv_heads),
                                      flash_vec_blocks(head_dim), stream);
        }
        return;
    }
    if (dense) {
        attention_paged_dynamic_f32_kernel<true><<<heads, kThreads, 0, stream>>>(
            out, q, k_pool, v_pool, scores, block_table, dynamic, block_size,
            kv_heads, heads, head_dim, kv_dim, score_stride, scale, dtype);
    } else {
        attention_paged_dynamic_f32_kernel<false><<<heads, kThreads, 0, stream>>>(
            out, q, k_pool, v_pool, scores, block_table, dynamic, block_size,
            kv_heads, heads, head_dim, kv_dim, score_stride, scale, dtype);
    }
}

void launch_attention_f32(float * out, const float * q, const float * k_cache,
                          const float * v_cache, float * scores,
                          int kv_heads, int heads, int head_dim, int kv_dim,
                          int position, int capacity, float scale, hipStream_t stream) {
    attention_f32_kernel<<<heads, kThreads, 0, stream>>>(
        out, q, k_cache, v_cache, scores, kv_heads, heads, head_dim, kv_dim,
        position, capacity, scale);
}

} // namespace yuuka
