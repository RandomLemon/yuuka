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

} // namespace

namespace yuuka {

void launch_attention_paged_f32(float * out, const float * q, const void * k_pool,
                                const void * v_pool, float * scores,
                                const int32_t * block_table, int block_size,
                                int kv_heads, int heads, int head_dim, int kv_dim,
                                int position, int score_stride, float scale,
                                uint32_t dtype, int dense, hipStream_t stream) {
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
