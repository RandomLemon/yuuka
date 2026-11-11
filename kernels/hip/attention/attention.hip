#include "../common.cuh"
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

} // namespace

namespace yuuka {

void launch_attention_f32(float * out, const float * q, const float * k_cache,
                          const float * v_cache, float * scores,
                          int kv_heads, int heads, int head_dim, int kv_dim,
                          int position, int capacity, float scale, hipStream_t stream) {
    attention_f32_kernel<<<heads, kThreads, 0, stream>>>(
        out, q, k_cache, v_cache, scores, kv_heads, heads, head_dim, kv_dim,
        position, capacity, scale);
}

} // namespace yuuka
