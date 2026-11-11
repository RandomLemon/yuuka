#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

// block_sum_dynamic reduces across a power-of-two block size. The partial sums
// live in caller-provided shared memory of blockDim.x floats.
__device__ __forceinline__ float block_sum_dynamic(float value, float * scratch) {
    const int tid = threadIdx.x;
    const int n = blockDim.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void rmsnorm_batch_f32_kernel(float * dst, const float * src,
                                         const float * weight, int cols,
                                         int src_stride, int dst_stride, float eps) {
    const int row = blockIdx.x;
    __shared__ float scratch[kThreads];
    const float * s = src + static_cast<int64_t>(row) * src_stride;
    float * d = dst + static_cast<int64_t>(row) * dst_stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += s[i] * s[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        d[i] = s[i] * scale * weight[i];
    }
}

__global__ void rmsnorm_heads_batch_f32_kernel(float * dst, const float * src,
                                               const float * weight, int heads,
                                               int size, int stride, float eps) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    __shared__ float scratch[kThreads];
    const float * s = src + static_cast<int64_t>(row) * heads * stride + head * stride;
    float * d = dst + static_cast<int64_t>(row) * heads * stride + head * stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        sum += s[i] * s[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(size) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        d[i] = s[i] * scale * weight[i];
    }
}

__global__ void rope_batch_f32_kernel(float * data, int stride, int rot_dim,
                                      int pos, float base, int row_stride) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int i = threadIdx.x;
    const int half = rot_dim / 2;
    if (i >= half) {
        return;
    }
    const int position = pos + row;
    const float theta = static_cast<float>(position) * powf(base, -2.0f * i / rot_dim);
    const float c = cosf(theta);
    const float s = sinf(theta);
    float * head_ptr = data + static_cast<int64_t>(row) * row_stride + head * stride;
    const float x0 = head_ptr[i];
    const float x1 = head_ptr[i + half];
    head_ptr[i] = x0 * c - x1 * s;
    head_ptr[i + half] = x0 * s + x1 * c;
}

__global__ void kv_write_batch_f32_kernel(float * k_cache, float * v_cache,
                                          const float * k, const float * v,
                                          int rows, int kv_dim, int position) {
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / kv_dim);
        const int col = static_cast<int>(idx % kv_dim);
        const int64_t dst = static_cast<int64_t>(position + row) * kv_dim + col;
        k_cache[dst] = k[idx];
        v_cache[dst] = v[idx];
    }
}

__device__ __forceinline__ float packed_value(uint32_t dtype, const uint8_t * row,
                                              int64_t index) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return *reinterpret_cast<const float *>(row + index * 4);
        case YUUKA_DTYPE_F16:
            return load_f16_device(row + index * 2);
        case YUUKA_DTYPE_Q8_0:
            return q8_value(row + (index / 32) * 34, static_cast<int>(index % 32));
        case YUUKA_DTYPE_Q4_K:
            return q4k_value(row + (index / 256) * 144, static_cast<int>(index % 256));
        case YUUKA_DTYPE_Q5_K:
            return q5k_value(row + (index / 256) * 176, static_cast<int>(index % 256));
        case YUUKA_DTYPE_Q6_K:
            return q6k_value(row + (index / 256) * 210, static_cast<int>(index % 256));
        default:
            return 0.0f;
    }
}

__global__ void embedding_batch_f32_kernel(float * out, const uint8_t * weights,
                                           const int32_t * tokens, int rows,
                                           uint64_t row_bytes, int64_t cols,
                                           uint32_t dtype) {
    (void)rows;
    const int row = blockIdx.x;
    const int32_t token = tokens[row];
    const uint8_t * src = weights + static_cast<uint64_t>(token) * row_bytes;
    float * dst = out + static_cast<int64_t>(row) * cols;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        dst[i] = packed_value(dtype, src, i);
    }
}

constexpr int kQPerBlock = 8;
constexpr int kKeysPerBlock = 32;
constexpr int kMaxHeadVec = 8;

// attention_prefill_f32_kernel implements tiled online-softmax attention over
// a prompt chunk. One wave owns one query row; the workgroup stages a block of
// K in shared memory and reads V directly. The causal limit is position+row,
// and each query head uses its GQA KV head.
__global__ void attention_prefill_f32_kernel(float * out, const float * q,
                                             const float * k_cache,
                                             const float * v_cache, int kv_heads,
                                             int heads, int head_dim, int kv_dim,
                                             int rows, int position, int capacity,
                                             float scale) {
    (void)capacity;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int head = blockIdx.x;
    const int q_base = blockIdx.y * kQPerBlock;
    const int q_row = q_base + wave;
    const bool active = q_row < rows;
    const int vec = (head_dim + 31) / 32;
    const int group = heads / kv_heads;
    const int kvh = head / group;
    const int last_row = min(q_base + kQPerBlock, rows) - 1;
    const int max_pos = position + last_row;
    const int pos = position + (active ? q_row : last_row);
    const int row_stride = heads * 2 * head_dim;
    const float * q_ptr = q + static_cast<int64_t>(active ? q_row : last_row) * row_stride +
                          head * 2 * head_dim;
    float qv[kMaxHeadVec];
    float acc[kMaxHeadVec];
    #pragma unroll
    for (int i = 0; i < kMaxHeadVec; ++i) {
        const int d = lane * vec + i;
        qv[i] = (i < vec && d < head_dim) ? q_ptr[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    extern __shared__ float k_shared[];
    for (int k0 = 0; k0 <= max_pos; k0 += kKeysPerBlock) {
        const int valid = min(kKeysPerBlock, max_pos + 1 - k0);
        for (int idx = tid; idx < valid * head_dim; idx += blockDim.x) {
            const int kk = idx / head_dim;
            const int d = idx % head_dim;
            k_shared[kk * head_dim + d] =
                k_cache[static_cast<int64_t>(k0 + kk) * kv_dim + kvh * head_dim + d];
        }
        __syncthreads();
        if (active) {
            const int stop = min(valid, pos - k0 + 1);
            for (int kk = 0; kk < stop; ++kk) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        dot += qv[i] * k_shared[kk * head_dim + d];
                    }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_xor(dot, offset);
                }
                const float s = dot * scale;
                const float m_new = fmaxf(m, s);
                const float corr = expf(m - m_new);
                const float p = expf(s - m_new);
                l = l * corr + p;
                m = m_new;
                const float * v_ptr = v_cache +
                    static_cast<int64_t>(k0 + kk) * kv_dim + kvh * head_dim + lane * vec;
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        acc[i] = acc[i] * corr + p * v_ptr[i];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) {
        return;
    }
    const float inv = 1.0f / l;
    const int64_t out_row = static_cast<int64_t>(q_row) * heads * head_dim + head * head_dim;
    #pragma unroll
    for (int i = 0; i < kMaxHeadVec; ++i) {
        const int d = lane * vec + i;
        if (i < vec && d < head_dim) {
            const float gate = q_ptr[head_dim + d];
            out[out_row + d] = acc[i] * inv / (1.0f + expf(-gate));
        }
    }
}

__global__ void conv1d_chunk_f32_kernel(float * state, const float * weights,
                                        const float * qkv, float * out,
                                        int channels, int kernel, int rows) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels;
         c += gridDim.x * blockDim.x) {
        const float * w = weights + c * kernel;
        float * s = state + c * (kernel - 1);
        for (int r = 0; r < rows; ++r) {
            const float * in = qkv + static_cast<int64_t>(r) * channels + c;
            float sum = 0.0f;
            for (int t = 0; t < kernel - 1; ++t) {
                sum += w[t] * s[t];
            }
            sum += w[kernel - 1] * (*in);
            out[static_cast<int64_t>(r) * channels + c] = sum / (1.0f + expf(-sum));
            for (int t = 0; t < kernel - 2; ++t) {
                s[t] = s[t + 1];
            }
            s[kernel - 2] = *in;
        }
    }
}

// gdn_chunk_f32_kernel processes a whole token chunk for one gated delta-net
// layer. It fuses L2 normalization of q/k, the recurrent state update, and the
// gated RMSNorm that the decode path submits as separate kernels. blockDim.x
// must equal state_size and state_size must be a power of two.
__global__ void gdn_chunk_f32_kernel(float * rec, const float * conv,
                                     const float * z, const float * alpha,
                                     const float * beta, const float * dt,
                                     const float * a, const float * norm,
                                     float * out, int rows, int groups, int heads,
                                     int state, int conv_dim, int inner,
                                     float eps, float q_scale) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float * qs = smem;
    float * ks = smem + state;
    float * ds = smem + 2 * state;
    float * scratch = smem + 3 * state;
    __shared__ float decay_shared;
    __shared__ float beta_shared;
    const int kh = head % groups;
    const int key_dim = groups * state;
    float * matrix = rec + static_cast<int64_t>(head) * state * state;
    for (int r = 0; r < rows; ++r) {
        const float * row = conv + static_cast<int64_t>(r) * conv_dim;
        const float q_raw = row[kh * state + tid];
        const float k_raw = row[key_dim + kh * state + tid];
        const float value = row[2 * key_dim + head * state + tid];
        const float q_sum = block_sum_dynamic(q_raw * q_raw, scratch);
        // All threads must consume scratch[0] before the next reduction writes
        // it again.
        __syncthreads();
        const float k_sum = block_sum_dynamic(k_raw * k_raw, scratch);
        qs[tid] = q_raw * rsqrtf(q_sum + eps) * q_scale;
        ks[tid] = k_raw * rsqrtf(k_sum + eps);
        if (tid == 0) {
            const float raw = alpha[static_cast<int64_t>(r) * heads + head] + dt[head];
            const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
            decay_shared = expf(softplus * a[head]);
            beta_shared = 1.0f / (1.0f + expf(-beta[static_cast<int64_t>(r) * heads + head]));
        }
        __syncthreads();
        const float decay = decay_shared;
        const float b = beta_shared;
        // Each thread owns one matrix column. Decay, update, and read that
        // column so cross-wave matrix dependencies cannot race.
        for (int i = 0; i < state; ++i) {
            matrix[i * state + tid] *= decay;
        }
        __syncthreads();
        float sk = 0.0f;
        for (int i = 0; i < state; ++i) {
            sk += matrix[i * state + tid] * ks[i];
        }
        ds[tid] = (value - sk) * b;
        __syncthreads();
        for (int i = 0; i < state; ++i) {
            matrix[i * state + tid] += ks[i] * ds[tid];
        }
        __syncthreads();
        float result = 0.0f;
        for (int i = 0; i < state; ++i) {
            result += matrix[i * state + tid] * qs[i];
        }
        const float rsum = block_sum_dynamic(result * result, scratch);
        const float nscale = rsqrtf(rsum / static_cast<float>(state) + eps);
        const float gate = z[static_cast<int64_t>(r) * inner + head * state + tid];
        out[static_cast<int64_t>(r) * inner + head * state + tid] =
            result * nscale * norm[tid] * (gate / (1.0f + expf(-gate)));
        __syncthreads();
    }
}

__global__ void cast_f32_to_f16_kernel(__half * dst, const float * src, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}


__device__ __forceinline__ int64_t paged_kv_row(const int32_t * block_table,
                                                int block_size, int pos) {
    const int block = block_table[pos / block_size];
    return static_cast<int64_t>(block) * block_size + (pos % block_size);
}

__global__ void kv_write_paged_f32_kernel(float * k_pool, float * v_pool,
                                          const float * k, const float * v,
                                          int rows, int kv_dim,
                                          const int32_t * block_table,
                                          int block_size, int position) {
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / kv_dim);
        const int col = static_cast<int>(idx % kv_dim);
        const int64_t dst = paged_kv_row(block_table, block_size, position + row) * kv_dim + col;
        k_pool[dst] = k[idx];
        v_pool[dst] = v[idx];
    }
}

__global__ void kv_write_paged_dynamic_f32_kernel(float * k_pool, float * v_pool,
                                                  const float * k, const float * v,
                                                  int rows, int kv_dim,
                                                  const int32_t * block_table,
                                                  const int32_t * dynamic,
                                                  int block_size) {
    const int position = dynamic[1];
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / kv_dim);
        const int col = static_cast<int>(idx % kv_dim);
        const int64_t dst = paged_kv_row(block_table, block_size, position + row) * kv_dim + col;
        k_pool[dst] = k[idx];
        v_pool[dst] = v[idx];
    }
}

__global__ void attention_prefill_paged_f32_kernel(
    float * out, const float * q, const float * k_pool, const float * v_pool,
    const int32_t * block_table, int block_size, int kv_heads, int heads,
    int head_dim, int kv_dim, int rows, int position, float scale) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int head = blockIdx.x;
    const int q_base = blockIdx.y * kQPerBlock;
    const int q_row = q_base + wave;
    const bool active = q_row < rows;
    const int vec = (head_dim + 31) / 32;
    const int group = heads / kv_heads;
    const int kvh = head / group;
    const int last_row = min(q_base + kQPerBlock, rows) - 1;
    const int max_pos = position + last_row;
    const int pos = position + (active ? q_row : last_row);
    const int row_stride = heads * 2 * head_dim;
    const float * q_ptr = q + static_cast<int64_t>(active ? q_row : last_row) * row_stride +
                          head * 2 * head_dim;
    float qv[kMaxHeadVec];
    float acc[kMaxHeadVec];
    #pragma unroll
    for (int i = 0; i < kMaxHeadVec; ++i) {
        const int d = lane * vec + i;
        qv[i] = (i < vec && d < head_dim) ? q_ptr[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    extern __shared__ float k_shared[];
    for (int k0 = 0; k0 <= max_pos; k0 += kKeysPerBlock) {
        const int valid = min(kKeysPerBlock, max_pos + 1 - k0);
        for (int idx = tid; idx < valid * head_dim; idx += blockDim.x) {
            const int kk = idx / head_dim;
            const int d = idx % head_dim;
            const int64_t row = paged_kv_row(block_table, block_size, k0 + kk);
            k_shared[kk * head_dim + d] = k_pool[row * kv_dim + kvh * head_dim + d];
        }
        __syncthreads();
        if (active) {
            const int stop = min(valid, pos - k0 + 1);
            for (int kk = 0; kk < stop; ++kk) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        dot += qv[i] * k_shared[kk * head_dim + d];
                    }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_xor(dot, offset);
                }
                const float s = dot * scale;
                const float m_new = fmaxf(m, s);
                const float corr = expf(m - m_new);
                const float p = expf(s - m_new);
                l = l * corr + p;
                m = m_new;
                const int64_t vrow = paged_kv_row(block_table, block_size, k0 + kk);
                const float * v_ptr = v_pool + vrow * kv_dim + kvh * head_dim + lane * vec;
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        acc[i] = acc[i] * corr + p * v_ptr[i];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) {
        return;
    }
    const float inv = 1.0f / l;
    const int64_t out_row = static_cast<int64_t>(q_row) * heads * head_dim + head * head_dim;
    #pragma unroll
    for (int i = 0; i < kMaxHeadVec; ++i) {
        const int d = lane * vec + i;
        if (i < vec && d < head_dim) {
            const float gate = q_ptr[head_dim + d];
            out[out_row + d] = acc[i] * inv / (1.0f + expf(-gate));
        }
    }
}


} // namespace

namespace yuuka {

void launch_rmsnorm_batch_f32(float * dst, const float * src, const float * weight,
                              int rows, int cols, int src_stride, int dst_stride,
                              float eps, hipStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    rmsnorm_batch_f32_kernel<<<rows, kThreads, 0, stream>>>(
        dst, src, weight, cols, src_stride, dst_stride, eps);
}

void launch_rmsnorm_heads_batch_f32(float * dst, const float * src,
                                    const float * weight, int rows, int heads,
                                    int size, int stride, float eps,
                                    hipStream_t stream) {
    if (rows <= 0 || heads <= 0 || size <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(rows));
    rmsnorm_heads_batch_f32_kernel<<<grid, kThreads, 0, stream>>>(
        dst, src, weight, heads, size, stride, eps);
}

void launch_rope_batch_f32(float * data, int rows, int heads, int stride,
                           int head_dim, int rot_dim, int pos, float base,
                           int row_stride, hipStream_t stream) {
    (void)head_dim;
    if (rows <= 0 || heads <= 0 || rot_dim <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(rows));
    rope_batch_f32_kernel<<<grid, static_cast<unsigned>(rot_dim / 2), 0, stream>>>(
        data, stride, rot_dim, pos, base, row_stride);
}

void launch_kv_write_batch_f32(float * k_cache, float * v_cache, const float * k,
                               const float * v, int rows, int kv_dim, int position,
                               hipStream_t stream) {
    if (rows <= 0 || kv_dim <= 0) {
        return;
    }
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    kv_write_batch_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        k_cache, v_cache, k, v, rows, kv_dim, position);
}

void launch_embedding_batch_f32(float * out, const void * weights,
                                const int32_t * tokens, int rows, uint64_t row_bytes,
                                int64_t cols, uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    embedding_batch_f32_kernel<<<rows, kThreads, 0, stream>>>(
        out, static_cast<const uint8_t *>(weights), tokens, rows, row_bytes, cols, dtype);
}

void launch_attention_prefill_f32(float * out, const float * q,
                                  const float * k_cache, const float * v_cache,
                                  int kv_heads, int heads, int head_dim, int kv_dim,
                                  int rows, int position, int capacity, float scale,
                                  hipStream_t stream) {
    if (rows <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    const int blocks = (rows + kQPerBlock - 1) / kQPerBlock;
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(blocks));
    const size_t shared = size_t(kKeysPerBlock) * size_t(head_dim) * sizeof(float);
    attention_prefill_f32_kernel<<<grid, kThreads, shared, stream>>>(
        out, q, k_cache, v_cache, kv_heads, heads, head_dim, kv_dim, rows,
        position, capacity, scale);
}


void launch_kv_write_paged_f32(float * k_pool, float * v_pool, const float * k,
                               const float * v, int rows, int kv_dim,
                               const int32_t * block_table, int block_size,
                               int position, hipStream_t stream) {
    if (rows <= 0 || kv_dim <= 0 || block_size <= 0 || position < 0) {
        return;
    }
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    kv_write_paged_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        k_pool, v_pool, k, v, rows, kv_dim, block_table, block_size, position);
}

void launch_kv_write_paged_dynamic_f32(float * k_pool, float * v_pool,
                                       const float * k, const float * v,
                                       int rows, int kv_dim,
                                       const int32_t * block_table,
                                       const int32_t * dynamic, int block_size,
                                       hipStream_t stream) {
    if (rows <= 0 || kv_dim <= 0 || block_size <= 0) {
        return;
    }
    const int64_t total = static_cast<int64_t>(rows) * kv_dim;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    kv_write_paged_dynamic_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        k_pool, v_pool, k, v, rows, kv_dim, block_table, dynamic, block_size);
}

void launch_attention_prefill_paged_f32(float * out, const float * q,
                                        const float * k_pool, const float * v_pool,
                                        const int32_t * block_table, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int rows, int position,
                                        float scale, hipStream_t stream) {
    if (rows <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        block_size <= 0) {
        return;
    }
    const int blocks = (rows + kQPerBlock - 1) / kQPerBlock;
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(blocks));
    const size_t shared = size_t(kKeysPerBlock) * size_t(head_dim) * sizeof(float);
    attention_prefill_paged_f32_kernel<<<grid, kThreads, shared, stream>>>(
        out, q, k_pool, v_pool, block_table, block_size, kv_heads, heads,
        head_dim, kv_dim, rows, position, scale);
}

void launch_recurrent_chunk_f32(float * conv_state, float * rec_state,
                                const float * qkv, const float * z,
                                const float * alpha, const float * beta,
                                const float * conv_weights, const float * dt,
                                const float * a, const float * norm, float * conv,
                                float * out, int rows, int groups, int heads,
                                int state_size, int conv_dim, int inner,
                                int conv_kernel, float eps, float q_scale,
                                hipStream_t stream) {
    if (rows <= 0 || groups <= 0 || heads <= 0 || state_size <= 0) {
        return;
    }
    const int conv_blocks = (conv_dim + kThreads - 1) / kThreads;
    conv1d_chunk_f32_kernel<<<conv_blocks, kThreads, 0, stream>>>(
        conv_state, conv_weights, qkv, conv, conv_dim, conv_kernel, rows);
    const size_t shared = size_t(4 * state_size) * sizeof(float);
    gdn_chunk_f32_kernel<<<heads, static_cast<unsigned>(state_size), shared, stream>>>(
        rec_state, conv, z, alpha, beta, dt, a, norm, out, rows, groups, heads,
        state_size, conv_dim, inner, eps, q_scale);
}

void launch_cast_f32_to_f16(void * dst, const float * src, uint64_t n,
                            hipStream_t stream) {
    if (n == 0) {
        return;
    }
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    cast_f32_to_f16_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<__half *>(dst), src, static_cast<int64_t>(n));
}

} // namespace yuuka
