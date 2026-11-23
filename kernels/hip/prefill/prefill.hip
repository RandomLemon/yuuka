#include "../common.cuh"
#include "../kv.cuh"
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
        case YUUKA_DTYPE_Q4_0: {
            const uint8_t * blk = row + (index / 32) * 18;
            const int64_t j = index % 32;
            const uint8_t packed = blk[2 + (j & 15)];
            const int q = (j < 16) ? (packed & 0xF) : (packed >> 4);
            return load_f16_device(blk) * static_cast<float>(q - 8);
        }
        case YUUKA_DTYPE_Q5_0: {
            const uint8_t * blk = row + (index / 32) * 22;
            const int64_t j = index % 32;
            const uint8_t packed = blk[6 + (j & 15)];
            const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                                (static_cast<uint32_t>(blk[3]) << 8) |
                                (static_cast<uint32_t>(blk[4]) << 16) |
                                (static_cast<uint32_t>(blk[5]) << 24);
            const int high = (j < 16)
                                 ? static_cast<int>(((qh >> j) << 4) & 0x10)
                                 : static_cast<int>((qh >> ((j & 15) + 12)) & 0x10);
            const int q = (j < 16 ? (packed & 0xF) : (packed >> 4)) | high;
            return load_f16_device(blk) * static_cast<float>(q - 16);
        }
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

constexpr int kQPerBlock = 16;
constexpr int kKeysPerBlock = 16;
constexpr int kMaxHeadVec = 8;
constexpr int kAttnThreads = 512;

// attention_prefill_f32_kernel implements tiled online-softmax attention over
// a prompt chunk. One wave owns one query row; the workgroup stages a block of
// K and V in shared memory, so the same V tile is reused by every query row in
// the workgroup. The causal limit is position+row, and each query head uses its
// GQA KV head.
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
    const int lane_stride = vec + 1;
    const int key_stride = 32 * lane_stride;
    extern __shared__ float kv_shared[];
    float * k_shared = kv_shared;
    float * v_shared = kv_shared + kKeysPerBlock * key_stride;
    for (int k0 = 0; k0 <= max_pos; k0 += kKeysPerBlock) {
        const int valid = min(kKeysPerBlock, max_pos + 1 - k0);
        for (int idx = tid; idx < valid * head_dim; idx += blockDim.x) {
            const int kk = idx / head_dim;
            const int d = idx % head_dim;
            const int64_t row = k0 + kk;
            const int slot = kk * key_stride + (d / vec) * lane_stride + d % vec;
            k_shared[slot] = k_cache[row * kv_dim + kvh * head_dim + d];
            v_shared[slot] = v_cache[row * kv_dim + kvh * head_dim + d];
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
                        dot += qv[i] *
                               k_shared[kk * key_stride + lane * lane_stride + i];
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
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        acc[i] = acc[i] * corr +
                                 p * v_shared[kk * key_stride + lane * lane_stride + i];
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

// conv1d_chunk_f32_kernel applies the depthwise causal convolution and its
// sigmoid gate to one prompt chunk. The carry state holds only the previous
// kernel-1 raw inputs, so output row r is a plain kernel-1 tap FIR over the raw
// chunk and every (row, channel) pair is independent: one thread per output
// element replaces a serial walk over the chunk. Summation order matches the
// state-shifting form, so the outputs are unchanged.
__global__ void conv1d_chunk_f32_kernel(const float * weights, const float * qkv,
                                        const float * state, float * out,
                                        int channels, int kernel, int rows) {
    const int64_t total = static_cast<int64_t>(rows) * channels;
    for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(idx / channels);
        const int c = static_cast<int>(idx % channels);
        const float * w = weights + c * kernel;
        float sum = 0.0f;
        for (int t = 0; t < kernel - 1; ++t) {
            const int src = r - (kernel - 1) + t;
            sum += w[t] * (src >= 0
                               ? qkv[static_cast<int64_t>(src) * channels + c]
                               : state[c * (kernel - 1) + src + kernel - 1]);
        }
        sum += w[kernel - 1] * qkv[idx];
        out[idx] = sum / (1.0f + expf(-sum));
    }
}

// conv1d_state_f32_kernel writes the trailing raw inputs the next chunk needs.
// Each element is owned by one thread, so an element that still comes from the
// previous chunk is left in place.
__global__ void conv1d_state_f32_kernel(float * state, const float * qkv,
                                        int channels, int kernel, int rows) {
    const int total = channels * (kernel - 1);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const int c = idx / (kernel - 1);
        const int src = rows - (kernel - 1) + idx % (kernel - 1);
        if (src >= 0) {
            state[idx] = qkv[static_cast<int64_t>(src) * channels + c];
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

// gdn_chunk_splitreg_f32_kernel keeps the recurrent state in registers for the
// whole chunk. One workgroup covers 32 state columns; four lanes cooperate on
// one column, each holding 32 of its 128 state rows. A layer therefore exposes
// heads * (State / 32) workgroups instead of heads workgroups, and the only
// block barriers are the two required by the q/k and decay/beta handoff per
// token row. Each column workgroup recomputes Q/K normalization.
template <int State>
__global__ void gdn_chunk_splitreg_f32_kernel(float * rec, const float * conv,
                                              const float * alpha, const float * beta,
                                              const float * dt, const float * a,
                                              float * out, int rows, int groups,
                                              int heads, int conv_dim, int inner,
                                              float eps, float q_scale) {
    constexpr int kSplit = 4;
    constexpr int kChunk = State / kSplit;
    constexpr int kCols = 32;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int head = blockIdx.x;
    const int column_in_block = tid / kSplit;
    const int part = tid % kSplit;
    const int col = blockIdx.y * kCols + column_in_block;
    const int r0 = part * kChunk;
    const int kh = head % groups;
    const int key_dim = groups * State;
    float * matrix = rec + static_cast<int64_t>(head) * State * State;
    __shared__ float qs[State];
    __shared__ float ks[State];
    __shared__ float q_part[State / 32];
    __shared__ float k_part[State / 32];
    __shared__ float decay_shared;
    __shared__ float beta_shared;
    float column[kChunk];
    #pragma unroll
    for (int j = 0; j < kChunk; ++j) {
        column[j] = matrix[(r0 + j) * State + col];
    }
    for (int r = 0; r < rows; ++r) {
        const float * row = conv + static_cast<int64_t>(r) * conv_dim;
        const float q_raw = row[kh * State + tid];
        const float k_raw = row[key_dim + kh * State + tid];
        const float q_sum = warp_reduce_sum(q_raw * q_raw);
        const float k_sum = warp_reduce_sum(k_raw * k_raw);
        if (lane == 0) {
            q_part[wave] = q_sum;
            k_part[wave] = k_sum;
        }
        if (tid == 0) {
            const float raw = alpha[static_cast<int64_t>(r) * heads + head] + dt[head];
            const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
            decay_shared = expf(softplus * a[head]);
            beta_shared = 1.0f / (1.0f + expf(-beta[static_cast<int64_t>(r) * heads + head]));
        }
        __syncthreads();
        float q_total = 0.0f;
        float k_total = 0.0f;
        #pragma unroll
        for (int w = 0; w < State / 32; ++w) {
            q_total += q_part[w];
            k_total += k_part[w];
        }
        qs[tid] = q_raw * q_scale * rsqrtf(q_total + eps);
        ks[tid] = k_raw * rsqrtf(k_total + eps);
        __syncthreads();
        const float decay = decay_shared;
        const float b = beta_shared;
        float sk = 0.0f;
        #pragma unroll
        for (int j = 0; j < kChunk; ++j) {
            column[j] *= decay;
            sk = fmaf(column[j], ks[r0 + j], sk);
        }
        sk += __shfl_xor_sync(0xffffffffffffffffull, sk, 1);
        sk += __shfl_xor_sync(0xffffffffffffffffull, sk, 2);
        const float value = row[2 * key_dim + head * State + col];
        const float delta = (value - sk) * b;
        float result = 0.0f;
        #pragma unroll
        for (int j = 0; j < kChunk; ++j) {
            column[j] = fmaf(ks[r0 + j], delta, column[j]);
            result = fmaf(column[j], qs[r0 + j], result);
        }
        result += __shfl_xor_sync(0xffffffffffffffffull, result, 1);
        result += __shfl_xor_sync(0xffffffffffffffffull, result, 2);
        if (part == 0) {
            out[static_cast<int64_t>(r) * inner + head * State + col] = result;
        }
    }
    #pragma unroll
    for (int j = 0; j < kChunk; ++j) {
        matrix[(r0 + j) * State + col] = column[j];
    }
}

__global__ void gdn_prepare_splitreg_f32_kernel(
        const float * conv, const float * alpha, const float * beta,
        const float * dt, const float * a, float * q, float * k,
        float * decay, float * beta_gate, int groups, int heads, int conv_dim,
        float eps, float q_scale) {
    constexpr int state = 128;
    const int group = blockIdx.x;
    const int64_t row = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t key_offset = (row * groups + group) * state;
    const float q_raw = conv[row * conv_dim + group * state + tid];
    const float k_raw = conv[row * conv_dim + (groups + group) * state + tid];
    const float q_sum = warp_reduce_sum(q_raw * q_raw);
    const float k_sum = warp_reduce_sum(k_raw * k_raw);
    __shared__ float q_part[4];
    __shared__ float k_part[4];
    if ((tid & 31) == 0) {
        q_part[tid / 32] = q_sum;
        k_part[tid / 32] = k_sum;
    }
    for (int64_t head = group + static_cast<int64_t>(tid) * groups;
         head < heads; head += static_cast<int64_t>(state) * groups) {
        const int64_t index = row * heads + head;
        const float raw = alpha[index] + dt[head];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        decay[index] = expf(softplus * a[head]);
        beta_gate[index] = 1.0f / (1.0f + expf(-beta[index]));
    }
    __syncthreads();
    float q_total = 0.0f;
    float k_total = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        q_total += q_part[w];
        k_total += k_part[w];
    }
    q[key_offset + tid] = q_raw * q_scale * rsqrtf(q_total + eps);
    k[key_offset + tid] = k_raw * rsqrtf(k_total + eps);
}

constexpr int kGdnPreparedLanes = 4;

__global__ void gdn_chunk_prepared_f32_kernel(
        float * rec, const float * conv, const float * q, const float * k,
        const float * decay, const float * beta, float * out, int rows,
        int groups, int heads, int conv_dim, int inner) {
    constexpr int state = 128;
    constexpr int lanes = kGdnPreparedLanes;
    constexpr int cols = 128 / lanes;
    constexpr int part_rows = state / lanes;
    const int head = blockIdx.x;
    const int col = blockIdx.y * cols + threadIdx.x / lanes;
    const int part = threadIdx.x % lanes;
    const int first = part * part_rows;
    const int key_dim = groups * state;
    const int key_offset = (head % groups) * state + first;
    float * matrix = rec + static_cast<int64_t>(head) * state * state + first * state + col;
    float column[part_rows];
    #pragma unroll
    for (int j = 0; j < part_rows; ++j) {
        column[j] = matrix[j * state];
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float * qr = q + r * key_dim + key_offset;
        const float * kr = k + r * key_dim + key_offset;
        const float d = decay[r * heads + head];
        const float b = beta[r * heads + head];
        float sk = 0.0f;
        #pragma unroll
        for (int j = 0; j < part_rows; ++j) {
            column[j] *= d;
            sk = fmaf(column[j], kr[j], sk);
        }
        #pragma unroll
        for (int distance = 1; distance < lanes; distance *= 2) {
            sk += __shfl_xor_sync(0xffffffffffffffffull, sk, distance);
        }
        const float value = conv[r * conv_dim + 2 * key_dim + head * state + col];
        const float delta = (value - sk) * b;
        float result = 0.0f;
        #pragma unroll
        for (int j = 0; j < part_rows; ++j) {
            column[j] = fmaf(kr[j], delta, column[j]);
            result = fmaf(column[j], qr[j], result);
        }
        #pragma unroll
        for (int distance = 1; distance < lanes; distance *= 2) {
            result += __shfl_xor_sync(0xffffffffffffffffull, result, distance);
        }
        if (part == 0) {
            out[r * inner + head * state + col] = result;
        }
    }
    #pragma unroll
    for (int j = 0; j < part_rows; ++j) {
        matrix[j * state] = column[j];
    }
}

// gdn_norm_gate_rows_f32_kernel applies the gated RMSNorm to the raw per-column
// dot products written by gdn_chunk_splitreg_f32_kernel. One wave owns one
// (row, head) pair, so the row reduction needs no block barrier and the column
// split does not change the normalization domain.
__global__ void gdn_norm_gate_rows_f32_kernel(float * out, const float * z,
                                              const float * norm, int heads,
                                              int state, int inner, float eps) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int row = blockIdx.y;
    if (head >= heads) {
        return;
    }
    float * h = out + static_cast<int64_t>(row) * inner + head * state;
    const float * gate = z + static_cast<int64_t>(row) * inner + head * state;
    float sum = 0.0f;
    for (int d = lane; d < state; d += 32) {
        sum += h[d] * h[d];
    }
    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(state) + eps);
    for (int d = lane; d < state; d += 32) {
        const float g = gate[d];
        h[d] *= scale * norm[d] * (g / (1.0f + expf(-g)));
    }
}

__global__ void cast_f32_to_f16_kernel(__half * dst, const float * src, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void cast_f16_to_f32_kernel(float * dst, const __half * src, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        dst[i] = __half2float(src[i]);
    }
}


__device__ __forceinline__ int64_t paged_kv_row(const int32_t * block_table,
                                                int block_size, int pos) {
    const int block = block_table[pos / block_size];
    return static_cast<int64_t>(block) * block_size + (pos % block_size);
}

__global__ void kv_write_paged_kv_kernel(void * k_pool, void * v_pool,
                                         const float * k, const float * v,
                                         int rows, int kv_dim,
                                         const int32_t * block_table,
                                         int block_size, int position,
                                         uint32_t dtype) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int64_t dst_row = paged_kv_row(block_table, block_size, position + row);
    const float * k_src = k + static_cast<int64_t>(row) * kv_dim;
    const float * v_src = v + static_cast<int64_t>(row) * kv_dim;
    if (dtype == YUUKA_DTYPE_Q8_0) {
        kv_quantize_q8_row(k_pool, dst_row, k_src, kv_dim, threadIdx.x);
        __syncthreads();
        kv_quantize_q8_row(v_pool, dst_row, v_src, kv_dim, threadIdx.x);
        return;
    }
    for (int col = threadIdx.x; col < kv_dim; col += blockDim.x) {
        kv_store(k_pool, dtype, dst_row, col, kv_dim, k_src[col]);
        kv_store(v_pool, dtype, dst_row, col, kv_dim, v_src[col]);
    }
}

__global__ void kv_write_paged_dynamic_kv_kernel(void * k_pool, void * v_pool,
                                                 const float * k, const float * v,
                                                 int rows, int kv_dim,
                                                 const int32_t * block_table,
                                                 const int32_t * dynamic,
                                                 int block_size, uint32_t dtype) {
    const int position = dynamic[1];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int64_t dst_row = paged_kv_row(block_table, block_size, position + row);
    const float * k_src = k + static_cast<int64_t>(row) * kv_dim;
    const float * v_src = v + static_cast<int64_t>(row) * kv_dim;
    if (dtype == YUUKA_DTYPE_Q8_0) {
        kv_quantize_q8_row(k_pool, dst_row, k_src, kv_dim, threadIdx.x);
        __syncthreads();
        kv_quantize_q8_row(v_pool, dst_row, v_src, kv_dim, threadIdx.x);
        return;
    }
    for (int col = threadIdx.x; col < kv_dim; col += blockDim.x) {
        kv_store(k_pool, dtype, dst_row, col, kv_dim, k_src[col]);
        kv_store(v_pool, dtype, dst_row, col, kv_dim, v_src[col]);
    }
}

// The FP16 MMA path in attention_wmma.hip replaces the earlier per-group tiled
// kernel that covered four query heads per workgroup: it serves the same group,
// runs both products on the WMMA pipe, and is measured faster at every chunk
// length and position it accepts (kernels.md section 7.5).

template <bool Dense>
__global__ void attention_prefill_paged_f32_kernel(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_heads, int heads,
    int head_dim, int kv_dim, int rows, int position, float scale,
    uint32_t dtype) {
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
    const int row_stride = heads * (Dense ? head_dim : 2 * head_dim);
    const float * q_ptr = q + static_cast<int64_t>(active ? q_row : last_row) * row_stride +
                          head * (Dense ? head_dim : 2 * head_dim);
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
    const int lane_stride = vec + 1;
    const int key_stride = 32 * lane_stride;
    extern __shared__ float kv_shared[];
    float * k_shared = kv_shared;
    float * v_shared = kv_shared + kKeysPerBlock * key_stride;
    for (int k0 = 0; k0 <= max_pos; k0 += kKeysPerBlock) {
        const int valid = min(kKeysPerBlock, max_pos + 1 - k0);
        for (int idx = tid; idx < valid * head_dim; idx += blockDim.x) {
            const int kk = idx / head_dim;
            const int d = idx % head_dim;
            const int64_t row = paged_kv_row(block_table, block_size, k0 + kk);
            const int slot = kk * key_stride + (d / vec) * lane_stride + d % vec;
            k_shared[slot] =
                kv_load(k_pool, dtype, row, kvh * head_dim + d, kv_dim);
            v_shared[slot] =
                kv_load(v_pool, dtype, row, kvh * head_dim + d, kv_dim);
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
                        dot += qv[i] *
                               k_shared[kk * key_stride + lane * lane_stride + i];
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
                #pragma unroll
                for (int i = 0; i < kMaxHeadVec; ++i) {
                    const int d = lane * vec + i;
                    if (i < vec && d < head_dim) {
                        acc[i] = acc[i] * corr +
                                 p * v_shared[kk * key_stride + lane * lane_stride + i];
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
            if constexpr (Dense) {
                out[out_row + d] = acc[i] * inv;
            } else {
                const float gate = q_ptr[head_dim + d];
                out[out_row + d] = acc[i] * inv / (1.0f + expf(-gate));
            }
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
    const int vec = (head_dim + 31) / 32;
    const size_t shared =
        size_t(2 * kKeysPerBlock) * size_t(32 * (vec + 1)) * sizeof(float);
    attention_prefill_f32_kernel<<<grid, kAttnThreads, shared, stream>>>(
        out, q, k_cache, v_cache, kv_heads, heads, head_dim, kv_dim, rows,
        position, capacity, scale);
}


void launch_kv_write_paged_f32(void * k_pool, void * v_pool, const float * k,
                               const float * v, int rows, int kv_dim,
                               const int32_t * block_table, int block_size,
                               int position, uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || kv_dim <= 0 || block_size <= 0 || position < 0) {
        return;
    }
    kv_write_paged_kv_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        k_pool, v_pool, k, v, rows, kv_dim, block_table, block_size, position, dtype);
}

void launch_kv_write_paged_dynamic_f32(void * k_pool, void * v_pool,
                                       const float * k, const float * v,
                                       int rows, int kv_dim,
                                       const int32_t * block_table,
                                       const int32_t * dynamic, int block_size,
                                       uint32_t dtype, hipStream_t stream) {
    if (rows <= 0 || kv_dim <= 0 || block_size <= 0) {
        return;
    }
    kv_write_paged_dynamic_kv_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        k_pool, v_pool, k, v, rows, kv_dim, block_table, dynamic, block_size, dtype);
}

hipError_t launch_attention_prefill_paged_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        const int32_t * block_table, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int rows, int position,
                                        float scale, uint32_t dtype, int dense,
                                        bool prefer_wmma, float * partial, int forced_splits,
                                        hipStream_t stream) {
    if (rows <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        block_size <= 0) {
        return hipSuccess;
    }
    if (partial != nullptr) {
        const bool f16 = dtype == YUUKA_DTYPE_F16;
        const bool measured_rows = rows == 512 || (f16 && (rows == 1024 || rows == 2048));
        const bool measured_tail = f16 && (rows == 64 || rows == 513) &&
            position >= 7680;
        const int min_position = dtype == YUUKA_DTYPE_Q8_0 ? 8192 : 6144;
        const int measured_splits = (measured_rows || measured_tail) &&
            position >= min_position && position <= 14336 ? 4 : 0;
        const int splits = forced_splits != 0 ? forced_splits : measured_splits;
        if (splits != 0) {
            return launch_attention_prefill_paged_split(
                out, q, k_pool, v_pool, block_table, block_size, kv_heads, heads,
                kv_dim, rows, position, scale, partial, splits, dtype, stream);
        }
    }
    if (prefer_wmma &&
        attention_wmma_supported(heads, kv_heads, head_dim, block_size, dtype)) {
        return launch_attention_prefill_paged_wmma_f32(
            out, q, k_pool, v_pool, block_table, block_size, kv_heads, heads,
            head_dim, kv_dim, rows, position, scale, dtype, dense, stream);
    }
    const int vec = (head_dim + 31) / 32;
    const size_t shared =
        size_t(2 * kKeysPerBlock) * size_t(32 * (vec + 1)) * sizeof(float);
    const int blocks = (rows + kQPerBlock - 1) / kQPerBlock;
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(blocks));
    if (dense) {
        attention_prefill_paged_f32_kernel<true><<<grid, kAttnThreads, shared, stream>>>(
            out, q, k_pool, v_pool, block_table, block_size, kv_heads, heads,
            head_dim, kv_dim, rows, position, scale, dtype);
    } else {
        attention_prefill_paged_f32_kernel<false><<<grid, kAttnThreads, shared, stream>>>(
            out, q, k_pool, v_pool, block_table, block_size, kv_heads, heads,
            head_dim, kv_dim, rows, position, scale, dtype);
    }
    return hipSuccess;
}

void launch_recurrent_chunk_f32(float * conv_state, float * rec_state,
                                const float * qkv, const float * z,
                                float * alpha, float * beta,
                                const float * conv_weights, const float * dt,
                                const float * a, const float * norm, float * conv,
                                float * out, float * scratch, float * prepared, int rows,
                                int groups, int heads, int state_size,
                                int conv_dim, int inner, int conv_kernel,
                                float eps, float q_scale, hipStream_t stream) {
    if (rows <= 0 || groups <= 0 || heads <= 0 || state_size <= 0) {
        return;
    }
    const int64_t conv_total = static_cast<int64_t>(rows) * conv_dim;
    const int conv_blocks = static_cast<int>((conv_total + kThreads - 1) / kThreads);
    conv1d_chunk_f32_kernel<<<conv_blocks, kThreads, 0, stream>>>(
        conv_weights, qkv, conv_state, conv, conv_dim, conv_kernel, rows);
    const int state_total = conv_dim * (conv_kernel - 1);
    const int state_blocks = (state_total + kThreads - 1) / kThreads;
    conv1d_state_f32_kernel<<<state_blocks, kThreads, 0, stream>>>(
        conv_state, qkv, conv_dim, conv_kernel, rows);
    if (state_size == 128) {
        if (scratch != nullptr) {
            launch_gdn_chunked_f32(rec_state, conv, alpha, beta, dt, a, out,
                                   scratch, rows, groups, heads, state_size,
                                   conv_dim, inner, eps, q_scale, stream);
        } else if (prepared != nullptr) {
            const int64_t key_count = static_cast<int64_t>(rows) * groups * 128;
            float * q = prepared;
            float * k = q + key_count;
            float * decay = k + key_count;
            float * beta_gate = decay + static_cast<int64_t>(rows) * heads;
            const dim3 prep_grid(static_cast<unsigned>(groups), static_cast<unsigned>(rows));
            gdn_prepare_splitreg_f32_kernel<<<prep_grid, 128, 0, stream>>>(
                conv, alpha, beta, dt, a, q, k, decay, beta_gate,
                groups, heads, conv_dim, eps, q_scale);
            const dim3 grid(static_cast<unsigned>(heads), kGdnPreparedLanes);
            gdn_chunk_prepared_f32_kernel<<<grid, 128, 0, stream>>>(
                rec_state, conv, q, k, decay, beta_gate, out,
                rows, groups, heads, conv_dim, inner);
        } else {
            // The split-register kernel is the reference form of the same
            // recurrence and the fallback when no chunk scratch is supplied.
            const dim3 grid(static_cast<unsigned>(heads),
                            static_cast<unsigned>(state_size / 32));
            gdn_chunk_splitreg_f32_kernel<128><<<grid, 128, 0, stream>>>(
                rec_state, conv, alpha, beta, dt, a, out, rows, groups, heads,
                conv_dim, inner, eps, q_scale);
        }
        const dim3 norm_grid(static_cast<unsigned>((heads + 3) / 4),
                             static_cast<unsigned>(rows));
        gdn_norm_gate_rows_f32_kernel<<<norm_grid, 128, 0, stream>>>(
            out, z, norm, heads, state_size, inner, eps);
        return;
    }
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

void launch_cast_f16_to_f32(void * dst, const void * src, uint64_t n,
                            hipStream_t stream) {
    if (n == 0) {
        return;
    }
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    cast_f16_to_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<float *>(dst), static_cast<const __half *>(src),
        static_cast<int64_t>(n));
}

} // namespace yuuka
