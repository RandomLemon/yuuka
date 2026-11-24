// Fused per-token kernels for the gated delta-net layers of qwen35.
//
// A decode token walks each recurrent layer's 128x128 FP32 state in one pass
// instead of five: the head's state is loaded into registers, the decay, the
// two S^T products, and the rank-1 update are applied there, and the state is
// written back once. The gated RMSNorm of the same head is folded into the same
// block, because its reduction is over that head's own values.
//
// conv_l2norm fuses the short convolution with the L2 norms of the q and k
// channels it produces, so the delta-net unit of one layer is two launches
// instead of five. The convolution ring state and the recurrent state keep the
// layout the chunked prefill path consumes (docs/kernels.md section 5.6).
#include "../common.cuh"
#include "../yuuka_kernels.h"

namespace {

// The fused kernels are compiled for the state width every supported qwen35
// layer uses; other widths keep the generic operators.
constexpr int kStateSize = 128;
// Four thread groups share a column, so 512 threads hold 32 rows each and the
// whole state of a head stays in registers with room for occupancy.
constexpr int kSplits = 4;
constexpr int kThreadsPerHead = kStateSize * kSplits;
constexpr int kRowsPerThread = kStateSize / kSplits;

// Channels per conv_l2norm block. A block covers a whole number of L2 norm
// groups, so the norm of a group never crosses a block boundary.
constexpr int kConvBlock = 256;

__global__ void recurrent_token_f32_kernel(float * state, const float * q, const float * k,
                                           const float * v, const float * alpha,
                                           const float * beta_raw, const float * dt,
                                           const float * a, float * out,
                                           const float * norm_weight, const float * gate,
                                           int groups, int heads, float eps) {
    extern __shared__ float smem[];
    float * ksh = smem;
    float * qsh = smem + kStateSize;
    float * part = qsh + kStateSize;
    float * res_part = part + kThreadsPerHead;
    float * dsh = res_part + kThreadsPerHead;
    float * red = dsh + kStateSize;
    float * nrm = red + kThreadsPerHead / 32;

    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    const int tid = threadIdx.x;
    const int col = tid % kStateSize;
    const int slice = tid / kStateSize;
    const int i0 = slice * kRowsPerThread;
    const int kh = head % groups;
    const float * qh = q + kh * kStateSize;
    const float * khs = k + kh * kStateSize;
    float * m = state + size_t(head) * kStateSize * kStateSize;

    float s[kRowsPerThread];
    #pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) s[j] = m[(i0 + j) * kStateSize + col];
    if (tid < kStateSize) {
        ksh[tid] = khs[tid];
        qsh[tid] = qh[tid];
    }
    __shared__ float hdr[2];
    if (tid == 0) {
        const float raw = alpha[head] + dt[head];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        hdr[0] = expf(softplus * a[head]);
        hdr[1] = 1.0f / (1.0f + expf(-beta_raw[head]));
    }
    __syncthreads();
    const float decay = hdr[0];
    const float beta = hdr[1];

    // (S^T k) for this column, then the delta the rank-1 update uses. The
    // decay is applied to the sum instead of to each element, which is the same
    // product with one rounding: S^T k over the decayed state is decay*(S^T k)
    // over the state as loaded.
    float sk = 0.0f;
    #pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) sk += s[j] * ksh[i0 + j];
    part[tid] = sk;
    __syncthreads();
    #pragma unroll
    for (int split = 1; split < kSplits; ++split) sk += part[tid ^ (split * kStateSize)];
    sk *= decay;
    if (slice == 0) dsh[col] = (v[head * kStateSize + col] - sk) * beta;
    __syncthreads();
    const float dc = dsh[col];

    // Decay, rank-1 update, and (S^T q) in one pass over the registers.
    float res = 0.0f;
    #pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
        const float nv = s[j] * decay + ksh[i0 + j] * dc;
        s[j] = nv;
        res += nv * qsh[i0 + j];
    }
    #pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) m[(i0 + j) * kStateSize + col] = s[j];
    res_part[tid] = res;
    __syncthreads();
    #pragma unroll
    for (int split = 1; split < kSplits; ++split) res += res_part[tid ^ (split * kStateSize)];

    if (slice == 0) {
        float sq = res * res;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sq += __shfl_xor_sync(0xffffffffffffffffull, sq, offset);
        }
        if ((tid & 31) == 0) red[tid >> 5] = sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kStateSize / 32; ++w) total += red[w];
        nrm[0] = rsqrtf(total / static_cast<float>(kStateSize) + eps);
    }
    __syncthreads();
    if (slice == 0) {
        const float g = gate[head * kStateSize + col];
        out[head * kStateSize + col] =
            res * nrm[0] * norm_weight[col] * (g / (1.0f + expf(-g)));
    }
}

__global__ void conv_l2norm_f32_kernel(float * conv_state, const float * weights,
                                       const float * qkv, float * out, int channels,
                                       int kernel, int qk_channels, int q_channels,
                                       float eps, float q_scale) {
    __shared__ float red[kConvBlock / 32];
    const int base = blockIdx.x * kConvBlock;
    const int c = base + threadIdx.x;
    if (c < channels) {
        const float * w = weights + c * kernel;
        float * s = conv_state + c * (kernel - 1);
        float sum = 0.0f;
        for (int t = 0; t < kernel - 1; ++t) {
            sum += w[t] * s[t];
        }
        sum += w[kernel - 1] * qkv[c];
        out[c] = sum / (1.0f + expf(-sum));
        for (int t = 0; t < kernel - 2; ++t) {
            s[t] = s[t + 1];
        }
        s[kernel - 2] = qkv[c];
    }
    if (base >= qk_channels) {
        return;
    }
    __syncthreads();
    const float value = out[c];
    float sq = value * value;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_xor_sync(0xffffffffffffffffull, sq, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = sq;
    }
    __syncthreads();
    constexpr int kGroup = kStateSize;
    constexpr int kWavesPerGroup = kGroup / 32;
    const int group = threadIdx.x / kGroup;
    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWavesPerGroup; ++w) {
        total += red[group * kWavesPerGroup + w];
    }
    const float scale = rsqrtf(total + eps) * (c < q_channels ? q_scale : 1.0f);
    out[c] = value * scale;
}

} // namespace

namespace yuuka {

void launch_recurrent_token_f32(float * state, const float * q, const float * k,
                                const float * v, const float * alpha, const float * beta,
                                const float * dt, const float * a, float * out,
                                const float * norm_weight, const float * gate,
                                int groups, int heads, float eps, hipStream_t stream) {
    const size_t shared =
        size_t(2 * kStateSize + 2 * kThreadsPerHead + kStateSize + kThreadsPerHead / 32 + 1) *
        sizeof(float);
    recurrent_token_f32_kernel<<<heads, kThreadsPerHead, shared, stream>>>(
        state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, groups, heads, eps);
}

void launch_conv_l2norm_f32(float * conv_state, const float * weights, const float * qkv,
                            float * out, int channels, int kernel, int qk_channels,
                            int q_channels, float eps, float q_scale, hipStream_t stream) {
    const int blocks = channels / kConvBlock;
    conv_l2norm_f32_kernel<<<blocks, kConvBlock, 0, stream>>>(
        conv_state, weights, qkv, out, channels, kernel, qk_channels, q_channels, eps,
        q_scale);
}

} // namespace yuuka
