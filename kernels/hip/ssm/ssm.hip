#include "../common.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

__global__ void conv1d_f32_kernel(float * state, const float * weights, const float * qkv,
                                  float * out, int channels, int kernel) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        const float * w = weights + c * kernel;
        float * s = state + c * (kernel - 1);
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
}

__global__ void gdn_f32_kernel(float * state, const float * q, const float * k,
                               const float * v, const float * alpha, const float * beta_raw,
                               const float * dt, const float * a, float * out,
                               int groups, int heads, int state_size) {
    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    extern __shared__ float smem[];
    float * qs = smem;
    float * ks = smem + state_size;
    float * ds = smem + 2 * state_size;
    __shared__ float decay_shared;
    __shared__ float beta_shared;
    const int tid = threadIdx.x;
    const int kh = head % groups;
    qs[tid] = q[kh * state_size + tid];
    ks[tid] = k[kh * state_size + tid];
    const float value = v[head * state_size + tid];
    if (tid == 0) {
        const float raw = alpha[head] + dt[head];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        decay_shared = expf(softplus * a[head]);
        beta_shared = 1.0f / (1.0f + expf(-beta_raw[head]));
    }
    __syncthreads();
    const float decay = decay_shared;
    const float beta = beta_shared;
    float * matrix = state + head * state_size * state_size;
    for (int j = 0; j < state_size; ++j) {
        matrix[tid * state_size + j] *= decay;
    }
    __syncthreads();
    float sk = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        sk += matrix[i * state_size + tid] * ks[i];
    }
    ds[tid] = (value - sk) * beta;
    __syncthreads();
    const float ki = ks[tid];
    for (int j = 0; j < state_size; ++j) {
        matrix[tid * state_size + j] += ki * ds[j];
    }
    __syncthreads();
    float result = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        result += matrix[i * state_size + tid] * qs[i];
    }
    out[head * state_size + tid] = result;
}

} // namespace

namespace yuuka {

void launch_conv1d_f32(float * state, const float * weights, const float * qkv,
                       float * out, int channels, int kernel, hipStream_t stream) {
    const int blocks = (channels + kThreads - 1) / kThreads;
    conv1d_f32_kernel<<<blocks, kThreads, 0, stream>>>(state, weights, qkv, out, channels, kernel);
}

void launch_gdn_f32(float * state, const float * q, const float * k, const float * v,
                    const float * alpha, const float * beta, const float * dt,
                    const float * a, float * out, int groups, int heads,
                    int state_size, hipStream_t stream) {
    const size_t shared = size_t(3 * state_size) * sizeof(float);
    gdn_f32_kernel<<<heads, state_size, shared, stream>>>(
        state, q, k, v, alpha, beta, dt, a, out, groups, heads, state_size);
}

} // namespace yuuka
