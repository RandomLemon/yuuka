#include "../common.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

__global__ void rmsnorm_f32_kernel(float * dst, const float * src, const float * weight,
                                   int64_t n, float eps) {
    __shared__ float scratch[kThreads];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += src[i] * src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i] * scale * weight[i];
    }
}

__global__ void l2norm_f32_kernel(float * dst, const float * src, int64_t n,
                                  float eps, float extra_scale) {
    __shared__ float scratch[kThreads];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += src[i] * src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum + eps) * extra_scale;
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i] * scale;
    }
}

__global__ void gated_rmsnorm_f32_kernel(float * dst, const float * src, const float * weight,
                                         const float * gate, int64_t n, float eps) {
    __shared__ float scratch[kThreads];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += src[i] * src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gate[i];
        const float silu = g / (1.0f + expf(-g));
        dst[i] = src[i] * scale * weight[i] * silu;
    }
}

__global__ void rmsnorm_heads_f32_kernel(float * dst, const float * src, const float * weight,
                                         int64_t heads, int64_t n, int64_t stride, float eps) {
    const int64_t head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float scratch[kThreads];
    const float * head_src = src + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += head_src[i] * head_src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        head_dst[i] = head_src[i] * scale * weight[i];
    }
}

__global__ void l2norm_heads_f32_kernel(float * dst, const float * src, int64_t heads,
                                        int64_t n, int64_t stride, float eps, float extra_scale) {
    const int64_t head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float scratch[kThreads];
    const float * head_src = src + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += head_src[i] * head_src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum + eps) * extra_scale;
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        head_dst[i] = head_src[i] * scale;
    }
}

__global__ void gated_rmsnorm_heads_f32_kernel(float * dst, const float * src, const float * weight,
                                               const float * gate, int64_t heads, int64_t n,
                                               int64_t stride, float eps) {
    const int64_t head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float scratch[kThreads];
    const float * head_src = src + head * stride;
    const float * head_gate = gate + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += head_src[i] * head_src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = head_gate[i];
        head_dst[i] = head_src[i] * scale * weight[i] * (g / (1.0f + expf(-g)));
    }
}

__global__ void silu_mul_f32_kernel(float * out, const float * gate, const float * up, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void add_f32_kernel_device(float * x, const float * delta, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] += delta[i];
    }
}

__global__ void copy_f32_kernel(float * dst, const float * src, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void scale_f32_kernel(float * x, int64_t n, float scale) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] *= scale;
    }
}

} // namespace

namespace yuuka {

void launch_rmsnorm_f32(float * dst, const float * src, const float * weight,
                        int64_t n, float eps, hipStream_t stream) {
    rmsnorm_f32_kernel<<<1, kThreads, 0, stream>>>(dst, src, weight, n, eps);
}

void launch_l2norm_f32(float * dst, const float * src, int64_t n, float eps,
                       float extra_scale, hipStream_t stream) {
    l2norm_f32_kernel<<<1, kThreads, 0, stream>>>(dst, src, n, eps, extra_scale);
}

void launch_gated_rmsnorm_f32(float * dst, const float * src, const float * weight,
                              const float * gate, int64_t n, float eps, hipStream_t stream) {
    gated_rmsnorm_f32_kernel<<<1, kThreads, 0, stream>>>(dst, src, weight, gate, n, eps);
}

void launch_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                              int heads, int64_t n, int64_t stride, float eps,
                              hipStream_t stream) {
    rmsnorm_heads_f32_kernel<<<heads, kThreads, 0, stream>>>(
        dst, src, weight, heads, n, stride, eps);
}

void launch_l2norm_heads_f32(float * dst, const float * src, int heads, int64_t n,
                             int64_t stride, float eps, float extra_scale,
                             hipStream_t stream) {
    l2norm_heads_f32_kernel<<<heads, kThreads, 0, stream>>>(
        dst, src, heads, n, stride, eps, extra_scale);
}

void launch_gated_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                                    const float * gate, int heads, int64_t n,
                                    int64_t stride, float eps, hipStream_t stream) {
    gated_rmsnorm_heads_f32_kernel<<<heads, kThreads, 0, stream>>>(
        dst, src, weight, gate, heads, n, stride, eps);
}

void launch_silu_mul_f32(float * out, const float * gate, const float * up,
                         int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    silu_mul_f32_kernel<<<blocks, kThreads, 0, stream>>>(out, gate, up, n);
}

void launch_add_f32_buffers(float * x, const float * delta, int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    add_f32_kernel_device<<<blocks, kThreads, 0, stream>>>(x, delta, n);
}

void launch_copy_f32(float * dst, const float * src, int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    copy_f32_kernel<<<blocks, kThreads, 0, stream>>>(dst, src, n);
}

void launch_scale_f32(float * x, int64_t n, float scale, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    scale_f32_kernel<<<blocks, kThreads, 0, stream>>>(x, n, scale);
}

} // namespace yuuka
