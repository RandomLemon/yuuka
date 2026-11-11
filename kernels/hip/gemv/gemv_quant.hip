#include "../yuuka_kernels.h"

#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float load_f16(const void * ptr) {
    return __half2float(*reinterpret_cast<const __half *>(ptr));
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q,
                                                 uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__device__ __forceinline__ float block_reduce(float value) {
    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void gemv_f32_kernel(float * out, const float * x, const float * w,
                                int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    const float * row_ptr = w + row * n_in;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        sum += x[i] * row_ptr[i];
    }
    sum = block_reduce(sum);
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

__global__ void gemv_f16_kernel(float * out, const float * x, const void * w,
                                int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) + row * n_in * 2;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        sum += x[i] * load_f16(row_ptr + i * 2);
    }
    sum = block_reduce(sum);
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

__global__ void gemv_q8_0_kernel(float * out, const float * x, const void * w,
                                 int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) + row * (n_in / 32) * 34;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        const int64_t block = i / 32;
        const int64_t index = i % 32;
        const uint8_t * block_ptr = row_ptr + block * 34;
        const float d = load_f16(block_ptr);
        const int8_t q = static_cast<int8_t>(block_ptr[2 + index]);
        sum += x[i] * d * static_cast<float>(q);
    }
    sum = block_reduce(sum);
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

template <bool Q5>
__global__ void gemv_k_kernel(float * out, const float * x, const void * w,
                              int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    constexpr int kBlockBytes = Q5 ? 176 : 144;
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) + row * (n_in / 256) * kBlockBytes;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    const int64_t blocks = n_in / 256;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t * blk = row_ptr + b * kBlockBytes;
        const uint8_t * scales = blk + 4;
        const uint8_t * qh = Q5 ? blk + 16 : nullptr;
        const uint8_t * ql = Q5 ? blk + 48 : blk + 16;
        __shared__ float d_scale[8];
        __shared__ float d_min[8];
        if (tid < 8) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(tid, scales, sc, m);
            d_scale[tid] = load_f16(blk) * static_cast<float>(sc);
            d_min[tid] = load_f16(blk + 2) * static_cast<float>(m);
        }
        __syncthreads();
        const int group = tid / 64;
        const int within = tid % 64;
        const int lane = within < 32 ? within : within - 32;
        const int sub = within < 32 ? group * 2 : group * 2 + 1;
        uint8_t q = 0;
        if (within < 32) {
            q = ql[group * 32 + lane] & 0xF;
        } else {
            q = ql[group * 32 + lane] >> 4;
        }
        if (Q5) {
            const uint8_t mask = static_cast<uint8_t>(1u << (2 * group + (within < 32 ? 0 : 1)));
            if (qh[lane] & mask) {
                q = static_cast<uint8_t>(q | 16);
            }
        }
        const float value = d_scale[sub] * static_cast<float>(q) - d_min[sub];
        sum += x[b * 256 + tid] * value;
        __syncthreads();
    }
    sum = block_reduce(sum);
    if (tid == 0) {
        out[row] = sum;
    }
}

__global__ void gemv_q6_k_kernel(float * out, const float * x, const void * w,
                                 int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) + row * (n_in / 256) * 210;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    const int64_t blocks = n_in / 256;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t * blk = row_ptr + b * 210;
        const uint8_t * ql = blk;
        const uint8_t * qh = blk + 128;
        const int8_t * scales = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = load_f16(blk + 208);
        const int n = tid / 128;
        const int l = tid % 128;
        const uint8_t * ql_n = ql + n * 64;
        const uint8_t * qh_n = qh + n * 32;
        const int8_t * sc_n = scales + n * 8;
        int q = 0;
        int sidx = 0;
        if (l < 32) {
            q = (ql_n[l] & 0xF) | (((qh_n[l] >> 0) & 3) << 4);
            sidx = l / 16 + 0;
        } else if (l < 64) {
            q = (ql_n[l] & 0xF) | (((qh_n[l - 32] >> 2) & 3) << 4);
            sidx = (l - 32) / 16 + 2;
        } else if (l < 96) {
            q = (ql_n[l - 64] >> 4) | (((qh_n[l - 64] >> 4) & 3) << 4);
            sidx = (l - 64) / 16 + 4;
        } else {
            q = (ql_n[l - 64] >> 4) | (((qh_n[l - 96] >> 6) & 3) << 4);
            sidx = (l - 96) / 16 + 6;
        }
        const float value = d * static_cast<float>(sc_n[sidx]) * static_cast<float>(q - 32);
        sum += x[b * 256 + tid] * value;
    }
    sum = block_reduce(sum);
    if (tid == 0) {
        out[row] = sum;
    }
}


enum class GemvKind { F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K };

template <GemvKind Kind>
__device__ __forceinline__ float weight_at(const uint8_t * row, int64_t i);

template <GemvKind Kind>
__global__ void gemv_packed_kernel(float * out, const float * x, const void * w,
                                   int64_t n_in, int64_t n_out) {
    const int64_t row = blockIdx.x;
    if (row >= n_out) {
        return;
    }
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) +
                              row * ((Kind == GemvKind::Q4_0)
                                         ? (n_in / 32) * 18
                                         : (n_in / 32) * 22);
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        sum += x[i] * weight_at<Kind>(row_ptr, i);
    }
    sum = block_reduce(sum);
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

template <GemvKind Kind>
__device__ __forceinline__ int64_t batch_row_bytes(int64_t n_in) {
    if constexpr (Kind == GemvKind::F32) return n_in * 4;
    if constexpr (Kind == GemvKind::F16) return n_in * 2;
    if constexpr (Kind == GemvKind::Q4_0) return (n_in / 32) * 18;
    if constexpr (Kind == GemvKind::Q5_0) return (n_in / 32) * 22;
    if constexpr (Kind == GemvKind::Q8_0) return (n_in / 32) * 34;
    if constexpr (Kind == GemvKind::Q4_K) return (n_in / 256) * 144;
    if constexpr (Kind == GemvKind::Q5_K) return (n_in / 256) * 176;
    return (n_in / 256) * 210;
}

__device__ __forceinline__ float k_scale(const uint8_t * blk, int sub, bool min) {
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min_k4(sub, blk + 4, sc, m);
    return min ? load_f16(blk + 2) * static_cast<float>(m)
               : load_f16(blk) * static_cast<float>(sc);
}

template <GemvKind Kind>
__device__ __forceinline__ float weight_at(const uint8_t * row, int64_t i) {
    if constexpr (Kind == GemvKind::F32) {
        return reinterpret_cast<const float *>(row)[i];
    } else if constexpr (Kind == GemvKind::F16) {
        return load_f16(row + i * 2);
    } else if constexpr (Kind == GemvKind::Q4_0) {
        const uint8_t * blk = row + (i / 32) * 18;
        const int64_t j = i % 32;
        const uint8_t packed = blk[2 + (j & 15)];
        const int q = (j < 16) ? (packed & 0xF) : (packed >> 4);
        return load_f16(blk) * static_cast<float>(q - 8);
    } else if constexpr (Kind == GemvKind::Q5_0) {
        const uint8_t * blk = row + (i / 32) * 22;
        const int64_t j = i % 32;
        const uint8_t packed = blk[6 + (j & 15)];
        const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                            (static_cast<uint32_t>(blk[3]) << 8) |
                            (static_cast<uint32_t>(blk[4]) << 16) |
                            (static_cast<uint32_t>(blk[5]) << 24);
        const int high = (j < 16)
                             ? static_cast<int>(((qh >> j) << 4) & 0x10)
                             : static_cast<int>((qh >> ((j & 15) + 12)) & 0x10);
        const int q = (j < 16 ? (packed & 0xF) : (packed >> 4)) | high;
        return load_f16(blk) * static_cast<float>(q - 16);
    } else if constexpr (Kind == GemvKind::Q8_0) {
        const uint8_t * blk = row + (i / 32) * 34;
        return load_f16(blk) * static_cast<float>(static_cast<int8_t>(blk[2 + i % 32]));
    } else if constexpr (Kind == GemvKind::Q4_K) {
        const uint8_t * blk = row + (i / 256) * 144;
        const int64_t j = i % 256;
        const int sub = static_cast<int>(j / 32);
        const uint8_t * qs = blk + 16 + (j / 64) * 32;
        const int64_t within = j % 64;
        const uint8_t q = within < 32 ? (qs[within] & 0xF) : (qs[within - 32] >> 4);
        return k_scale(blk, sub, false) * static_cast<float>(q) - k_scale(blk, sub, true);
    } else if constexpr (Kind == GemvKind::Q5_K) {
        const uint8_t * blk = row + (i / 256) * 176;
        const int64_t j = i % 256;
        const int sub = static_cast<int>(j / 32);
        const uint8_t * qs = blk + 48 + (j / 64) * 32;
        const int64_t within = j % 64;
        uint8_t q = within < 32 ? (qs[within] & 0xF) : (qs[within - 32] >> 4);
        const int bit = 2 * static_cast<int>(j / 64) + (within < 32 ? 0 : 1);
        if (blk[16 + (within % 32)] & (1u << bit)) {
            q = static_cast<uint8_t>(q + 16);
        }
        return k_scale(blk, sub, false) * static_cast<float>(q) - k_scale(blk, sub, true);
    } else {
        const uint8_t * blk = row + (i / 256) * 210;
        const int64_t j = i % 256;
        const int half = static_cast<int>(j / 128);
        const int64_t within = j % 128;
        const int segment = static_cast<int>(within / 32);
        const int l = static_cast<int>(within % 32);
        const uint8_t * ql = blk + half * 64;
        const uint8_t * qh = blk + 128 + half * 32;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192 + half * 8);
        int q = 0;
        int si = 0;
        if (segment == 0) {
            q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            si = l / 16 + 0;
        } else if (segment == 1) {
            q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            si = l / 16 + 2;
        } else if (segment == 2) {
            q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            si = l / 16 + 4;
        } else {
            q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            si = l / 16 + 6;
        }
        return load_f16(blk + 208) * static_cast<float>(sc[si]) *
               static_cast<float>(q - 32);
    }
}

template <GemvKind Kind>
__global__ void gemv_batch_kernel(float * out, const float * x, const void * w,
                                  int64_t n_in, int64_t n_out, int64_t batch) {
    const int64_t row = blockIdx.x;
    if (row >= n_out || batch > 16) {
        return;
    }
    const uint8_t * row_ptr = static_cast<const uint8_t *>(w) +
                              row * batch_row_bytes<Kind>(n_in);
    const int tid = threadIdx.x;
    float sums[16];
    for (int b = 0; b < batch; ++b) {
        sums[b] = 0.0f;
    }
    for (int64_t i = tid; i < n_in; i += blockDim.x) {
        const float weight = weight_at<Kind>(row_ptr, i);
        for (int b = 0; b < batch; ++b) {
            sums[b] += x[b * n_in + i] * weight;
        }
    }
    __shared__ float scratch[kThreads];
    for (int b = 0; b < batch; ++b) {
        scratch[tid] = sums[b];
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[b * n_out + row] = scratch[0];
        }
        __syncthreads();
    }
}

} // namespace

namespace yuuka {

void launch_gemv_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_f32_kernel<<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_f16_kernel<<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q4_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_packed_kernel<GemvKind::Q4_0><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q5_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_packed_kernel<GemvKind::Q5_0><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q8_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_q8_0_kernel<<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q4_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_k_kernel<false><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q5_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_k_kernel<true><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_q6_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream) {
    gemv_q6_k_kernel<<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out));
}

void launch_gemv_batch_f32(float * out, const float * x, const float * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream) {
    gemv_batch_kernel<GemvKind::F32><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_f16(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream) {
    gemv_batch_kernel<GemvKind::F16><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q4_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q4_0><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q5_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q5_0><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q8_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q8_0><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q4_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q4_K><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q5_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q5_K><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

void launch_gemv_batch_q6_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream) {
    gemv_batch_kernel<GemvKind::Q6_K><<<static_cast<unsigned>(n_out), kThreads, 0, stream>>>(
        out, x, w, static_cast<int64_t>(n_in), static_cast<int64_t>(n_out), static_cast<int64_t>(batch));
}

} // namespace yuuka
