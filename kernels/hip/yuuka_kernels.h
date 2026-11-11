#ifndef YUUKA_KERNELS_H
#define YUUKA_KERNELS_H

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace yuuka {

void launch_add_f32(float * dst, const float * a, const float * b, uint64_t n,
                    hipStream_t stream);

void launch_gemv_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q8_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q4_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q5_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q6_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);

void launch_gemm_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);
void launch_gemm_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);

void launch_rmsnorm_f32(float * dst, const float * src, const float * weight,
                        int64_t n, float eps, hipStream_t stream);
void launch_l2norm_f32(float * dst, const float * src, int64_t n, float eps,
                       float extra_scale, hipStream_t stream);
void launch_gated_rmsnorm_f32(float * dst, const float * src, const float * weight,
                              const float * gate, int64_t n, float eps, hipStream_t stream);
void launch_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                              int heads, int64_t n, int64_t stride, float eps,
                              hipStream_t stream);
void launch_l2norm_heads_f32(float * dst, const float * src, int heads, int64_t n,
                             int64_t stride, float eps, float extra_scale,
                             hipStream_t stream);
void launch_gated_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                                    const float * gate, int heads, int64_t n,
                                    int64_t stride, float eps, hipStream_t stream);
void launch_silu_mul_f32(float * out, const float * gate, const float * up,
                         int64_t n, hipStream_t stream);
void launch_add_f32_buffers(float * x, const float * delta, int64_t n, hipStream_t stream);
void launch_copy_f32(float * dst, const float * src, int64_t n, hipStream_t stream);
void launch_scale_f32(float * x, int64_t n, float scale, hipStream_t stream);
void launch_rope_f32(float * data, int heads, int stride, int head_dim, int rot_dim,
                     int pos, float base, hipStream_t stream);
void launch_attention_f32(float * out, const float * q, const float * k_cache,
                          const float * v_cache, float * scores,
                          int kv_heads, int heads, int head_dim, int kv_dim,
                          int position, int capacity, float scale, hipStream_t stream);
void launch_conv1d_f32(float * state, const float * weights, const float * qkv,
                       float * out, int channels, int kernel, hipStream_t stream);
void launch_gdn_f32(float * state, const float * q, const float * k, const float * v,
                    const float * alpha, const float * beta, const float * dt,
                    const float * a, float * out, int groups, int heads,
                    int state_size, hipStream_t stream);
void launch_embedding_f32(float * out, const void * weights, int32_t token,
                          uint64_t row_bytes, int64_t cols, uint32_t dtype,
                          hipStream_t stream);

} // namespace yuuka

#endif
