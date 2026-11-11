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

} // namespace yuuka

#endif
