#include "../common.cuh"
#include "../quant.cuh"
#include "yuuka_hip.h"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

__global__ void embedding_f32_kernel(float * out, const uint8_t * weights, int32_t token,
                                     uint64_t row_bytes, int64_t cols, uint32_t dtype) {
    const uint8_t * row = weights + static_cast<uint64_t>(token) * row_bytes;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        out[i] = packed_value(dtype, row, i);
    }
}

__global__ void embedding_dynamic_f32_kernel(float * out, const uint8_t * weights,
                                             const int32_t * dynamic,
                                             uint64_t row_bytes, int64_t cols,
                                             uint32_t dtype) {
    const int32_t token = dynamic[0];
    const uint8_t * row = weights + static_cast<uint64_t>(token) * row_bytes;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        out[i] = packed_value(dtype, row, i);
    }
}

} // namespace

namespace yuuka {

void launch_embedding_f32(float * out, const void * weights, int32_t token,
                          uint64_t row_bytes, int64_t cols, uint32_t dtype,
                          hipStream_t stream) {
    embedding_f32_kernel<<<1, kThreads, 0, stream>>>(
        out, static_cast<const uint8_t *>(weights), token, row_bytes, cols, dtype);
}

void launch_embedding_dynamic_f32(float * out, const void * weights,
                                  const int32_t * dynamic, uint64_t row_bytes,
                                  int64_t cols, uint32_t dtype, hipStream_t stream) {
    embedding_dynamic_f32_kernel<<<1, kThreads, 0, stream>>>(
        out, static_cast<const uint8_t *>(weights), dynamic, row_bytes, cols, dtype);
}

} // namespace yuuka
