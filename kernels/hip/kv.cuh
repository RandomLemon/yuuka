#pragma once

#include "common.cuh"
#include "yuuka_hip.h"

// kv_row_bytes returns the stored byte size of one KV row. It mirrors the host
// validation in the C ABI and the Go planner.
__device__ __host__ __forceinline__ int64_t kv_row_bytes(uint32_t dtype, int64_t kv_dim) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return kv_dim * 4;
        case YUUKA_DTYPE_F16:
            return kv_dim * 2;
        case YUUKA_DTYPE_Q8_0:
            return ((kv_dim + 31) / 32) * 34;
        default:
            return 0;
    }
}

// kv_load reads one element from a packed KV pool row.
__device__ __forceinline__ float kv_load(const void * pool, uint32_t dtype,
                                         int64_t row, int64_t col,
                                         int64_t kv_dim) {
    if (dtype == YUUKA_DTYPE_F32) {
        return static_cast<const float *>(pool)[row * kv_dim + col];
    }
    if (dtype == YUUKA_DTYPE_F16) {
        return load_f16_device(static_cast<const uint8_t *>(pool) +
                               row * kv_dim * 2 + col * 2);
    }
    const int64_t row_bytes = ((kv_dim + 31) / 32) * 34;
    const uint8_t * block = static_cast<const uint8_t *>(pool) +
                            row * row_bytes + (col / 32) * 34;
    return load_f16_device(block) *
           static_cast<float>(static_cast<int8_t>(block[2 + col % 32]));
}

__device__ __forceinline__ void kv_store(void * pool, uint32_t dtype,
                                         int64_t row, int64_t col,
                                         int64_t kv_dim, float value) {
    if (dtype == YUUKA_DTYPE_F32) {
        static_cast<float *>(pool)[row * kv_dim + col] = value;
    } else if (dtype == YUUKA_DTYPE_F16) {
        uint8_t * dst = static_cast<uint8_t *>(pool) + row * kv_dim * 2 + col * 2;
        *reinterpret_cast<__half *>(dst) = __float2half(value);
    }
}

// kv_quantize_q8_row quantizes one FP32 row into a Q8_0 pool row. Each group of
// 32 columns is quantized independently, including a partial trailing group.
__device__ __forceinline__ void kv_quantize_q8_row(
    void * pool, int64_t row, const float * src, int64_t kv_dim,
    int kvThread) {
    const int64_t row_bytes = ((kv_dim + 31) / 32) * 34;
    uint8_t * dst = static_cast<uint8_t *>(pool) + row * row_bytes;
    for (int64_t group = kvThread / 32; group < (kv_dim + 31) / 32;
         group += blockDim.x / 32) {
        const int lane = kvThread % 32;
        const int64_t start = group * 32;
        const int64_t count = min<int64_t>(32, kv_dim - start);
        float value = (lane < count) ? src[start + lane] : 0.0f;
        float max_abs = fabsf(value);
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_xor(max_abs, offset));
        }
        const float scale = max_abs / 127.0f;
        const int q = (lane < count && scale != 0.0f)
                          ? static_cast<int>(roundf(value / scale))
                          : 0;
        if (lane == 0) {
            *reinterpret_cast<__half *>(dst + group * 34) = __float2half(scale);
        }
        if (lane < 32) {
            dst[group * 34 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q));
        }
    }
}
