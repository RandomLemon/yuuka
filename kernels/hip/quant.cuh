#pragma once

#include "common.cuh"

#include "yuuka_hip.h"

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

__device__ __forceinline__ float q4k_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    const float dm = load_f16_device(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t * qs = blk + 16;
    const int group = j / 64;
    const int within = j % 64;
    const int sub = group * 2 + (within >= 32 ? 1 : 0);
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min_k4(sub, scales, sc, m);
    const int lane = within % 32;
    const uint8_t q = within < 32 ? (qs[group * 32 + lane] & 0xF)
                                  : (qs[group * 32 + lane] >> 4);
    return d * sc * q - dm * m;
}

__device__ __forceinline__ float q5k_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    const float dm = load_f16_device(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t * qh = blk + 16;
    const uint8_t * ql = blk + 48;
    const int group = j / 64;
    const int within = j % 64;
    const int sub = group * 2 + (within >= 32 ? 1 : 0);
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min_k4(sub, scales, sc, m);
    const int lane = within % 32;
    int q = within < 32 ? (ql[group * 32 + lane] & 0xF)
                        : (ql[group * 32 + lane] >> 4);
    const uint8_t mask = static_cast<uint8_t>(1u << (2 * group + (within < 32 ? 0 : 1)));
    if (qh[lane] & mask) {
        q += 16;
    }
    return d * sc * q - dm * m;
}

__device__ __forceinline__ float q6k_value(const uint8_t * blk, int j) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = load_f16_device(blk + 208);
    const int n = j / 128;
    const int l = j % 128;
    const uint8_t * qlN = ql + n * 64;
    const uint8_t * qhN = qh + n * 32;
    const int8_t * scN = sc + n * 8;
    int q = 0;
    int sidx = 0;
    if (l < 32) {
        q = (qlN[l] & 0xF) | (((qhN[l] >> 0) & 3) << 4);
        sidx = l / 16 + 0;
    } else if (l < 64) {
        q = (qlN[l] & 0xF) | (((qhN[l - 32] >> 2) & 3) << 4);
        sidx = (l - 32) / 16 + 2;
    } else if (l < 96) {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 64] >> 4) & 3) << 4);
        sidx = (l - 64) / 16 + 4;
    } else {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 96] >> 6) & 3) << 4);
        sidx = (l - 96) / 16 + 6;
    }
    return d * scN[sidx] * static_cast<float>(q - 32);
}

__device__ __forceinline__ float q8_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    return d * static_cast<float>(static_cast<int8_t>(blk[2 + j]));
}
