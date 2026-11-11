#include "yuuka_hip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int32_t exp = int32_t((bits >> 23) & 0xff) - 127;
    const uint32_t mant = bits & 0x7fffff;
    if (exp > 15) return uint16_t(sign | 0x7c00);
    if (exp < -24) return uint16_t(sign);
    if (exp < -14) {
        const uint32_t shift = uint32_t(-exp - 14);
        const uint32_t full = mant | 0x800000;
        return uint16_t(sign | ((full + (1u << (shift + 12))) >> (shift + 13)));
    }
    uint32_t rounded = mant + 0x1000;
    int32_t e = exp;
    if (rounded & 0x800000) {
        rounded = 0;
        e++;
        if (e > 15) return uint16_t(sign | 0x7c00);
    }
    return uint16_t(sign | ((e + 15) << 10) | (rounded >> 13));
}

void put_half(uint8_t * dst, float value) {
    const uint16_t half = float_to_half(value);
    dst[0] = uint8_t(half & 0xff);
    dst[1] = uint8_t(half >> 8);
}

void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

float dot_q4_k(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 256; ++block) {
        const uint8_t * blk = row + block * 144;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const float dm = half_to_float(uint16_t(blk[2] | (blk[3] << 8)));
        const uint8_t * scales = blk + 4;
        const uint8_t * qs = blk + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(is, scales, s0, m0);
            get_scale_min_k4(is + 1, scales, s1, m1);
            const float d1 = d * s0, min1 = dm * m0;
            const float d2 = d * s1, min2 = dm * m1;
            for (int l = 0; l < 32; ++l) {
                sum += x[block * 256 + j + l] * (d1 * (qs[l] & 0xF) - min1);
                sum += x[block * 256 + j + 32 + l] * (d2 * (qs[l] >> 4) - min2);
            }
            qs += 32;
            is += 2;
        }
    }
    return sum;
}

float dot_q5_k(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 256; ++block) {
        const uint8_t * blk = row + block * 176;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const float dm = half_to_float(uint16_t(blk[2] | (blk[3] << 8)));
        const uint8_t * scales = blk + 4;
        const uint8_t * qh = blk + 16;
        const uint8_t * ql = blk + 48;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(is, scales, s0, m0);
            get_scale_min_k4(is + 1, scales, s1, m1);
            const float d1 = d * s0, min1 = dm * m0;
            const float d2 = d * s1, min2 = dm * m1;
            const int g = j / 64;
            for (int l = 0; l < 32; ++l) {
                int q0 = ql[l] & 0xF;
                if (qh[l] & (1u << (2 * g))) q0 += 16;
                int q1 = ql[l] >> 4;
                if (qh[l] & (1u << (2 * g + 1))) q1 += 16;
                sum += x[block * 256 + j + l] * (d1 * q0 - min1);
                sum += x[block * 256 + j + 32 + l] * (d2 * q1 - min2);
            }
            ql += 32;
            is += 2;
        }
    }
    return sum;
}

float dot_q6_k(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 256; ++block) {
        const uint8_t * blk = row + block * 210;
        const uint8_t * ql = blk;
        const uint8_t * qh = blk + 128;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = half_to_float(uint16_t(blk[208] | (blk[209] << 8)));
        for (int half = 0; half < 2; ++half) {
            const uint8_t * qlN = ql + half * 64;
            const uint8_t * qhN = qh + half * 32;
            const int8_t * scN = sc + half * 8;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int8_t((qlN[l] & 0xF) | (((qhN[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int8_t((qlN[l + 32] & 0xF) | (((qhN[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int8_t((qlN[l] >> 4) | (((qhN[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int8_t((qlN[l + 32] >> 4) | (((qhN[l] >> 6) & 3) << 4)) - 32;
                sum += x[block * 256 + half * 128 + l] * d * scN[is + 0] * q1;
                sum += x[block * 256 + half * 128 + l + 32] * d * scN[is + 2] * q2;
                sum += x[block * 256 + half * 128 + l + 64] * d * scN[is + 4] * q3;
                sum += x[block * 256 + half * 128 + l + 96] * d * scN[is + 6] * q4;
            }
        }
    }
    return sum;
}

float dot_q8_0(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 32; ++block) {
        const uint8_t * blk = row + block * 34;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        for (int l = 0; l < 32; ++l) {
            sum += x[block * 32 + l] * d * int8_t(blk[2 + l]);
        }
    }
    return sum;
}

float dot_f32(const uint8_t * row, const float * x, int64_t n) {
    const float * w = reinterpret_cast<const float *>(row);
    float sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += x[i] * w[i];
    }
    return sum;
}

float dot_f16(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += x[i] * half_to_float(uint16_t(row[i * 2] | (row[i * 2 + 1] << 8)));
    }
    return sum;
}

float dot_q4_0(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 32; ++block) {
        const uint8_t * blk = row + block * 18;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        for (int l = 0; l < 16; ++l) {
            sum += x[block * 32 + l] * d * float((blk[2 + l] & 0xF) - 8);
            sum += x[block * 32 + l + 16] * d * float((blk[2 + l] >> 4) - 8);
        }
    }
    return sum;
}

float dot_q5_0(const uint8_t * row, const float * x, int64_t n) {
    float sum = 0;
    for (int64_t block = 0; block < n / 32; ++block) {
        const uint8_t * blk = row + block * 22;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const uint32_t qh = uint32_t(blk[2]) | (uint32_t(blk[3]) << 8) |
                            (uint32_t(blk[4]) << 16) | (uint32_t(blk[5]) << 24);
        for (int l = 0; l < 16; ++l) {
            const int xh0 = int(((qh >> l) << 4) & 0x10);
            const int xh1 = int((qh >> (l + 12)) & 0x10);
            const int q0 = int(blk[6 + l] & 0xF) | xh0;
            const int q1 = int(blk[6 + l] >> 4) | xh1;
            sum += x[block * 32 + l] * d * float(q0 - 16);
            sum += x[block * 32 + l + 16] * d * float(q1 - 16);
        }
    }
    return sum;
}

using DotFn = float (*)(const uint8_t *, const float *, int64_t);

uint64_t row_bytes(yuuka_dtype dtype, int64_t n) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return uint64_t(n) * 4;
        case YUUKA_DTYPE_F16: return uint64_t(n) * 2;
        case YUUKA_DTYPE_Q4_0: return uint64_t(n / 32) * 18;
        case YUUKA_DTYPE_Q5_0: return uint64_t(n / 32) * 22;
        case YUUKA_DTYPE_Q8_0: return uint64_t(n / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return uint64_t(n / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return uint64_t(n / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return uint64_t(n / 256) * 210;
        default: return 0;
    }
}

DotFn dot_for(yuuka_dtype dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return dot_f32;
        case YUUKA_DTYPE_F16: return dot_f16;
        case YUUKA_DTYPE_Q4_0: return dot_q4_0;
        case YUUKA_DTYPE_Q5_0: return dot_q5_0;
        case YUUKA_DTYPE_Q8_0: return dot_q8_0;
        case YUUKA_DTYPE_Q4_K: return dot_q4_k;
        case YUUKA_DTYPE_Q5_K: return dot_q5_k;
        case YUUKA_DTYPE_Q6_K: return dot_q6_k;
        default: return nullptr;
    }
}

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

struct Buffer {
    yuuka_buffer * buf = nullptr;
    uint64_t bytes = 0;
};

bool alloc(yuuka_device * device, Buffer & out, uint64_t bytes) {
    out.bytes = bytes;
    return check(yuuka_buffer_alloc(device, bytes, &out.buf), "buffer alloc");
}

void release(Buffer & b) {
    if (b.buf != nullptr) {
        yuuka_buffer_free(b.buf);
        b.buf = nullptr;
    }
}

bool upload(const Buffer & b, const void * data, uint64_t bytes) {
    return check(yuuka_memcpy_h2d(b.buf, data, bytes), "h2d");
}

bool download(const Buffer & b, void * data, uint64_t bytes) {
    return check(yuuka_memcpy_d2h(data, b.buf, bytes), "d2h");
}

template <typename T>
bool upload_vec(const Buffer & b, const std::vector<T> & data) {
    return upload(b, data.data(), data.size() * sizeof(T));
}

template <typename T>
bool download_vec(const Buffer & b, std::vector<T> & data) {
    return download(b, data.data(), data.size() * sizeof(T));
}

bool execute(yuuka_plan * plan, yuuka_stream * stream, yuuka_event * event,
             uint32_t position) {
    if (!check(yuuka_plan_execute(plan, stream, 0, position, 1, event), "plan execute")) {
        return false;
    }
    return check(yuuka_event_wait(event), "event wait");
}

std::vector<uint8_t> pack_weights(yuuka_dtype dtype, std::mt19937 & rng, int64_t n_in,
                                  int64_t n_out) {
    std::vector<uint8_t> packed(row_bytes(dtype, n_in) * uint64_t(n_out));
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    for (int64_t row = 0; row < n_out; ++row) {
        uint8_t * dst = packed.data() + row * row_bytes(dtype, n_in);
        if (dtype == YUUKA_DTYPE_F32) {
            for (int64_t i = 0; i < n_in; ++i) {
                const float v = dist(rng);
                std::memcpy(dst + i * 4, &v, 4);
            }
        } else if (dtype == YUUKA_DTYPE_F16) {
            for (int64_t i = 0; i < n_in; ++i) {
                put_half(dst + i * 2, dist(rng));
            }
        } else if (dtype == YUUKA_DTYPE_Q4_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                put_half(dst + b * 18, 0.01f);
                for (int i = 0; i < 16; ++i) dst[b * 18 + 2 + i] = uint8_t(rng() & 0xff);
            }
        } else if (dtype == YUUKA_DTYPE_Q5_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                put_half(dst + b * 22, 0.01f);
                for (int i = 0; i < 20; ++i) dst[b * 22 + 2 + i] = uint8_t(rng() & 0xff);
            }
        } else if (dtype == YUUKA_DTYPE_Q8_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                put_half(dst + b * 34, 0.01f);
                for (int i = 0; i < 32; ++i) {
                    dst[b * 34 + 2 + i] = uint8_t(int8_t(std::uniform_int_distribution<int>(-127, 127)(rng)));
                }
            }
        } else if (dtype == YUUKA_DTYPE_Q4_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * blk = dst + b * 144;
                put_half(blk, 0.002f);
                put_half(blk + 2, 0.001f);
                for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() & 0xff);
                for (int i = 0; i < 128; ++i) blk[16 + i] = uint8_t(rng() & 0xff);
            }
        } else if (dtype == YUUKA_DTYPE_Q5_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * blk = dst + b * 176;
                put_half(blk, 0.002f);
                put_half(blk + 2, 0.001f);
                for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() & 0xff);
                for (int i = 0; i < 160; ++i) blk[16 + i] = uint8_t(rng() & 0xff);
            }
        } else if (dtype == YUUKA_DTYPE_Q6_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * blk = dst + b * 210;
                for (int i = 0; i < 192; ++i) blk[i] = uint8_t(rng() & 0xff);
                for (int i = 0; i < 16; ++i) blk[192 + i] = uint8_t(int8_t(std::uniform_int_distribution<int>(-10, 10)(rng)));
                put_half(blk + 208, 0.002f);
            }
        }
    }
    return packed;
}

bool test_gemm(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
               yuuka_dtype dtype, const char * name, int64_t n_in, int64_t n_out,
               int64_t rows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    Buffer wb, xb, ob;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x);
    if (ok) {
        ok = check(yuuka_gemm_wmma(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                   dtype, event), "gemm wmma") &&
             check(yuuka_event_wait(event), "gemm wait");
    }
    std::vector<float> got(size_t(n_out * rows));
    ok = ok && download_vec(ob, got);
    const DotFn dot = dot_for(dtype);
    if (ok && dot != nullptr) {
        const uint64_t rb = row_bytes(dtype, n_in);
        double max_abs = 0.0;
        for (int64_t row = 0; row < rows && ok; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                const uint8_t * wrow = packed.data() + col * rb;
                const float expected = dot(wrow, x.data() + row * n_in, n_in);
                const float value = got[size_t(row * n_out + col)];
                const double diff = std::fabs(double(value) - double(expected));
                max_abs = std::max(max_abs, diff);
                if (diff > 2e-2 * (1.0 + std::fabs(double(expected)))) {
                    std::fprintf(stderr, "%s [%lld,%lld]: got %.6f expected %.6f\n",
                                 name, static_cast<long long>(row),
                                 static_cast<long long>(col), value, expected);
                    ok = false;
                    break;
                }
            }
        }
        std::printf("%s: %s (max abs %.3g)\n", name, ok ? "ok" : "FAILED", max_abs);
    }
    release(ob);
    release(xb);
    release(wb);
    return ok;
}

bool test_blas(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
               std::mt19937 & rng) {
    const int64_t n_in = 256, n_out = 96, rows = 17;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    std::vector<uint8_t> packed = pack_weights(YUUKA_DTYPE_F16, rng, n_in, n_out);
    std::vector<uint16_t> x16(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = dist(rng);
        x16[i] = float_to_half(x[i]);
    }
    Buffer wb, xb, ob;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x16.size() * 2) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x16);
    if (ok) {
        ok = check(yuuka_gemm_blas(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                   event), "gemm blas") &&
             check(yuuka_event_wait(event), "blas wait");
    }
    std::vector<float> got(size_t(n_out * rows));
    ok = ok && download_vec(ob, got);
    const uint64_t rb = row_bytes(YUUKA_DTYPE_F16, n_in);
    if (ok) {
        for (int64_t row = 0; row < rows && ok; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                const float expected = dot_f16(packed.data() + col * rb,
                                               x.data() + row * n_in, n_in);
                if (std::fabs(got[size_t(row * n_out + col)] - expected) >
                    1e-3f * (1.0f + std::fabs(expected))) {
                    std::fprintf(stderr, "blas mismatch at [%lld,%lld]\n",
                                 static_cast<long long>(row), static_cast<long long>(col));
                    ok = false;
                    break;
                }
            }
        }
    }
    std::printf("rocBLAS F16 baseline: %s\n", ok ? "ok" : "FAILED");

    // The plan path must produce the same result through an explicit cast.
    Buffer src32, dst16;
    yuuka_plan * plan = nullptr;
    bool plan_ok = ok && alloc(device, src32, x.size() * 4) &&
                   alloc(device, dst16, x.size() * 2) && upload_vec(src32, x) &&
                   check(yuuka_plan_create(device, &plan), "plan create");
    if (plan_ok) {
        yuuka_op cast{};
        cast.code = YUUKA_OP_CAST_F16;
        cast.buf[0] = src32.buf;
        cast.buf[1] = dst16.buf;
        cast.n[0] = uint64_t(x.size());
        plan_ok = check(yuuka_plan_add(plan, &cast), "add cast");
        yuuka_op gemm{};
        gemm.code = YUUKA_OP_GEMM;
        gemm.buf[0] = wb.buf;
        gemm.buf[1] = dst16.buf;
        gemm.buf[2] = ob.buf;
        gemm.n[0] = uint64_t(n_in);
        gemm.n[1] = uint64_t(n_out);
        gemm.n[2] = uint64_t(rows);
        gemm.u[0] = YUUKA_DTYPE_F16;
        gemm.u[1] = YUUKA_GEMM_PATH_BLAS;
        plan_ok = plan_ok && check(yuuka_plan_add(plan, &gemm), "add blas gemm");
    }
    if (plan_ok) {
        plan_ok = execute(plan, stream, event, 0);
    }
    if (plan_ok) {
        std::vector<float> again(size_t(n_out * rows));
        plan_ok = download_vec(ob, again);
        double max_abs = 0;
        for (int64_t row = 0; row < rows && plan_ok; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                const float expected = dot_f16(packed.data() + col * rb,
                                               x.data() + row * n_in, n_in);
                max_abs = std::max(max_abs, double(std::fabs(again[size_t(row * n_out + col)] - expected)));
                if (std::fabs(again[size_t(row * n_out + col)] - expected) >
                    1e-3f * (1.0f + std::fabs(expected))) {
                    plan_ok = false;
                    break;
                }
            }
        }
        std::printf("rocBLAS F16 plan path: %s (max abs %.3g)\n", plan_ok ? "ok" : "FAILED", max_abs);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(dst16);
    release(src32);
    release(ob);
    release(xb);
    release(wb);
    return ok && plan_ok;
}

bool test_norm_and_rope(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                        std::mt19937 & rng) {
    const int rows = 7, cols = 64, heads = 4, head_dim = 16, stride = 2 * head_dim;
    const int rot = 16;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> src(rows * cols), weight(cols), dst(rows * cols);
    std::vector<float> hsrc(rows * heads * stride), hweight(head_dim);
    std::vector<float> hdst(rows * heads * stride);
    for (float & v : src) v = dist(rng);
    for (float & v : weight) v = dist(rng);
    for (float & v : hsrc) v = dist(rng);
    for (float & v : hweight) v = dist(rng);
    Buffer sb, wb, db, hsb, hwb, hdb;
    bool ok = alloc(device, sb, src.size() * 4) && alloc(device, wb, weight.size() * 4) &&
              alloc(device, db, dst.size() * 4) && alloc(device, hsb, hsrc.size() * 4) &&
              alloc(device, hwb, hweight.size() * 4) && alloc(device, hdb, hdst.size() * 4) &&
              upload_vec(sb, src) && upload_vec(wb, weight) &&
              upload_vec(hsb, hsrc) && upload_vec(hwb, hweight);
    yuuka_plan * plan = nullptr;
    const float eps = 1e-6f;
    if (ok) {
        ok = check(yuuka_plan_create(device, &plan), "plan create");
    }
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_RMSNORM_BATCH;
        op.buf[0] = sb.buf;
        op.buf[1] = wb.buf;
        op.buf[2] = db.buf;
        op.n[0] = cols;
        op.n[1] = rows;
        op.n[2] = cols;
        op.n[3] = cols;
        op.f[0] = eps;
        ok = check(yuuka_plan_add(plan, &op), "add rmsnorm batch");
    }
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_RMSNORM_HEADS_BATCH;
        op.buf[0] = hsb.buf;
        op.buf[1] = hwb.buf;
        op.buf[2] = hdb.buf;
        op.n[0] = head_dim;
        op.n[1] = heads;
        op.n[2] = stride;
        op.n[3] = rows;
        op.f[0] = eps;
        ok = check(yuuka_plan_add(plan, &op), "add head norm batch");
    }
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_ROPE_BATCH;
        op.buf[0] = hdb.buf;
        op.n[0] = heads;
        op.n[1] = stride;
        op.n[2] = head_dim;
        op.n[3] = rot;
        op.n[4] = rows;
        op.n[5] = heads * stride;
        op.f[0] = 10000.0f;
        ok = check(yuuka_plan_add(plan, &op), "add rope batch");
    }
    if (ok) {
        ok = execute(plan, stream, event, 3);
    }
    ok = ok && download_vec(db, dst) && download_vec(hdb, hdst);
    if (ok) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0;
            for (int i = 0; i < cols; ++i) sum += src[r * cols + i] * src[r * cols + i];
            const float scale = 1.0f / std::sqrt(sum / cols + eps);
            for (int i = 0; i < cols; ++i) {
                const float expected = src[r * cols + i] * scale * weight[i];
                if (std::fabs(dst[r * cols + i] - expected) > 1e-5f) {
                    std::fprintf(stderr, "rmsnorm batch mismatch\n");
                    ok = false;
                }
            }
        }
    }
    if (ok) {
        for (int r = 0; r < rows; ++r) {
            for (int h = 0; h < heads; ++h) {
                const int base = r * heads * stride + h * stride;
                float sum = 0;
                for (int i = 0; i < head_dim; ++i) sum += hsrc[base + i] * hsrc[base + i];
                const float nscale = 1.0f / std::sqrt(sum / head_dim + eps);
                for (int i = 0; i < head_dim; ++i) {
                    hdst[base + i] = hsrc[base + i] * nscale * hweight[i];
                }
                const float theta_base = float(3 + r);
                for (int i = 0; i < rot / 2; ++i) {
                    const float theta = theta_base * std::pow(10000.0f, -2.0f * i / rot);
                    const float c = std::cos(theta), s = std::sin(theta);
                    const float x0 = hdst[base + i];
                    const float x1 = hdst[base + i + rot / 2];
                    hdst[base + i] = x0 * c - x1 * s;
                    hdst[base + i + rot / 2] = x0 * s + x1 * c;
                }
            }
        }
        std::vector<float> got(rows * heads * stride);
        ok = download_vec(hdb, got);
        double max_abs = 0;
        if (ok) {
            for (size_t i = 0; i < got.size(); ++i) {
                max_abs = std::max(max_abs, double(std::fabs(got[i] - hdst[i])));
                if (std::fabs(got[i] - hdst[i]) > 1e-4f * (1.0f + std::fabs(hdst[i]))) {
                    ok = false;
                    break;
                }
            }
        }
        std::printf("batch norm + RoPE: %s (max abs %.3g)\n", ok ? "ok" : "FAILED", max_abs);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(hdb);
    release(hwb);
    release(hsb);
    release(db);
    release(wb);
    release(sb);
    return ok;
}

bool test_embedding_and_kv(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                           std::mt19937 & rng) {
    const int rows = 5, cols = 64, vocab = 8, capacity = 12, kv_dim = 32;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> table(size_t(vocab * cols));
    for (float & v : table) v = dist(rng);
    std::vector<int32_t> tokens = {3, 0, 7, 1, 3};
    std::vector<float> k(size_t(rows * kv_dim)), v(size_t(rows * kv_dim));
    for (float & x : k) x = dist(rng);
    for (float & x : v) x = dist(rng);
    Buffer tb, tk, ob, kb, vb, kc, vc;
    bool ok = alloc(device, tb, table.size() * 4) && alloc(device, tk, tokens.size() * 4) &&
              alloc(device, ob, size_t(rows * cols) * 4) && alloc(device, kb, k.size() * 4) &&
              alloc(device, vb, v.size() * 4) && alloc(device, kc, size_t(capacity * kv_dim) * 4) &&
              alloc(device, vc, size_t(capacity * kv_dim) * 4) && upload_vec(tb, table) &&
              upload_vec(tk, tokens) && upload_vec(kb, k) && upload_vec(vb, v);
    yuuka_plan * plan = nullptr;
    if (ok) ok = check(yuuka_plan_create(device, &plan), "plan create");
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_EMBEDDING_BATCH;
        op.buf[0] = tb.buf;
        op.buf[1] = ob.buf;
        op.buf[2] = tk.buf;
        op.n[0] = cols;
        op.n[1] = rows;
        op.n[2] = cols * 4;
        op.n[3] = vocab;
        op.u[0] = YUUKA_DTYPE_F32;
        ok = check(yuuka_plan_add(plan, &op), "add embedding batch");
    }
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_KV_WRITE_BATCH;
        op.buf[0] = kb.buf;
        op.buf[1] = vb.buf;
        op.buf[2] = kc.buf;
        op.buf[3] = vc.buf;
        op.n[0] = kv_dim;
        op.n[1] = rows;
        op.u[0] = capacity;
        ok = check(yuuka_plan_add(plan, &op), "add kv batch");
    }
    if (ok) ok = execute(plan, stream, event, 4);
    std::vector<float> got_emb(size_t(rows * cols)), got_k(size_t(capacity * kv_dim)),
        got_v(size_t(capacity * kv_dim));
    ok = ok && download_vec(ob, got_emb) && download_vec(kc, got_k) && download_vec(vc, got_v);
    if (ok) {
        for (int r = 0; r < rows && ok; ++r) {
            for (int i = 0; i < cols; ++i) {
                if (got_emb[size_t(r * cols + i)] != table[size_t(tokens[r] * cols + i)]) {
                    std::fprintf(stderr, "embedding batch mismatch\n");
                    ok = false;
                    break;
                }
            }
        }
        for (int r = 0; r < rows && ok; ++r) {
            for (int i = 0; i < kv_dim; ++i) {
                if (got_k[size_t((4 + r) * kv_dim + i)] != k[size_t(r * kv_dim + i)] ||
                    got_v[size_t((4 + r) * kv_dim + i)] != v[size_t(r * kv_dim + i)]) {
                    std::fprintf(stderr, "kv batch mismatch\n");
                    ok = false;
                    break;
                }
            }
        }
        std::printf("batch embedding + KV write: %s\n", ok ? "ok" : "FAILED");
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(vc);
    release(kc);
    release(vb);
    release(kb);
    release(ob);
    release(tk);
    release(tb);
    return ok;
}

bool test_attention_prefill(yuuka_device * device, yuuka_stream * stream,
                            yuuka_event * event, std::mt19937 & rng) {
    const int heads = 4, kv_heads = 2, head_dim = 16, kv_dim = kv_heads * head_dim;
    const int capacity = 24, position = 5, rows = 9;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> q(size_t(rows * heads * 2 * head_dim));
    std::vector<float> kc(size_t(capacity * kv_dim)), vc(size_t(capacity * kv_dim));
    for (float & v : q) v = dist(rng);
    for (float & v : kc) v = dist(rng);
    for (float & v : vc) v = dist(rng);
    const float scale = 1.0f / std::sqrt(float(head_dim));
    Buffer qb, kb, vb, ob;
    bool ok = alloc(device, qb, q.size() * 4) && alloc(device, kb, kc.size() * 4) &&
              alloc(device, vb, vc.size() * 4) && alloc(device, ob, size_t(rows * heads * head_dim) * 4) &&
              upload_vec(qb, q) && upload_vec(kb, kc) && upload_vec(vb, vc);
    yuuka_plan * plan = nullptr;
    if (ok) ok = check(yuuka_plan_create(device, &plan), "plan create");
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_ATTENTION_PREFILL;
        op.buf[0] = qb.buf;
        op.buf[1] = kb.buf;
        op.buf[2] = vb.buf;
        op.buf[3] = ob.buf;
        op.n[0] = kv_heads;
        op.n[1] = heads;
        op.n[2] = head_dim;
        op.n[3] = kv_dim;
        op.n[4] = rows;
        op.u[0] = capacity;
        op.f[0] = scale;
        ok = check(yuuka_plan_add(plan, &op), "add attention prefill");
    }
    if (ok) ok = execute(plan, stream, event, position);
    std::vector<float> got(size_t(rows * heads * head_dim));
    ok = ok && download_vec(ob, got);
    if (ok) {
        double max_abs = 0;
        for (int r = 0; r < rows && ok; ++r) {
            const int pos = position + r;
            for (int h = 0; h < heads; ++h) {
                const int kvh = h / (heads / kv_heads);
                const float * qp = q.data() + size_t(r * heads + h) * 2 * head_dim;
                std::vector<float> scores(pos + 1);
                float max_s = -INFINITY;
                for (int p = 0; p <= pos; ++p) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += qp[d] * kc[size_t(p * kv_dim + kvh * head_dim + d)];
                    }
                    scores[p] = dot * scale;
                    max_s = std::max(max_s, scores[p]);
                }
                float total = 0;
                for (float & s : scores) {
                    s = std::exp(s - max_s);
                    total += s;
                }
                for (int d = 0; d < head_dim; ++d) {
                    float value = 0;
                    for (int p = 0; p <= pos; ++p) {
                        value += scores[p] / total * vc[size_t(p * kv_dim + kvh * head_dim + d)];
                    }
                    const float gate = qp[head_dim + d];
                    value *= 1.0f / (1.0f + std::exp(-gate));
                    const float actual = got[size_t((r * heads + h) * head_dim + d)];
                    max_abs = std::max(max_abs, double(std::fabs(actual - value)));
                    if (std::fabs(actual - value) > 2e-4f * (1.0f + std::fabs(value))) {
                        std::fprintf(stderr, "attention prefill mismatch [%d,%d,%d]: %.6f vs %.6f\n",
                                     r, h, d, actual, value);
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
            }
        }
        std::printf("flash attention prefill: %s (max abs %.3g)\n", ok ? "ok" : "FAILED", max_abs);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(ob);
    release(vb);
    release(kb);
    release(qb);
    return ok;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

bool test_recurrent_chunk(yuuka_device * device, yuuka_stream * stream,
                          yuuka_event * event, std::mt19937 & rng) {
    const int rows = 128, groups = 16, heads = 32, state = 128, kernel = 4;
    const int inner = heads * state, conv_dim = 2 * groups * state + inner;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> qkv(size_t(rows * conv_dim)), z(size_t(rows * inner));
    std::vector<float> alpha(size_t(rows * heads)), beta(size_t(rows * heads));
    std::vector<float> conv_w(size_t(conv_dim * kernel)), dt(heads), a(heads), norm(state);
    for (float & v : qkv) v = dist(rng);
    for (float & v : z) v = dist(rng);
    for (float & v : alpha) v = dist(rng);
    for (float & v : beta) v = dist(rng);
    for (float & v : conv_w) v = dist(rng);
    for (float & v : dt) v = dist(rng) * 0.1f;
    // Negative a keeps the gated delta-net decay below one so a 128-row
    // reference run stays well-conditioned.
    for (float & v : a) v = -std::fabs(dist(rng)) - 0.05f;
    for (float & v : norm) v = dist(rng);
    std::vector<float> conv_state(size_t(conv_dim * (kernel - 1))), rec_state(size_t(heads * state * state));
    for (float & v : conv_state) v = dist(rng);
    for (float & v : rec_state) v = dist(rng);
    const std::vector<float> conv_state0 = conv_state;
    const std::vector<float> rec_state0 = rec_state;

    Buffer qb, zb, ab, bb, cwb, dtb, avb, nb, csb, rsb, convb, ob;
    bool ok = alloc(device, qb, qkv.size() * 4) && alloc(device, zb, z.size() * 4) &&
              alloc(device, ab, alpha.size() * 4) && alloc(device, bb, beta.size() * 4) &&
              alloc(device, cwb, conv_w.size() * 4) && alloc(device, dtb, dt.size() * 4) &&
              alloc(device, avb, a.size() * 4) && alloc(device, nb, norm.size() * 4) &&
              alloc(device, csb, conv_state.size() * 4) && alloc(device, rsb, rec_state.size() * 4) &&
              alloc(device, convb, qkv.size() * 4) && alloc(device, ob, z.size() * 4) &&
              upload_vec(qb, qkv) && upload_vec(zb, z) && upload_vec(ab, alpha) &&
              upload_vec(bb, beta) && upload_vec(cwb, conv_w) && upload_vec(dtb, dt) &&
              upload_vec(avb, a) && upload_vec(nb, norm) && upload_vec(csb, conv_state) &&
              upload_vec(rsb, rec_state);
    yuuka_plan * plan = nullptr;
    if (ok) ok = check(yuuka_plan_create(device, &plan), "plan create");
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_RECURRENT_CHUNK;
        op.buf[0] = csb.buf;
        op.buf[1] = rsb.buf;
        op.buf[2] = qb.buf;
        op.buf[3] = zb.buf;
        op.buf[4] = ab.buf;
        op.buf[5] = bb.buf;
        op.buf[6] = cwb.buf;
        op.buf[7] = dtb.buf;
        op.buf[8] = avb.buf;
        op.buf[9] = nb.buf;
        op.buf[10] = ob.buf;
        op.buf[11] = convb.buf;
        op.n[0] = rows;
        op.n[1] = groups;
        op.n[2] = heads;
        op.n[3] = state;
        op.n[4] = conv_dim;
        op.n[5] = inner;
        op.u[0] = kernel;
        op.f[0] = 1e-6f;
        op.f[1] = 1.0f / std::sqrt(float(state));
        ok = check(yuuka_plan_add(plan, &op), "add recurrent chunk");
    }
    if (ok) ok = execute(plan, stream, event, 0);
    std::vector<float> got(size_t(rows * inner));
    std::vector<float> got_conv_state(conv_state.size()), got_rec_state(rec_state.size());
    std::vector<float> got_conv(size_t(rows * conv_dim));
    ok = ok && download_vec(ob, got) && download_vec(csb, got_conv_state) &&
         download_vec(rsb, got_rec_state) && download_vec(convb, got_conv);
    if (ok) {
        std::vector<float> conv(size_t(rows * conv_dim));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < conv_dim; ++c) {
                const float * w = conv_w.data() + c * kernel;
                float * s = conv_state.data() + c * (kernel - 1);
                float sum = 0;
                for (int t = 0; t < kernel - 1; ++t) sum += w[t] * s[t];
                sum += w[kernel - 1] * qkv[size_t(r * conv_dim + c)];
                conv[size_t(r * conv_dim + c)] = sum / (1.0f + std::exp(-sum));
                for (int t = 0; t < kernel - 2; ++t) s[t] = s[t + 1];
                s[kernel - 2] = qkv[size_t(r * conv_dim + c)];
            }
        }
        const std::vector<float> raw_conv = conv;
        std::vector<float> expected(size_t(rows * inner));
        const int key_dim = groups * state;
        double conv_max = 0;
        for (size_t i = 0; i < raw_conv.size(); ++i) {
            conv_max = std::max(conv_max, double(std::fabs(got_conv[i] - raw_conv[i])));
        }
        if (conv_max > 1e-5) {
            std::fprintf(stderr, "recurrent conv scratch mismatch: %.3g\n", conv_max);
            ok = false;
        }
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < groups; ++g) {
                float qsum = 0, ksum = 0;
                for (int d = 0; d < state; ++d) {
                    const float qv = conv[size_t(r * conv_dim + g * state + d)];
                    const float kv = conv[size_t(r * conv_dim + key_dim + g * state + d)];
                    qsum += qv * qv;
                    ksum += kv * kv;
                }
                const float qscale = (1.0f / std::sqrt(float(state))) / std::sqrt(qsum + 1e-6f);
                const float kscale = 1.0f / std::sqrt(ksum + 1e-6f);
                for (int d = 0; d < state; ++d) {
                    conv[size_t(r * conv_dim + g * state + d)] *= qscale;
                    conv[size_t(r * conv_dim + key_dim + g * state + d)] *= kscale;
                }
            }
            for (int h = 0; h < heads; ++h) {
                const int kh = h % groups;
                float * matrix = rec_state.data() + h * state * state;
                const float * q = conv.data() + r * conv_dim + kh * state;
                const float * k = conv.data() + r * conv_dim + key_dim + kh * state;
                const float * v = conv.data() + r * conv_dim + 2 * key_dim + h * state;
                const float raw = alpha[size_t(r * heads + h)] + dt[h];
                const float softplus = raw > 20.0f ? raw : std::log1p(std::exp(raw));
                const float decay = std::exp(softplus * a[h]);
                const float b = 1.0f / (1.0f + std::exp(-beta[size_t(r * heads + h)]));
                for (int i = 0; i < state * state; ++i) matrix[i] *= decay;
                std::vector<float> sk(state, 0.0f), d(state, 0.0f);
                for (int i = 0; i < state; ++i) {
                    for (int j = 0; j < state; ++j) sk[j] += matrix[i * state + j] * k[i];
                }
                for (int j = 0; j < state; ++j) d[j] = (v[j] - sk[j]) * b;
                for (int i = 0; i < state; ++i) {
                    for (int j = 0; j < state; ++j) matrix[i * state + j] += k[i] * d[j];
                }
                std::vector<float> out(state, 0.0f);
                for (int i = 0; i < state; ++i) {
                    for (int j = 0; j < state; ++j) out[j] += matrix[i * state + j] * q[i];
                }
                float rsum = 0;
                for (float x : out) rsum += x * x;
                const float nscale = 1.0f / std::sqrt(rsum / state + 1e-6f);
                for (int j = 0; j < state; ++j) {
                    const float gate = z[size_t(r * inner + h * state + j)];
                    expected[size_t(r * inner + h * state + j)] =
                        out[j] * nscale * norm[j] * silu(gate);
                }
            }
        }
        double max_abs = 0;
        for (int rr = 0; rr < rows; ++rr) {
            double row_max = 0;
            for (int j = 0; j < inner; ++j) {
                const size_t i = size_t(rr * inner + j);
                row_max = std::max(row_max, double(std::fabs(got[i] - expected[i])));
            }
            max_abs = std::max(max_abs, row_max);
            if (row_max > 2e-4 * (1.0 + std::fabs(double(expected[size_t(rr * inner)])))) {
                std::fprintf(stderr, "recurrent chunk output mismatch at row %d: %.6g\n",
                             rr, row_max);
                ok = false;
                break;
            }
        }
        for (size_t i = 0; i < conv_state.size() && ok; ++i) {
            if (std::fabs(got_conv_state[i] - conv_state[i]) > 1e-5f) {
                std::fprintf(stderr, "recurrent conv state mismatch\n");
                ok = false;
            }
        }
        for (size_t i = 0; i < rec_state.size(); ++i) {
            if (std::fabs(double(got_rec_state[i]) - double(rec_state[i])) >
                2e-4 * (1.0 + std::fabs(double(rec_state[i])))) {
                std::fprintf(stderr, "recurrent matrix state mismatch at %zu\n", i);
                ok = false;
                break;
            }
        }
        // Second execution with the original state must be bitwise identical.
        if (upload_vec(csb, conv_state0) && upload_vec(rsb, rec_state0) &&
            execute(plan, stream, event, 0)) {
            std::vector<float> again(size_t(rows * inner));
            std::vector<float> conv_again(size_t(rows * conv_dim));
            if (download_vec(ob, again) && download_vec(convb, conv_again)) {
                double det = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    det = std::max(det, double(std::fabs(got[i] - again[i])));
                }
                double cdet = 0;
                for (size_t i = 0; i < conv_again.size(); ++i) {
                    cdet = std::max(cdet, double(std::fabs(got_conv[i] - conv_again[i])));
                }
                if (det != 0.0 || cdet != 0.0) {
                    std::fprintf(stderr,
                                 "recurrent chunk is not deterministic: out %.3g conv %.3g\n",
                                 det, cdet);
                    ok = false;
                }
            }
        }
        std::printf("recurrent chunk: %s (max abs %.3g)\n", ok ? "ok" : "FAILED", max_abs);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(ob);
    release(convb);
    release(rsb);
    release(csb);
    release(nb);
    release(avb);
    release(dtb);
    release(cwb);
    release(bb);
    release(ab);
    release(zb);
    release(qb);
    return ok;
}

} // namespace

int main() {
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
    if (!check(yuuka_stream_create(device, &stream), "stream") ||
        !check(yuuka_event_create(device, &event), "event")) {
        return 1;
    }
    if (yuuka_abi_version() != YUUKA_HIP_ABI_VERSION) {
        std::fprintf(stderr, "ABI version mismatch\n");
        return 1;
    }
    std::mt19937 rng(4321);
    bool ok = true;
    const struct {
        yuuka_dtype dtype;
        const char * name;
    } cases[] = {
        {YUUKA_DTYPE_F32, "wmma F32"},
        {YUUKA_DTYPE_F16, "wmma F16"},
        {YUUKA_DTYPE_Q4_0, "wmma Q4_0"},
        {YUUKA_DTYPE_Q5_0, "wmma Q5_0"},
        {YUUKA_DTYPE_Q8_0, "wmma Q8_0"},
        {YUUKA_DTYPE_Q4_K, "wmma Q4_K"},
        {YUUKA_DTYPE_Q5_K, "wmma Q5_K"},
        {YUUKA_DTYPE_Q6_K, "wmma Q6_K"},
    };
    const struct {
        int64_t n_in;
        int64_t n_out;
        int64_t rows;
    } shapes[] = {
        {256, 64, 1}, {256, 96, 7}, {512, 128, 16}, {768, 160, 33},
    };
    for (const auto & c : cases) {
        for (const auto & s : shapes) {
            ok = test_gemm(device, stream, event, c.dtype, c.name, s.n_in, s.n_out,
                           s.rows, rng) && ok;
        }
    }
    ok = test_blas(device, stream, event, rng) && ok;
    ok = test_norm_and_rope(device, stream, event, rng) && ok;
    ok = test_embedding_and_kv(device, stream, event, rng) && ok;
    ok = test_attention_prefill(device, stream, event, rng) && ok;
    ok = test_recurrent_chunk(device, stream, event, rng) && ok;

    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
