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
            float d1 = d * s0, min1 = dm * m0;
            float d2 = d * s1, min2 = dm * m1;
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
            float d1 = d * s0, min1 = dm * m0;
            float d2 = d * s1, min2 = dm * m1;
            int g = j / 64;
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
                int is = l / 16;
                int q1 = int8_t((qlN[l] & 0xF) | (((qhN[l] >> 0) & 3) << 4)) - 32;
                int q2 = int8_t((qlN[l + 32] & 0xF) | (((qhN[l] >> 2) & 3) << 4)) - 32;
                int q3 = int8_t((qlN[l] >> 4) | (((qhN[l] >> 4) & 3) << 4)) - 32;
                int q4 = int8_t((qlN[l + 32] >> 4) | (((qhN[l] >> 6) & 3) << 4)) - 32;
                sum += x[block * 256 + half * 128 + l] * d * scN[is + 0] * q1;
                sum += x[block * 256 + half * 128 + l + 32] * d * scN[is + 2] * q2;
                sum += x[block * 256 + half * 128 + l + 64] * d * scN[is + 4] * q3;
                sum += x[block * 256 + half * 128 + l + 96] * d * scN[is + 6] * q4;
            }
        }
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

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

struct Case {
    const char * name;
    yuuka_dtype dtype;
    int row_bytes;
    float (*ref)(const uint8_t *, const float *, int64_t);
};

bool make_case(yuuka_dtype dtype, const char * name, int64_t n_in, Case & tc) {
    tc.name = name;
    tc.dtype = dtype;
    switch (dtype) {
        case YUUKA_DTYPE_F32: tc.row_bytes = int(n_in * 4); tc.ref = dot_f32; break;
        case YUUKA_DTYPE_F16: tc.row_bytes = int(n_in * 2); tc.ref = dot_f16; break;
        case YUUKA_DTYPE_Q4_0: tc.row_bytes = int(n_in / 32 * 18); tc.ref = dot_q4_0; break;
        case YUUKA_DTYPE_Q5_0: tc.row_bytes = int(n_in / 32 * 22); tc.ref = dot_q5_0; break;
        case YUUKA_DTYPE_Q8_0: tc.row_bytes = int(n_in / 32 * 34); tc.ref = dot_q8_0; break;
        case YUUKA_DTYPE_Q4_K: tc.row_bytes = int(n_in / 256 * 144); tc.ref = dot_q4_k; break;
        case YUUKA_DTYPE_Q5_K: tc.row_bytes = int(n_in / 256 * 176); tc.ref = dot_q5_k; break;
        case YUUKA_DTYPE_Q6_K: tc.row_bytes = int(n_in / 256 * 210); tc.ref = dot_q6_k; break;
        default: return false;
    }
    return true;
}

void fill_weights(uint8_t * base, yuuka_dtype dtype, int64_t n_in, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    if (dtype == YUUKA_DTYPE_F32) {
        for (int64_t i = 0; i < n_in; ++i) {
            const float value = dist(rng) * 0.1f;
            std::memcpy(base + i * 4, &value, sizeof(value));
        }
    } else if (dtype == YUUKA_DTYPE_F16) {
        for (int64_t i = 0; i < n_in; ++i) put_half(base + i * 2, dist(rng) * 0.1f);
    } else if (dtype == YUUKA_DTYPE_Q4_0) {
        for (int64_t b = 0; b < n_in / 32; ++b) {
            uint8_t * blk = base + b * 18;
            put_half(blk, 0.03125f);
            for (int i = 0; i < 16; ++i) blk[2 + i] = uint8_t(rng() % 256);
        }
    } else if (dtype == YUUKA_DTYPE_Q5_0) {
        for (int64_t b = 0; b < n_in / 32; ++b) {
            uint8_t * blk = base + b * 22;
            put_half(blk, 0.03125f);
            for (int i = 0; i < 20; ++i) blk[2 + i] = uint8_t(rng() % 256);
        }
    } else if (dtype == YUUKA_DTYPE_Q8_0) {
        for (int64_t b = 0; b < n_in / 32; ++b) {
            uint8_t * blk = base + b * 34;
            put_half(blk, 0.03125f);
            for (int i = 0; i < 32; ++i) blk[2 + i] = uint8_t(rng() % 256);
        }
    } else if (dtype == YUUKA_DTYPE_Q4_K) {
        for (int64_t b = 0; b < n_in / 256; ++b) {
            uint8_t * blk = base + b * 144;
            put_half(blk, 0.02f);
            put_half(blk + 2, 0.01f);
            for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() % 256);
            for (int i = 0; i < 128; ++i) blk[16 + i] = uint8_t(rng() % 256);
        }
    } else if (dtype == YUUKA_DTYPE_Q5_K) {
        for (int64_t b = 0; b < n_in / 256; ++b) {
            uint8_t * blk = base + b * 176;
            put_half(blk, 0.02f);
            put_half(blk + 2, 0.01f);
            for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() % 256);
            for (int i = 0; i < 32; ++i) blk[16 + i] = uint8_t(rng() % 256);
            for (int i = 0; i < 128; ++i) blk[48 + i] = uint8_t(rng() % 256);
        }
    } else if (dtype == YUUKA_DTYPE_Q6_K) {
        for (int64_t b = 0; b < n_in / 256; ++b) {
            uint8_t * blk = base + b * 210;
            for (int i = 0; i < 128; ++i) blk[i] = uint8_t(rng() % 256);
            for (int i = 0; i < 64; ++i) blk[128 + i] = uint8_t(rng() % 256);
            for (int i = 0; i < 16; ++i) blk[192 + i] = uint8_t(rng() % 256);
            put_half(blk + 208, 0.01f);
        }
    }
}

} // namespace

bool run_case(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
              yuuka_dtype dtype, const char * name, int64_t n_in, int64_t n_out,
              std::mt19937 & rng) {
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(n_in));
    for (float & value : x) value = xdist(rng);

    Case tc;
    if (!make_case(dtype, name, n_in, tc)) {
        return false;
    }

    std::vector<uint8_t> weights(static_cast<size_t>(tc.row_bytes) * n_out);
    std::vector<float> expected(n_out);
    for (int64_t row = 0; row < n_out; ++row) {
        uint8_t * base = weights.data() + row * tc.row_bytes;
        fill_weights(base, tc.dtype, n_in, rng);
        expected[row] = tc.ref(base, x.data(), n_in);
    }

    yuuka_buffer * wbuf = nullptr;
    yuuka_buffer * xbuf = nullptr;
    yuuka_buffer * obuf = nullptr;
    bool ok = check(yuuka_buffer_alloc(device, weights.size(), &wbuf), "alloc weights") &&
              check(yuuka_buffer_alloc(device, x.size() * 4, &xbuf), "alloc x") &&
              check(yuuka_buffer_alloc(device, n_out * 4, &obuf), "alloc out") &&
              check(yuuka_memcpy_h2d(wbuf, weights.data(), weights.size()), "h2d weights") &&
              check(yuuka_memcpy_h2d(xbuf, x.data(), x.size() * 4), "h2d x") &&
              check(yuuka_gemv(stream, obuf, xbuf, wbuf, n_in, n_out, tc.dtype, event), "gemv") &&
              check(yuuka_event_wait(event), "event wait");
    std::vector<float> got(n_out);
    ok = ok && check(yuuka_memcpy_d2h(got.data(), obuf, n_out * 4), "d2h");
    if (ok) {
        for (int64_t row = 0; row < n_out; ++row) {
            const float tolerance = 2e-3f * (1.0f + std::fabs(expected[row]));
            if (std::fabs(got[row] - expected[row]) > tolerance) {
                std::fprintf(stderr, "%s n_in=%lld n_out=%lld row %lld: got %.6f expected %.6f\n",
                             tc.name, static_cast<long long>(n_in),
                             static_cast<long long>(n_out),
                             static_cast<long long>(row), got[row], expected[row]);
                ok = false;
                break;
            }
        }
    }
    std::printf("%s n_in=%lld n_out=%lld: %s\n", tc.name,
                static_cast<long long>(n_in), static_cast<long long>(n_out),
                ok ? "ok" : "FAILED");
    yuuka_buffer_free(obuf);
    yuuka_buffer_free(xbuf);
    yuuka_buffer_free(wbuf);
    return ok;
}

bool run_batch_case(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                    yuuka_dtype dtype, const char * name, int64_t n_in, int64_t n_out,
                    int64_t batch, std::mt19937 & rng) {
    Case tc;
    if (!make_case(dtype, name, n_in, tc)) {
        return false;
    }
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(n_in) * batch);
    for (float & value : x) value = xdist(rng);

    std::vector<uint8_t> weights(static_cast<size_t>(tc.row_bytes) * n_out);
    for (int64_t row = 0; row < n_out; ++row) {
        fill_weights(weights.data() + row * tc.row_bytes, tc.dtype, n_in, rng);
    }
    std::vector<float> expected(static_cast<size_t>(n_out) * batch);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < n_out; ++row) {
            expected[b * n_out + row] = tc.ref(
                weights.data() + row * tc.row_bytes, x.data() + b * n_in, n_in);
        }
    }

    yuuka_buffer * wbuf = nullptr;
    yuuka_buffer * xbuf = nullptr;
    yuuka_buffer * obuf = nullptr;
    bool ok = check(yuuka_buffer_alloc(device, weights.size(), &wbuf), "alloc batch weights") &&
              check(yuuka_buffer_alloc(device, x.size() * 4, &xbuf), "alloc batch x") &&
              check(yuuka_buffer_alloc(device, expected.size() * 4, &obuf), "alloc batch out") &&
              check(yuuka_memcpy_h2d(wbuf, weights.data(), weights.size()), "h2d batch weights") &&
              check(yuuka_memcpy_h2d(xbuf, x.data(), x.size() * 4), "h2d batch x") &&
              check(yuuka_gemv_batch(stream, obuf, xbuf, wbuf, n_in, n_out, batch, tc.dtype, event),
                    "gemv batch") &&
              check(yuuka_event_wait(event), "batch event wait");
    std::vector<float> got(expected.size());
    ok = ok && check(yuuka_memcpy_d2h(got.data(), obuf, got.size() * 4), "d2h batch");
    if (ok) {
        for (size_t i = 0; i < expected.size(); ++i) {
            const float tolerance = 2e-3f * (1.0f + std::fabs(expected[i]));
            if (std::fabs(got[i] - expected[i]) > tolerance) {
                std::fprintf(stderr, "%s n_in=%lld n_out=%lld batch=%lld index %zu: "
                             "got %.6f expected %.6f\n",
                             tc.name, static_cast<long long>(n_in),
                             static_cast<long long>(n_out), static_cast<long long>(batch),
                             i, got[i], expected[i]);
                ok = false;
                break;
            }
        }
    }
    std::printf("%s n_in=%lld n_out=%lld batch=%lld: %s\n", tc.name,
                static_cast<long long>(n_in), static_cast<long long>(n_out),
                static_cast<long long>(batch), ok ? "ok" : "FAILED");
    yuuka_buffer_free(obuf);
    yuuka_buffer_free(xbuf);
    yuuka_buffer_free(wbuf);
    return ok;
}

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
    std::mt19937 rng(1234);
    bool ok = true;
    const int64_t q8_widths[] = {32, 96, 160, 544};
    const int64_t k_widths[] = {256, 512, 768};
    ok = ok && run_case(device, stream, event, YUUKA_DTYPE_F32, "F32", 37, 1, rng);
    ok = ok && run_case(device, stream, event, YUUKA_DTYPE_F32, "F32", 35, 7, rng);
    ok = ok && run_case(device, stream, event, YUUKA_DTYPE_F16, "F16", 35, 1, rng);
    ok = ok && run_case(device, stream, event, YUUKA_DTYPE_F16, "F16", 37, 7, rng);
    ok = ok && run_case(device, stream, event, YUUKA_DTYPE_F16, "F16", 512, 17, rng);
    for (int64_t width : q8_widths) {
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q4_0, "Q4_0", width, 1, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q4_0, "Q4_0", width, 17, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q5_0, "Q5_0", width, 1, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q5_0, "Q5_0", width, 17, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q8_0, "Q8_0", width, 1, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q8_0, "Q8_0", width, 17, rng);
    }
    for (int64_t width : k_widths) {
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q4_K, "Q4_K", width, 1, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q4_K, "Q4_K", width, 17, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q5_K, "Q5_K", width, 17, rng);
        ok = ok && run_case(device, stream, event, YUUKA_DTYPE_Q6_K, "Q6_K", width, 17, rng);
    }

    // The batched GEMV path reuses each packed weight element across rows and
    // must match the reference for every dtype and every measured batch class.
    // Large-K shapes cover the prompt prefill down projection.
    const int64_t batches[] = {1, 2, 3, 4, 8, 16};
    for (int64_t batch : batches) {
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_F32, "F32", 37, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_F16, "F16", 37, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q4_0, "Q4_0", 96, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q5_0, "Q5_0", 96, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q8_0, "Q8_0", 96, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q4_K, "Q4_K", 512, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q5_K, "Q5_K", 256, 17, batch, rng);
        ok = ok && run_batch_case(device, stream, event,
                                  YUUKA_DTYPE_Q6_K, "Q6_K", 768, 17, batch, rng);
    }
    ok = ok && run_batch_case(device, stream, event, YUUKA_DTYPE_Q4_K,
                              "Q4_K large-K", 9216, 2560, 2, rng);
    ok = ok && run_batch_case(device, stream, event, YUUKA_DTYPE_Q4_K,
                              "Q4_K large-K", 9216, 2560, 8, rng);
    ok = ok && run_batch_case(device, stream, event, YUUKA_DTYPE_Q6_K,
                              "Q6_K large-K", 9216, 2560, 2, rng);

    yuuka_buffer * dummy = nullptr;
    if (ok && check(yuuka_buffer_alloc(device, 4, &dummy), "alloc dummy")) {
        const struct { yuuka_dtype dtype; uint64_t width; } invalid[] = {
            {YUUKA_DTYPE_Q4_0, 33}, {YUUKA_DTYPE_Q5_0, 33},
            {YUUKA_DTYPE_Q4_K, 100}, {YUUKA_DTYPE_Q5_K, 100},
            {YUUKA_DTYPE_Q6_K, 100}, {YUUKA_DTYPE_Q8_0, 33},
        };
        for (const auto & item : invalid) {
            if (yuuka_gemv(stream, dummy, dummy, dummy, item.width, 1, item.dtype, nullptr) !=
                YUUKA_ERR_UNSUPPORTED) {
                std::fprintf(stderr, "invalid width %llu for dtype %u was not rejected\n",
                             static_cast<unsigned long long>(item.width), item.dtype);
                ok = false;
            }
        }
        if (yuuka_gemv(stream, dummy, dummy, dummy, 32, UINT64_MAX / 8,
                       YUUKA_DTYPE_Q8_0, nullptr) != YUUKA_ERR_INVALID_ARGUMENT) {
            std::fprintf(stderr, "overflowing output width was not rejected\n");
            ok = false;
        }

        // Plan construction must reject unsupported dtype/shape combinations
        // instead of submitting a silent no-op.
        yuuka_plan * plan = nullptr;
        if (ok && check(yuuka_plan_create(device, &plan), "plan create")) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMV;
            op.buf[0] = dummy;
            op.buf[1] = dummy;
            op.buf[2] = dummy;
            op.n[0] = 256;
            op.n[1] = 1;
            op.u[0] = 100; /* No device kernel or block layout. */
            if (yuuka_plan_add(plan, &op) != YUUKA_ERR_UNSUPPORTED) {
                std::fprintf(stderr, "unsupported plan dtype was not rejected\n");
                ok = false;
            }
            op.u[0] = YUUKA_DTYPE_Q4_K;
            op.n[0] = 100;
            if (yuuka_plan_add(plan, &op) != YUUKA_ERR_UNSUPPORTED) {
                std::fprintf(stderr, "misaligned plan width was not rejected\n");
                ok = false;
            }
            op.code = YUUKA_OP_GEMV_BATCH;
            op.u[0] = YUUKA_DTYPE_Q8_0;
            op.n[0] = 32;
            op.n[2] = 17;
            if (yuuka_plan_add(plan, &op) != YUUKA_ERR_INVALID_ARGUMENT) {
                std::fprintf(stderr, "batch 17 plan op was not rejected\n");
                ok = false;
            }
            op.n[2] = 1;
            op.buf[0] = nullptr;
            if (yuuka_plan_add(plan, &op) != YUUKA_ERR_INVALID_ARGUMENT) {
                std::fprintf(stderr, "null plan weight buffer was not rejected\n");
                ok = false;
            }
            op.buf[0] = dummy;
            if (yuuka_plan_add(plan, &op) != YUUKA_ERR_INVALID_ARGUMENT) {
                std::fprintf(stderr, "undersized plan weight buffer was not rejected\n");
                ok = false;
            }
            yuuka_plan_destroy(plan);
        }

        yuuka_buffer * vw = nullptr;
        yuuka_buffer * vx = nullptr;
        yuuka_buffer * vo = nullptr;
        plan = nullptr;
        if (check(yuuka_buffer_alloc(device, 34, &vw), "alloc valid w") &&
            check(yuuka_buffer_alloc(device, 32 * 4, &vx), "alloc valid x") &&
            check(yuuka_buffer_alloc(device, 4, &vo), "alloc valid out") &&
            check(yuuka_plan_create(device, &plan), "valid plan create")) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMV;
            op.buf[0] = vw;
            op.buf[1] = vx;
            op.buf[2] = vo;
            op.n[0] = 32;
            op.n[1] = 1;
            op.u[0] = YUUKA_DTYPE_Q8_0;
            if (yuuka_plan_add(plan, &op) != YUUKA_OK) {
                std::fprintf(stderr, "valid plan op was rejected: %s\n", yuuka_last_error());
                ok = false;
            }
            yuuka_plan_destroy(plan);
        }
        yuuka_buffer_free(vo);
        yuuka_buffer_free(vx);
        yuuka_buffer_free(vw);
        yuuka_buffer_free(dummy);
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
