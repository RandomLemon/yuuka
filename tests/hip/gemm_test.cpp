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

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

bool run_case(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
              uint32_t dtype, const char * name, int64_t n_in, int64_t n_out,
              int64_t batch, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * batch));
    std::vector<float> w(size_t(n_in * n_out));
    for (float & value : x) value = dist(rng);
    for (float & value : w) value = dist(rng);

    std::vector<uint8_t> packed;
    const uint8_t * weight_data = reinterpret_cast<const uint8_t *>(w.data());
    uint64_t weight_bytes = uint64_t(w.size() * sizeof(float));
    if (dtype == YUUKA_DTYPE_F16) {
        packed.resize(w.size() * 2);
        for (size_t i = 0; i < w.size(); ++i) {
            const uint16_t h = float_to_half(w[i]);
            packed[i * 2] = uint8_t(h & 0xff);
            packed[i * 2 + 1] = uint8_t(h >> 8);
        }
        weight_data = packed.data();
        weight_bytes = packed.size();
    }

    yuuka_buffer * wbuf = nullptr;
    yuuka_buffer * xbuf = nullptr;
    yuuka_buffer * obuf = nullptr;
    bool ok = check(yuuka_buffer_alloc(device, weight_bytes, &wbuf), "alloc weights") &&
              check(yuuka_buffer_alloc(device, x.size() * 4, &xbuf), "alloc x") &&
              check(yuuka_buffer_alloc(device, size_t(n_out * batch) * 4, &obuf), "alloc out") &&
              check(yuuka_memcpy_h2d(wbuf, weight_data, weight_bytes), "h2d weights") &&
              check(yuuka_memcpy_h2d(xbuf, x.data(), x.size() * 4), "h2d x");
    if (ok) {
        ok = check(yuuka_gemm(stream, obuf, xbuf, wbuf, n_in, n_out, batch, dtype, event),
                   "gemm") &&
             check(yuuka_event_wait(event), "event wait");
    }
    std::vector<float> got(size_t(n_out * batch));
    ok = ok && check(yuuka_memcpy_d2h(got.data(), obuf, got.size() * 4), "d2h");
    if (ok) {
        for (int64_t row = 0; row < batch; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                float expected = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) {
                    float weight = w[size_t(col * n_in + i)];
                    if (dtype == YUUKA_DTYPE_F16) {
                        weight = half_to_float(uint16_t(packed[size_t(col * n_in + i) * 2] |
                                                        (packed[size_t(col * n_in + i) * 2 + 1] << 8)));
                    }
                    expected += x[size_t(row * n_in + i)] * weight;
                }
                const float value = got[size_t(row * n_out + col)];
                if (std::fabs(value - expected) > 2e-3f * (1.0f + std::fabs(expected))) {
                    std::fprintf(stderr, "%s [%lld,%lld]: got %.6f expected %.6f\n",
                                 name, static_cast<long long>(row),
                                 static_cast<long long>(col), value, expected);
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }
    std::printf("%s: %s\n", name, ok ? "ok" : "FAILED");
    yuuka_buffer_free(obuf);
    yuuka_buffer_free(xbuf);
    yuuka_buffer_free(wbuf);
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
    std::mt19937 rng(1234);
    bool ok = run_case(device, stream, event, YUUKA_DTYPE_F32, "F32", 37, 19, 7, rng) &&
              run_case(device, stream, event, YUUKA_DTYPE_F16, "F16", 37, 19, 7, rng) &&
              run_case(device, stream, event, YUUKA_DTYPE_F32, "F32 batch1", 512, 17, 1, rng);

    yuuka_buffer * dummy = nullptr;
    if (ok && check(yuuka_buffer_alloc(device, 4, &dummy), "alloc dummy")) {
        if (yuuka_gemm(stream, dummy, dummy, dummy, 256, 1, 1, YUUKA_DTYPE_Q4_K, nullptr) !=
            YUUKA_ERR_UNSUPPORTED) {
            std::fprintf(stderr, "quantized GEMM was not rejected\n");
            ok = false;
        }
        yuuka_buffer_free(dummy);
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
