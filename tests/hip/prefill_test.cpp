#include "yuuka_hip.h"

#include <algorithm>
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

// dequant_ref_value decodes one element of a packed K-quant row. It restates
// the format arithmetic of the dot_* reference decoders, so the dequant kernel
// has an independent oracle for every value it writes.
float dequant_ref_value(yuuka_dtype dtype, const uint8_t * row, int64_t index) {
    if (dtype == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (index / 256) * 144;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const float dm = half_to_float(uint16_t(blk[2] | (blk[3] << 8)));
        const int sub = int((index % 256) / 32);
        const int within = int(index & 31);
        uint8_t sc = 0, m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const uint8_t qs = blk[16 + (sub / 2) * 32 + within];
        const int q = (sub & 1) ? (qs >> 4) : (qs & 0xF);
        return d * float(sc) * float(q) - dm * float(m);
    }
    if (dtype == YUUKA_DTYPE_Q5_K) {
        const uint8_t * blk = row + (index / 256) * 176;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const float dm = half_to_float(uint16_t(blk[2] | (blk[3] << 8)));
        const int sub = int((index % 256) / 32);
        const int within = int(index & 31);
        uint8_t sc = 0, m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const uint8_t ql = blk[48 + (sub / 2) * 32 + within];
        const uint8_t qh = blk[16 + within];
        int q = (sub & 1) ? (ql >> 4) : (ql & 0xF);
        if (qh & uint8_t(1u << sub)) q += 16;
        return d * float(sc) * float(q) - dm * float(m);
    }
    const uint8_t * blk = row + (index / 256) * 210;
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = half_to_float(uint16_t(blk[208] | (blk[209] << 8)));
    const int n = int((index % 256) / 128);
    const int l = int(index & 127);
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
    return d * float(scN[sidx]) * float(q - 32);
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

// The short K-quant WMMA path rounds both operands to FP16 before
// accumulation. The rounded oracle isolates the kernel from the expected
// FP16 approximation, which the SwiGLU product can amplify.
float dot_rounded_kquant(yuuka_dtype dtype, const uint8_t * row, const float * x,
                          int64_t n_in) {
    double sum = 0.0;
    for (int64_t k = 0; k < n_in; ++k) {
        const float w = static_cast<float>(static_cast<_Float16>(dequant_ref_value(dtype, row, k)));
        const float a = static_cast<float>(static_cast<_Float16>(x[k]));
        sum += double(a) * w;
    }
    return static_cast<float>(sum);
}

// compare_gemm checks a GEMM result against the CPU dot-product reference at
// the documented tolerance.
bool compare_gemm(const char * name, const std::vector<uint8_t> & packed,
                  const std::vector<float> & x, const std::vector<float> & got,
                  yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows,
                  bool fp16_oracle = false) {
    const DotFn dot = dot_for(dtype);
    if (dot == nullptr) {
        return true;
    }
    const uint64_t rb = row_bytes(dtype, n_in);
    double max_abs = 0.0;
    bool ok = true;
    const double tolerance = fp16_oracle ? 1e-3 : 2e-2;
    for (int64_t row = 0; row < rows && ok; ++row) {
        for (int64_t col = 0; col < n_out; ++col) {
            const uint8_t * wrow = packed.data() + col * rb;
            const float * xrow = x.data() + row * n_in;
            const float expected = fp16_oracle ? dot_rounded_kquant(dtype, wrow, xrow, n_in)
                                               : dot(wrow, xrow, n_in);
            const float value = got[size_t(row * n_out + col)];
            const double diff = std::fabs(double(value) - double(expected));
            max_abs = std::max(max_abs, diff);
            if (!std::isfinite(diff) || diff > tolerance * (1.0 + std::fabs(double(expected)))) {
                std::fprintf(stderr, "%s [%lld,%lld]: got %.6f expected %.6f\n",
                             name, static_cast<long long>(row),
                             static_cast<long long>(col), value, expected);
                ok = false;
                break;
            }
        }
    }
    std::printf("%s: %s (max abs %.3g)\n", name, ok ? "ok" : "FAILED", max_abs);
    return ok;
}

constexpr float kGemmSentinel = -12345.0f;

template <typename Launch>
bool check_gemm_partial_output(Buffer & out, size_t valid,
                               std::vector<float> & got, Launch launch) {
    const std::vector<float> initial(got.size(), kGemmSentinel);
    std::vector<float> repeated(got.size());
    if (!upload_vec(out, initial) || !launch() || !download_vec(out, got) ||
        !upload_vec(out, initial) || !launch() || !download_vec(out, repeated)) {
        return false;
    }
    if (std::memcmp(got.data(), repeated.data(), got.size() * sizeof(float)) != 0 ||
        !std::all_of(got.begin() + valid, got.end(), [](float v) { return v == kGemmSentinel; })) {
        std::fprintf(stderr, "WMMA output changed between runs or wrote past the last row\n");
        return false;
    }
    return true;
}

bool test_gemm(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
               yuuka_dtype dtype, const char * name, int64_t n_in, int64_t n_out,
               int64_t rows, std::mt19937 & rng, bool fp16_oracle = false) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    std::vector<float> got(size_t(n_out * (rows + 64)));
    Buffer wb, xb, ob;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, got.size() * sizeof(float)) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x);
    ok = ok && check_gemm_partial_output(ob, size_t(n_out * rows), got, [&]() {
        return check(yuuka_gemm_wmma(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                    dtype, event), "gemm wmma") &&
               check(yuuka_event_wait(event), "gemm wait");
    });
    if (ok) {
        ok = compare_gemm(name, packed, x, got, dtype, n_in, n_out, rows, fp16_oracle);
    }
    release(ob);
    release(xb);
    release(wb);
    return ok;
}

// check_q8_1 re-derives the Q8_1 quantization on the host and compares it with
// the device output. Q8_1 activations are lossy by design, so the GEMM below is
// checked against the dequantized activations rather than the FP32 source; this
// function is what pins the quantization rule itself, digit for digit.
bool check_q8_1(const char * name, const std::vector<float> & x,
                const std::vector<int8_t> & qs, const std::vector<float> & ds,
                int64_t n_in, int64_t rows) {
    bool ok = true;
    int64_t mismatched = 0;
    double max_scale_diff = 0.0;
    double max_sum_diff = 0.0;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t block = 0; block < n_in / 32; ++block) {
            const float * src = x.data() + row * n_in + block * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[i]));
            const float d = amax / 127.0f;
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            int sum = 0;
            for (int i = 0; i < 32; ++i) {
                // nearbyint matches the device rintf rounding mode: ties to even.
                const int q = int(std::max(-127.0f,
                                           std::min(127.0f, std::nearbyint(src[i] * inv))));
                sum += q;
                const int8_t got = qs[size_t(row * n_in + block * 32 + i)];
                if (got != int8_t(q)) {
                    ++mismatched;
                }
            }
            const float ds_block = ds[size_t((row * (n_in / 32) + block) * 2)];
            const float ds_sum = ds[size_t((row * (n_in / 32) + block) * 2 + 1)];
            max_scale_diff = std::max(max_scale_diff, double(std::fabs(ds_block - d)));
            max_sum_diff =
                std::max(max_sum_diff, double(std::fabs(ds_sum - d * float(sum))));
        }
    }
    if (mismatched != 0) {
        std::fprintf(stderr, "%s: %lld quantized values differ from the host rule\n",
                     name, static_cast<long long>(mismatched));
        ok = false;
    }
    if (max_scale_diff > 1e-7 || max_sum_diff > 1e-4) {
        std::fprintf(stderr, "%s: q8_1 scale diff %.3g sum diff %.3g\n", name,
                     max_scale_diff, max_sum_diff);
        ok = false;
    }
    return ok;
}

// test_gemm_mmq runs the int8 MMQ route: Q8_1 activation quantization followed
// by the int8 WMMA GEMM. The quantizer is checked exactly, then the GEMM is
// checked against the CPU dot product over the dequantized Q8_1 activations at
// the documented GEMM tolerance.
bool test_gemm_mmq(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                   yuuka_dtype dtype, const char * name, int64_t n_in, int64_t n_out,
                   int64_t rows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    const uint64_t quant_bytes = uint64_t(n_in) * uint64_t(rows);
    const uint64_t scale_bytes = uint64_t(rows) * uint64_t(n_in / 32) * 8;
    Buffer wb, xb, qsb, dsb, ob;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, qsb, quant_bytes) && alloc(device, dsb, scale_bytes) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x);
    if (ok) {
        ok = check(yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf, n_in, rows,
                                       event), "q8_1 quantize") &&
             check(yuuka_event_wait(event), "quantize wait") &&
             check(yuuka_gemm_mmq(stream, ob.buf, qsb.buf, dsb.buf, wb.buf, n_in,
                                  n_out, rows, dtype, event), "gemm mmq") &&
             check(yuuka_event_wait(event), "mmq wait");
    }
    std::vector<float> got(size_t(n_out * rows));
    std::vector<int8_t> qs(static_cast<size_t>(quant_bytes));
    std::vector<float> ds(static_cast<size_t>(scale_bytes / 4));
    ok = ok && download_vec(ob, got) && download_vec(qsb, qs) &&
         download_vec(dsb, ds);
    if (ok) {
        ok = check_q8_1(name, x, qs, ds, n_in, rows);
    }
    if (ok) {
        // Dequantize the activations exactly as the kernel reads them.
        std::vector<float> xq(x.size());
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t block = 0; block < n_in / 32; ++block) {
                const float d = ds[size_t((row * (n_in / 32) + block) * 2)];
                for (int i = 0; i < 32; ++i) {
                    const int64_t index = row * n_in + block * 32 + i;
                    xq[size_t(index)] = d * float(qs[size_t(index)]);
                }
            }
        }
        ok = compare_gemm(name, packed, xq, got, dtype, n_in, n_out, rows);
    }
    release(ob);
    release(dsb);
    release(qsb);
    release(xb);
    release(wb);
    return ok;
}

// test_gemm_mmq_direct_control compares a selected direct MMQ cell at M=512
// with the generic tile on the same first 512 rows of an M=513 call. The
// activations are quantized separately for each row count, so the Q8_1 digits
// also have to agree row by row. Direct and generic differ only in staging and
// in how the two int8 WMMAs of a 32-value scale block are accumulated, which
// is exact int32 arithmetic followed by the same FP32 scale expression.
bool test_gemm_mmq_direct_control(yuuka_device * device, yuuka_stream * stream,
                                  yuuka_event * event, yuuka_dtype dtype,
                                  const char * name, int64_t n_in, int64_t n_out,
                                  std::mt19937 & rng) {
    constexpr int64_t rows = 512;
    constexpr int64_t control_rows = rows + 1;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * control_rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    const uint64_t qbytes = uint64_t(n_in) * uint64_t(control_rows);
    const uint64_t sbytes = uint64_t(control_rows) * uint64_t(n_in / 32) * 8;
    Buffer wb, xb, qsb, dsb, ob, ocb;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, qsb, qbytes) && alloc(device, dsb, sbytes) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              alloc(device, ocb, size_t(n_out * control_rows) * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x);
    if (ok) {
        ok = check(yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf, n_in, rows,
                                       event), "control q8_1 direct") &&
             check(yuuka_event_wait(event), "control quantize direct wait") &&
             check(yuuka_gemm_mmq(stream, ob.buf, qsb.buf, dsb.buf, wb.buf, n_in,
                                  n_out, rows, dtype, event), "control mmq direct") &&
             check(yuuka_event_wait(event), "control direct wait") &&
             check(yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf, n_in,
                                       control_rows, event), "control q8_1 generic") &&
             check(yuuka_event_wait(event), "control quantize generic wait") &&
             check(yuuka_gemm_mmq(stream, ocb.buf, qsb.buf, dsb.buf, wb.buf, n_in,
                                  n_out, control_rows, dtype, event), "control mmq generic") &&
             check(yuuka_event_wait(event), "control generic wait");
    }
    std::vector<float> got(size_t(n_out * rows));
    std::vector<float> control(size_t(n_out * control_rows));
    ok = ok && download_vec(ob, got) && download_vec(ocb, control);
    if (ok) {
        const size_t bytes = got.size() * sizeof(float);
        if (std::memcmp(got.data(), control.data(), bytes) != 0) {
            size_t first = 0;
            bool found = false;
            double max_diff = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (got[i] != control[i] && !found) {
                    first = i;
                    found = true;
                }
                max_diff = std::max(max_diff,
                                    double(std::fabs(got[i] - control[i])));
            }
            std::fprintf(stderr,
                         "%s direct/generic first difference at row %lld col %lld,"
                         " max abs %.3g\n", name,
                         static_cast<long long>(first / size_t(n_out)),
                         static_cast<long long>(first % size_t(n_out)), max_diff);
            ok = false;
        }
    }
    std::printf("%s direct vs generic M=%lld: %s\n", name,
                static_cast<long long>(rows), ok ? "ok" : "FAILED");
    release(ocb);
    release(ob);
    release(dsb);
    release(qsb);
    release(xb);
    release(wb);
    return ok;
}

bool test_gemm_mmq_row_tail(yuuka_device * device, yuuka_stream * stream,
                            yuuka_event * event, yuuka_dtype dtype, int64_t channels,
                            int64_t full_rows, std::mt19937 & rng) {
    constexpr int64_t width = 2560;
    constexpr float sentinel = -12345.0f;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(width * full_rows);
    for (float & value : x) value = dist(rng);
    const auto packed = pack_weights(dtype, rng, width, channels);
    const std::vector<float> initial(channels * full_rows, sentinel);
    Buffer wb, xb, qs, ds, out;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, qs, x.size()) && alloc(device, ds, x.size() / 32 * 8) &&
              alloc(device, out, initial.size() * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x);
    const auto run = [&](int64_t rows, std::vector<float> & output) {
        return upload_vec(out, initial) &&
               check(yuuka_gemm_mmq(stream, out.buf, qs.buf, ds.buf, wb.buf, width,
                                    channels, rows, dtype, event), "mmq tail gemm") &&
               check(yuuka_event_wait(event), "mmq tail wait") && download_vec(out, output);
    };
    std::vector<float> reference(initial.size()), got(initial.size());
    ok = ok && check(yuuka_quantize_q8_1(stream, qs.buf, ds.buf, xb.buf, width,
                                        full_rows, event), "mmq tail quantize") &&
         check(yuuka_event_wait(event), "mmq tail quantize wait") && run(full_rows, reference);
    if (ok && !std::all_of(reference.begin(), reference.end(), [](float v) {
            return std::isfinite(v) && v != sentinel;
        })) {
        std::fprintf(stderr, "MMQ full tile did not write all finite outputs\n");
        ok = false;
    }
    for (int64_t rows : {int64_t{129}, full_rows - 65, full_rows - 1}) {
        if (!ok || !run(rows, got)) {
            ok = false;
            break;
        }
        const size_t valid = size_t(rows * channels);
        if (std::memcmp(got.data(), reference.data(), valid * sizeof(float)) != 0 ||
            std::memcmp(got.data() + valid, initial.data() + valid,
                        (got.size() - valid) * sizeof(float)) != 0) {
            std::fprintf(stderr, "MMQ tail changed valid rows or wrote past row %lld\n",
                         static_cast<long long>(rows));
            ok = false;
            break;
        }
    }
    release(out);
    release(ds);
    release(qs);
    release(xb);
    release(wb);
    return ok;
}

bool test_direct_q6_mmq(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                        int64_t width, int64_t channels, std::mt19937 & rng) {
    constexpr int64_t full_rows = 512;
    constexpr int64_t block_values = 256;
    constexpr int64_t block_bytes = 210;
    const int64_t weight_stride = width / block_values * block_bytes;
    auto packed = pack_weights(YUUKA_DTYPE_Q6_K, rng, width, channels);
    std::vector<int8_t> weights(width * channels), digits(width * full_rows);
    std::vector<float> scales(channels * width / 16), activation_scales(full_rows * width / 16);
    for (int64_t n = 0; n < channels; ++n) {
        for (int64_t k = 0; k < width; k += block_values) {
            uint8_t * block = packed.data() + n * weight_stride + k / block_values * block_bytes;
            put_half(block + 208, (1 + rng() % 7) * 0.0005f);
            const float d = half_to_float(uint16_t(block[208] | block[209] << 8));
            for (int j = 0; j < 16; ++j) {
                const int magnitude = 1 + int(rng() % 31);
                const int sign = j % 2 == 0 ? 1 : -1;
                block[192 + j] = uint8_t(int8_t(sign * magnitude));
                scales[(n * width + k) / 16 + j] = d * float(sign * magnitude);
            }
            for (int j = 0; j < block_values; ++j) {
                const int half = j / 128;
                const int region = (j % 128) / 32;
                const int lane = j % 32;
                const uint8_t ql = block[half * 64 + (region % 2) * 32 + lane];
                const uint8_t qh = block[128 + half * 32 + lane];
                const int low = (ql >> (region >= 2 ? 4 : 0)) & 15;
                const int high = (qh >> (2 * region)) & 3;
                weights[n * width + k + j] = int8_t(low + 16 * high - 32);
            }
        }
    }
    for (size_t block = 0; block < digits.size() / 32; ++block) {
        const float dx = (1 + rng() % 5) * 0.001f;
        int total = 0;
        for (int j = 0; j < 32; ++j) {
            const int digit = int(rng() % 255) - 127;
            digits[32 * block + j] = int8_t(digit);
            total += digit;
        }
        activation_scales[2 * block] = dx;
        activation_scales[2 * block + 1] = dx * float(total);
    }
    std::vector<float> expected(full_rows * channels);
    for (int64_t r = 0; r < full_rows; ++r) {
        const int8_t * x = digits.data() + r * width;
        const float * ds = activation_scales.data() + r * (width / 16);
        for (int64_t n = 0; n < channels; ++n) {
            const int8_t * w = weights.data() + n * width;
            const float * dw = scales.data() + n * (width / 16);
            float acc = 0;
            for (int64_t k = 0; k < width; k += 16) {
                int sum = 0;
                for (int j = 0; j < 16; ++j) {
                    sum += int(x[k + j]) * int(w[k + j]);
                }
                acc = std::fma(dw[k / 16] * ds[(k / 32) * 2], float(sum), acc);
            }
            expected[r * channels + n] = acc;
        }
    }

    Buffer wb, qs, ds, out;
    std::vector<float> got((full_rows + 1) * channels);
    bool ok = alloc(device, wb, packed.size()) && upload_vec(wb, packed) &&
              alloc(device, qs, digits.size()) && upload_vec(qs, digits) &&
              alloc(device, ds, activation_scales.size() * sizeof(float)) &&
              upload_vec(ds, activation_scales) &&
              alloc(device, out, got.size() * sizeof(float));
    const auto verify = [&](int64_t rows, const char * route, auto launch) {
        const size_t valid = rows * channels;
        bool match = check_gemm_partial_output(out, valid, got, launch);
        double max_diff = 0;
        for (size_t i = 0; i < valid && match; ++i) {
            const double diff = std::fabs(double(got[i]) - double(expected[i]));
            match = std::isfinite(diff) && diff <= 2e-4;
            max_diff = std::max(max_diff, diff);
        }
        std::printf("Q6_K MMQ %lldx%lld M=%lld %s: %s (max abs %.3g)\n",
                    static_cast<long long>(width), static_cast<long long>(channels),
                    static_cast<long long>(rows), route, match ? "ok" : "FAILED", max_diff);
        return match;
    };
    for (int64_t rows : {127, 128, 129, 192, 511, 512}) {
        if (!ok) break;
        ok = verify(rows, "aligned", [&]() {
            return check(yuuka_gemm_mmq(stream, out.buf, qs.buf, ds.buf, wb.buf, width,
                                        channels, rows, YUUKA_DTYPE_Q6_K, event), "Q6_K MMQ") &&
                   check(yuuka_event_wait(event), "Q6_K MMQ wait");
        });
    }
    if (ok) {
        // Q6_K blocks need only two-byte alignment. Starting this buffer at
        // offset two exercises the generic path for an otherwise direct shape.
        std::vector<uint8_t> shifted(2, 0);
        shifted.insert(shifted.end(), packed.begin(), packed.end());
        Buffer unaligned;
        yuuka_plan * plan = nullptr;
        ok = alloc(device, unaligned, shifted.size()) && upload_vec(unaligned, shifted) &&
             check(yuuka_plan_create(device, &plan), "create Q6_K alignment plan");
        if (ok) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMM;
            op.buf[0] = unaligned.buf;
            op.buf[1] = qs.buf;
            op.buf[2] = out.buf;
            op.buf[3] = ds.buf;
            op.off[0] = 2;
            op.n[0] = width;
            op.n[1] = channels;
            op.n[2] = 129;
            op.u[0] = YUUKA_DTYPE_Q6_K;
            op.u[1] = YUUKA_GEMM_PATH_MMQ;
            ok = check(yuuka_plan_add(plan, &op), "add Q6_K alignment op") &&
                 verify(129, "weight offset 2", [&]() { return execute(plan, stream, event, 0); });
        }
        yuuka_plan_destroy(plan);
        release(unaligned);
    }
    release(out);
    release(ds);
    release(qs);
    release(wb);
    return ok;
}

// Use the binary16 spacing and the default nearest-even rounding mode as
// the independent CPU oracle for the finite operand range of the fixture.
float round_pair_fp16(float value) {
    int exponent = 0;
    std::frexp(value, &exponent);
    const int precision = std::min(24, 11 - exponent);
    return std::ldexp(std::nearbyint(std::ldexp(value, precision)), -precision);
}

bool test_prefill_q8_pair(yuuka_device * device, yuuka_stream * stream,
                          yuuka_event * event, int64_t rows, bool exact_sum,
                          std::mt19937 & rng) {
    constexpr int64_t width = 2560, rank = 32, stride = width / 32 * 34;
    const size_t valid = rows * rank;
    std::uniform_real_distribution<float> input(-0.5f, 0.5f);
    std::uniform_int_distribution<int> exact_input(-7, 7);
    std::vector<float> x(width * rows);
    for (float & value : x) {
        // After FP16 rounding this fixture is on a 1/64 grid. A value just
        // above the grid point also makes the operand conversion observable.
        value = exact_sum ? std::nextafter(std::ldexp(float(exact_input(rng)), -6), 1.0f)
                          : input(rng);
    }
    std::vector<float> rounded_x(x.size());
    std::transform(x.begin(), x.end(), rounded_x.begin(), round_pair_fp16);
    std::vector<uint8_t> weights[2];
    Buffer packed[2], output[2], activation;
    const std::vector<float> sentinel(valid + rank, kGemmSentinel);
    bool ok = alloc(device, activation, x.size() * sizeof(float)) && upload_vec(activation, x);
    for (int p = 0; p < 2; ++p) {
        weights[p] = pack_weights(YUUKA_DTYPE_Q8_0, rng, width, rank);
        for (size_t offset = 0; offset < weights[p].size(); offset += 34) {
            uint8_t * block = weights[p].data() + offset;
            const int scale_index = offset / 34 % 5;
            const float scale = exact_sum ? std::ldexp(float(scale_index), -10)
                                          : scale_index * 0.00075f;
            put_half(block, scale);
            block[2] = 0x80;
            block[33] = 0x7f;
        }
        ok = ok && alloc(device, packed[p], weights[p].size()) &&
             alloc(device, output[p], sentinel.size() * sizeof(float)) &&
             upload_vec(packed[p], weights[p]) && upload_vec(output[p], sentinel);
    }
    yuuka_plan * plan = nullptr;
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_PREFILL_Q8_0_PAIR;
        op.buf[0] = packed[0].buf;
        op.buf[1] = packed[1].buf;
        op.buf[2] = activation.buf;
        op.buf[3] = output[0].buf;
        op.buf[4] = output[1].buf;
        op.n[0] = width;
        op.n[1] = rank;
        op.n[2] = rows;
        op.u[0] = op.u[1] = YUUKA_DTYPE_Q8_0;
        ok = check(yuuka_plan_create(device, &plan), "create Q8_0 pair plan") &&
             check(yuuka_plan_add(plan, &op), "add Q8_0 pair") &&
             execute(plan, stream, event, 0);
    }
    for (int p = 0; p < 2 && ok; ++p) {
        std::vector<float> got(sentinel.size());
        ok = download_vec(output[p], got) &&
             std::equal(got.begin() + valid, got.end(), sentinel.begin() + valid);
        std::vector<float> baseline(sentinel.size());
        ok = ok && check(yuuka_gemm_wmma(stream, output[p].buf, activation.buf,
                                        packed[p].buf, width, rank, rows, YUUKA_DTYPE_Q8_0, event),
                         "original Q8_0 projection") &&
             check(yuuka_event_wait(event), "original Q8_0 projection wait") &&
             download_vec(output[p], baseline);
        if (!ok || std::memcmp(got.data(), baseline.data(), valid * sizeof(float)) != 0) {
            std::fprintf(stderr, "Q8_0 pair differs from original WMMA output %d\n", p);
            ok = false;
            break;
        }
        std::vector<float> rounded_w(width * rank);
        for (int64_t c = 0; c < rank; ++c) {
            const uint8_t * packed_row = weights[p].data() + c * stride;
            for (int64_t k = 0; k < width; k += 32) {
                const uint8_t * block = packed_row + k / 32 * 34;
                const float d = half_to_float(uint16_t(block[0] | (block[1] << 8)));
                for (int l = 0; l < 32; ++l) {
                    rounded_w[c * width + k + l] = round_pair_fp16(d * int8_t(block[2 + l]));
                }
            }
        }
        double max_diff = 0;
        for (int64_t i = 0; i < rows && ok; ++i) {
            for (int64_t c = 0; c < rank; ++c) {
                double expected = 0;
                for (int64_t k = 0; k < width; ++k) {
                    expected += double(rounded_w[c * width + k]) * rounded_x[i * width + k];
                }
                const double diff = std::fabs(got[i * rank + c] - expected);
                // The dyadic fixture bounds the integer numerator of every
                // partial sum below 2^24. The random fixture checks exact
                // legacy WMMA parity above and reports its CPU differential.
                ok = ok && std::isfinite(diff) && (!exact_sum || diff <= 1e-5);
                max_diff = std::max(max_diff, diff);
            }
        }
        std::printf("Q8_0 pair, %lld rows, output %d, %s: %s (CPU max abs %.3g)\n",
                    static_cast<long long>(rows), p, exact_sum ? "exact-sum" : "random",
                    ok ? "ok" : "FAILED", max_diff);
    }
    yuuka_plan_destroy(plan);
    for (int p = 0; p < 2; ++p) {
        release(output[p]);
        release(packed[p]);
    }
    release(activation);
    return ok;
}

bool test_silu_mul_quant_q8_1(yuuka_device * device, yuuka_stream * stream,
                             yuuka_event * event, std::mt19937 & rng) {
    constexpr uint64_t width = 256, rows = 17, channels = 65, offset = 16;
    constexpr uint64_t count = width * rows;
    constexpr uint64_t float_prefix = offset / sizeof(float);
    std::vector<float> gate(count + float_prefix), up(count + float_prefix);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (uint64_t i = 32; i < count; ++i) {
        gate[i + float_prefix] = dist(rng);
        up[i + float_prefix] = dist(rng);
    }
    // The first block is zero. The second has an exact amax of 127 and
    // positive and negative halfway values to exercise nearest-even rounding.
    for (uint64_t i = 32; i < 64; ++i) {
        gate[i + float_prefix] = 32.0f;
        up[i + float_prefix] = (i == 32 ? 127.0f : i % 2 ? -1.5f : 0.5f) / 32.0f;
    }
    const auto weights = pack_weights(YUUKA_DTYPE_Q4_K, rng, width, channels);
    Buffer gb, ub, ab, qb, sb, wb, ob;
    bool ok = alloc(device, gb, gate.size() * sizeof(float)) &&
              alloc(device, ub, up.size() * sizeof(float)) &&
              alloc(device, ab, count * sizeof(float)) &&
              alloc(device, qb, count + offset) &&
              alloc(device, sb, count / 4 + offset) &&
              alloc(device, wb, weights.size()) &&
              alloc(device, ob, rows * channels * sizeof(float)) &&
              upload_vec(gb, gate) && upload_vec(ub, up) && upload_vec(wb, weights);
    const auto run = [&](bool fused) {
        yuuka_plan * plan = nullptr;
        if (!check(yuuka_plan_create(device, &plan), "SwiGLU Q8_1 plan")) return false;
        yuuka_op quantize{};
        quantize.n[0] = width;
        quantize.n[1] = rows;
        bool success = true;
        if (fused) {
            quantize.code = YUUKA_OP_SILU_MUL_QUANT_Q8_1;
            quantize.buf[0] = gb.buf;
            quantize.buf[1] = ub.buf;
            quantize.buf[2] = qb.buf;
            quantize.buf[3] = sb.buf;
            for (int i = 0; i < 4; ++i) quantize.off[i] = offset;
        } else {
            yuuka_op silu{};
            silu.code = YUUKA_OP_SILU_MUL;
            silu.n[0] = count;
            silu.buf[0] = gb.buf;
            silu.buf[1] = ub.buf;
            silu.buf[2] = ab.buf;
            silu.off[0] = silu.off[1] = offset;
            success = check(yuuka_plan_add(plan, &silu), "add separate SwiGLU");
            quantize.code = YUUKA_OP_QUANT_Q8_1;
            quantize.buf[0] = ab.buf;
            quantize.buf[1] = qb.buf;
            quantize.buf[2] = sb.buf;
        }
        yuuka_op down{};
        down.code = YUUKA_OP_GEMM;
        down.buf[0] = wb.buf;
        down.buf[1] = qb.buf;
        down.buf[2] = ob.buf;
        down.buf[3] = sb.buf;
        down.off[1] = down.off[3] = fused ? offset : 0;
        down.n[0] = width;
        down.n[1] = channels;
        down.n[2] = rows;
        down.u[0] = YUUKA_DTYPE_Q4_K;
        down.u[1] = YUUKA_GEMM_PATH_MMQ;
        success = success && check(yuuka_plan_add(plan, &quantize), "add SwiGLU Q8_1 quantizer") &&
            check(yuuka_plan_add(plan, &down), "add down MMQ") && execute(plan, stream, event, 0);
        yuuka_plan_destroy(plan);
        return success;
    };
    std::vector<int8_t> want_q(count), got_q(count);
    std::vector<float> want_scales(count / 16), got_scales(count / 16);
    std::vector<float> activated(count), want_out(rows * channels), got_out(rows * channels);
    ok = ok && run(false) && download_vec(ab, activated) && download_vec(qb, want_q) &&
        download_vec(sb, want_scales) && download_vec(ob, want_out) &&
        check_q8_1("SwiGLU Q8_1", activated, want_q, want_scales, width, rows);
    for (int rep = 0; ok && rep < 2; ++rep) {
        ok = run(true) &&
            check(yuuka_memcpy_d2h_at(got_q.data(), qb.buf, offset, count), "SwiGLU digits") &&
            check(yuuka_memcpy_d2h_at(got_scales.data(), sb.buf, offset, count / 4), "SwiGLU scales") &&
            download_vec(ob, got_out) && got_q == want_q && got_scales == want_scales &&
            got_out == want_out;
    }
    for (Buffer * buf : {&gb, &ub, &ab, &qb, &sb, &wb, &ob}) release(*buf);
    std::printf("SwiGLU Q8_1 and down MMQ: %s\n", ok ? "ok" : "FAILED");
    return ok;
}

bool test_gemm_swiglu(yuuka_device * device, yuuka_stream * stream,
                      yuuka_event * event, yuuka_dtype dtype, const char * name,
                      int64_t n_in, int64_t n_out, int64_t rows,
                      std::mt19937 & rng, bool fp16_oracle = false) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint8_t> gate = pack_weights(dtype, rng, n_in, n_out);
    std::vector<uint8_t> up = pack_weights(dtype, rng, n_in, n_out);
    std::vector<float> got(size_t(n_out * (rows + 64)));
    Buffer gb, ub, xb, ob;
    bool ok = alloc(device, gb, gate.size()) && alloc(device, ub, up.size()) &&
              alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, got.size() * sizeof(float)) &&
              upload(gb, gate.data(), gate.size()) &&
              upload(ub, up.data(), up.size()) && upload_vec(xb, x);
    yuuka_plan * plan = nullptr;
    if (ok) ok = check(yuuka_plan_create(device, &plan), "fused plan create");
    if (ok) {
        yuuka_op op{};
        op.code = YUUKA_OP_GEMM_SWIGLU;
        op.buf[0] = gb.buf;
        op.buf[1] = ub.buf;
        op.buf[2] = xb.buf;
        op.buf[3] = ob.buf;
        op.n[0] = uint64_t(n_in);
        op.n[1] = uint64_t(n_out);
        op.n[2] = uint64_t(rows);
        op.u[0] = dtype;
        ok = check(yuuka_plan_add(plan, &op), "add fused swiglu");
    }
    ok = ok && check_gemm_partial_output(ob, size_t(n_out * rows), got, [&]() {
        return check(yuuka_plan_execute(plan, stream, 0, 0, 1, event), "execute fused") &&
               check(yuuka_event_wait(event), "fused wait");
    });
    const DotFn dot = dot_for(dtype);
    if (ok && dot != nullptr) {
        const uint64_t rb = row_bytes(dtype, n_in);
        const auto product = [&](const uint8_t * wrow, const float * xrow) {
            return fp16_oracle ? dot_rounded_kquant(dtype, wrow, xrow, n_in)
                               : dot(wrow, xrow, n_in);
        };
        const double tolerance = fp16_oracle ? 1e-3 : 2e-2;
        double max_abs = 0.0;
        for (int64_t row = 0; row < rows && ok; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                const float g = product(gate.data() + col * rb, x.data() + row * n_in);
                const float u = product(up.data() + col * rb, x.data() + row * n_in);
                const float expected = (g / (1.0f + std::exp(-g))) * u;
                const float value = got[size_t(row * n_out + col)];
                const double diff = std::fabs(double(value) - double(expected));
                max_abs = std::max(max_abs, diff);
                if (!std::isfinite(diff) || diff > tolerance * (1.0 + std::fabs(double(expected)))) {
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
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(ob);
    release(xb);
    release(ub);
    release(gb);
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
    Buffer wb, xb, ob, ob16;
    bool ok = alloc(device, wb, packed.size()) && alloc(device, xb, x16.size() * 2) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              alloc(device, ob16, size_t(n_out * rows) * 2) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x16);
    if (ok) {
        ok = check(yuuka_gemm_blas(stream, ob.buf, ob16.buf, xb.buf, wb.buf, n_in,
                                   n_out, rows, event), "gemm blas") &&
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
    std::printf("rocBLAS F16 route: %s\n", ok ? "ok" : "FAILED");

    // The plan path must produce the same result through an explicit cast.
    Buffer src32, dst16, o16;
    yuuka_plan * plan = nullptr;
    bool plan_ok = ok && alloc(device, src32, x.size() * 4) &&
                   alloc(device, dst16, x.size() * 2) &&
                   alloc(device, o16, size_t(n_out * rows) * 2) &&
                   upload_vec(src32, x) &&
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
        gemm.buf[3] = o16.buf;
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
    release(o16);
    release(dst16);
    release(src32);
    release(ob16);
    release(ob);
    release(xb);
    release(wb);
    return ok && plan_ok;
}

// test_dequant_f16 checks the dequantization kernel element by element. Every
// element of the expanded matrix must match an independent decoder of the
// packed format; the kernel rounds its result to FP16, so the comparison allows
// one FP16 unit of relative headroom for a contracted multiply-add.
bool test_dequant_f16(yuuka_device * device, yuuka_stream * stream,
                      yuuka_event * event, yuuka_dtype dtype, const char * name,
                      int64_t n_in, int64_t n_out, std::mt19937 & rng) {
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    Buffer wb, w16;
    bool ok = alloc(device, wb, packed.size()) &&
              alloc(device, w16, size_t(n_in * n_out) * 2) &&
              upload(wb, packed.data(), packed.size());
    if (ok) {
        ok = check(yuuka_dequant_f16(stream, w16.buf, wb.buf, n_in, n_out, dtype,
                                     event), "dequant f16") &&
             check(yuuka_event_wait(event), "dequant wait");
    }
    std::vector<uint16_t> got(size_t(n_in * n_out));
    ok = ok && download_vec(w16, got);
    const uint64_t rb = row_bytes(dtype, n_in);
    double max_rel = 0.0;
    for (int64_t col = 0; col < n_out && ok; ++col) {
        const uint8_t * wrow = packed.data() + col * rb;
        for (int64_t i = 0; i < n_in; ++i) {
            const float expected = dequant_ref_value(dtype, wrow, i);
            const float value = half_to_float(got[size_t(col * n_in + i)]);
            const double rel = std::fabs(double(value) - double(expected)) /
                               (1.0 + std::fabs(double(expected)));
            max_rel = std::max(max_rel, rel);
            if (std::fabs(double(value) - double(expected)) >
                9.77e-4 * (1.0 + std::fabs(double(expected)))) {
                std::fprintf(stderr, "%s element [%lld,%lld]: got %.6f expected %.6f\n",
                             name, static_cast<long long>(col),
                             static_cast<long long>(i), value, expected);
                ok = false;
                break;
            }
        }
    }
    std::printf("%s: %s (max rel %.3g)\n", name, ok ? "ok" : "FAILED", max_rel);
    release(w16);
    release(wb);
    return ok;
}

// test_gemm_blas_dequant checks the dequant-plus-rocBLAS route end to end: the
// packed K-quant matrix is expanded into FP16, multiplied by FP16 activations
// with an FP32 accumulator, and the FP16 result is cast back to FP32.
//
// The reference is built from the downloaded FP16 weights and the FP16
// activations in float64, so it implements the same operand rounding the route
// uses and the bound covers only the accumulator and the single FP16 rounding
// of the result. Comparing against full-precision weights instead would measure
// the operand rounding of a strongly cancelling synthetic row, which reaches
// 2e-2 here for any FP16-staged route and hides the accumulator behavior.
bool test_gemm_blas_dequant(yuuka_device * device, yuuka_stream * stream,
                            yuuka_event * event, yuuka_dtype dtype,
                            const char * name, int64_t n_in, int64_t n_out,
                            int64_t rows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = dist(rng);
    std::vector<uint16_t> x16(x.size());
    for (size_t i = 0; i < x.size(); ++i) x16[i] = float_to_half(x[i]);
    std::vector<uint8_t> packed = pack_weights(dtype, rng, n_in, n_out);
    Buffer wb, w16, xb, o16, ob;
    bool ok = alloc(device, wb, packed.size()) &&
              alloc(device, w16, size_t(n_in * n_out) * 2) &&
              alloc(device, xb, x16.size() * 2) &&
              alloc(device, o16, size_t(n_out * rows) * 2) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              upload(wb, packed.data(), packed.size()) && upload_vec(xb, x16);
    if (ok) {
        ok = check(yuuka_dequant_f16(stream, w16.buf, wb.buf, n_in, n_out, dtype,
                                     event), "dequant f16") &&
             check(yuuka_event_wait(event), "dequant wait");
    }
    if (ok) {
        ok = check(yuuka_gemm_blas(stream, ob.buf, o16.buf, xb.buf, w16.buf, n_in,
                                   n_out, rows, event), "gemm blas dequant") &&
             check(yuuka_event_wait(event), "blas-dequant wait");
    }
    std::vector<uint16_t> host_w16(size_t(n_in * n_out));
    std::vector<float> got(size_t(n_out * rows));
    ok = ok && download_vec(w16, host_w16) && download_vec(ob, got);
    // The full element set of these shapes is tens of billions of reference
    // multiply-adds. The first eight rows cover the row stride, the column
    // mapping, and the tile edge; the elementwise dequant check above covers
    // every expanded weight, and the WMMA test covers row indexing at these
    // shapes.
    const int64_t checked_rows = rows < 8 ? rows : 8;
    if (ok) {
        double max_abs = 0.0;
        double max_rel = 0.0;
        for (int64_t row = 0; row < checked_rows && ok; ++row) {
            for (int64_t col = 0; col < n_out; ++col) {
                float expected = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) {
                    const float w = half_to_float(host_w16[size_t(col * n_in + i)]);
                    expected += w * half_to_float(x16[size_t(row * n_in + i)]);
                }
                const double value = double(got[size_t(row * n_out + col)]);
                const double diff = std::fabs(value - expected);
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, diff / (1.0 + std::fabs(expected)));
                // One FP16 unit roundoff of the result, plus the extra
                // roundings rocBLAS performs when it splits K on tall shapes:
                // the FP16 destination forces that reduction through FP16, so
                // the bound is two units rather than one.
                if (diff > 2e-3 * (1.0 + std::fabs(expected))) {
                    std::fprintf(stderr, "%s [%lld,%lld]: got %.6f expected %.6f\n",
                                 name, static_cast<long long>(row),
                                 static_cast<long long>(col), value, expected);
                    ok = false;
                    break;
                }
            }
        }
        std::printf("%s: %s (max abs %.3g, max rel %.3g)\n", name,
                    ok ? "ok" : "FAILED", max_abs, max_rel);
    }
    release(ob);
    release(o16);
    release(xb);
    release(w16);
    release(wb);
    return ok;
}

// test_lm_head runs the streaming prefill LM head over every supported dtype
// and row-block size, including a row block that starts inside the last
// channel tile. The kernel is checked against the independent host dot
// products, and the plan path is checked to reject a row block above the cap.
bool test_lm_head(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                  std::mt19937 & rng) {
    const struct {
        yuuka_dtype dtype;
        const char * name;
    } cases[] = {
        {YUUKA_DTYPE_F32, "lm head F32"},
        {YUUKA_DTYPE_F16, "lm head F16"},
        {YUUKA_DTYPE_Q4_0, "lm head Q4_0"},
        {YUUKA_DTYPE_Q5_0, "lm head Q5_0"},
        {YUUKA_DTYPE_Q8_0, "lm head Q8_0"},
        {YUUKA_DTYPE_Q4_K, "lm head Q4_K"},
        {YUUKA_DTYPE_Q5_K, "lm head Q5_K"},
        {YUUKA_DTYPE_Q6_K, "lm head Q6_K"},
    };
    // 2560 keeps ten full K steps; 400 outputs makes the last block serve four
    // channels and leave the fifth warp idle, which exercises the channel tail.
    const struct {
        int64_t n_in;
        int64_t n_out;
        int64_t rows;
    } shapes[] = {
        {2560, 400, 1},
        {2560, 400, 3},
        {2560, 400, 16},
        {512, 96, 2},
    };
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    bool ok = true;
    for (const auto & c : cases) {
        for (const auto & s : shapes) {
            std::vector<float> x(size_t(s.n_in * s.rows));
            for (float & v : x) v = dist(rng);
            std::vector<uint8_t> packed = pack_weights(c.dtype, rng, s.n_in, s.n_out);
            Buffer wb, xb, ob;
            bool pass = alloc(device, wb, packed.size()) &&
                        alloc(device, xb, x.size() * 4) &&
                        alloc(device, ob, size_t(s.n_out * s.rows) * 4) &&
                        upload(wb, packed.data(), packed.size()) &&
                        upload_vec(xb, x);
            std::vector<float> got(size_t(s.n_out * s.rows));
            std::vector<float> again(size_t(s.n_out * s.rows));
            if (pass) {
                pass = check(yuuka_lm_head(stream, ob.buf, xb.buf, wb.buf, s.n_in,
                                          s.n_out, s.rows, c.dtype, event),
                             "lm head") &&
                       check(yuuka_event_wait(event), "lm head wait") &&
                       download_vec(ob, got);
            }
            // A second run must reproduce the first one exactly: the kernel
            // reduces each output over a fixed lane order.
            if (pass) {
                pass = check(yuuka_lm_head(stream, ob.buf, xb.buf, wb.buf, s.n_in,
                                          s.n_out, s.rows, c.dtype, event),
                             "lm head repeat") &&
                       check(yuuka_event_wait(event), "lm head repeat wait") &&
                       download_vec(ob, again);
            }
            if (pass && got != again) {
                std::fprintf(stderr, "%s [%lld]: repeated run differs\n", c.name,
                             static_cast<long long>(s.rows));
                pass = false;
            }
            if (pass) {
                pass = compare_gemm(c.name, packed, x, got, c.dtype, s.n_in,
                                    s.n_out, s.rows);
            }
            ok = ok && pass;
            release(ob);
            release(xb);
            release(wb);
        }
    }
    // The plan entry point rejects a row block above the kernel cap instead of
    // launching a partially covered output block.
    {
        yuuka_plan * plan = nullptr;
        Buffer wb, xb, ob;
        const int64_t n_in = 512, n_out = 64, rows = 17;
        const uint64_t rb = row_bytes(YUUKA_DTYPE_Q8_0, n_in);
        bool pass = alloc(device, wb, rb * uint64_t(n_out)) &&
                    alloc(device, xb, size_t(n_in * rows) * 4) &&
                    alloc(device, ob, size_t(n_out * rows) * 4) &&
                    check(yuuka_plan_create(device, &plan), "lm head plan create");
        if (pass) {
            yuuka_op op{};
            op.code = YUUKA_OP_LM_HEAD;
            op.buf[0] = wb.buf;
            op.buf[1] = xb.buf;
            op.buf[2] = ob.buf;
            op.n[0] = n_in;
            op.n[1] = n_out;
            op.n[2] = rows;
            op.u[0] = YUUKA_DTYPE_Q8_0;
            const yuuka_status status = yuuka_plan_add(plan, &op);
            if (status != YUUKA_ERR_INVALID_ARGUMENT) {
                std::fprintf(stderr, "lm head: %lld rows accepted with status %d\n",
                             static_cast<long long>(rows), int(status));
                pass = false;
            }
        }
        if (plan != nullptr) {
            yuuka_plan_destroy(plan);
        }
        release(ob);
        release(xb);
        release(wb);
        std::printf("lm head row cap: %s\n", pass ? "ok" : "FAILED");
        ok = ok && pass;
    }
    return ok;
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

// Compare tiled and WMMA attention to a CPU oracle over the same stored KV.
bool test_attention_prefill_mma(yuuka_device * device, yuuka_stream * stream,
                                yuuka_event * event, std::mt19937 & rng,
                                yuuka_dtype dtype = YUUKA_DTYPE_F32, int head_dim = 256,
                                bool dense = false, int position = 0, int rows = 64) {
    const int heads = 16, kv_heads = 4, block_size = 16;
    const int kv_dim = kv_heads * head_dim;
    const int query_width = dense ? head_dim : 2 * head_dim;
    const int blocks = (position + rows + block_size - 1) / block_size + 1;
    const int capacity = blocks * block_size;
    const bool packed = dtype == YUUKA_DTYPE_Q8_0;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> q(size_t(rows) * heads * query_width);
    std::vector<float> kc(size_t(capacity) * kv_dim), vc(size_t(capacity) * kv_dim);
    for (float & x : q) x = dist(rng);
    for (float & x : kc) x = dist(rng);
    for (float & x : vc) x = dist(rng);
    const float scale = 1.0f / std::sqrt(float(head_dim));
    // Permute the block order so a kernel that ignores the table disagrees.
    std::vector<int32_t> block_table(static_cast<size_t>(blocks), 0);
    for (int i = 0; i < blocks; ++i) block_table[size_t(i)] = blocks - 1 - i;
    std::vector<float> pk(kc.size()), pv(vc.size());
    std::vector<uint8_t> q8k, q8v;
    if (packed) {
        q8k = pack_weights(dtype, rng, kv_dim, capacity);
        q8v = pack_weights(dtype, rng, kv_dim, capacity);
    }
    const auto decode_pool_row = [kv_dim](uint8_t * source, float * decoded) {
        for (int c = 0; c < kv_dim; c += 32) {
            uint8_t * block = source + (c / 32) * 34;
            put_half(block, 0.002f * float(1 + c / 32 % 5));
            const float d = half_to_float(uint16_t(block[0] | block[1] << 8));
            for (int j = 0; j < 32; ++j) decoded[c + j] = d * int8_t(block[2 + j]);
        }
    };
    for (int r = 0; r < capacity; ++r) {
        const size_t dst = size_t(block_table[r / block_size]) * block_size + r % block_size;
        if (packed) {
            const size_t offset = dst * row_bytes(dtype, kv_dim);
            decode_pool_row(q8k.data() + offset, kc.data() + size_t(r) * kv_dim);
            decode_pool_row(q8v.data() + offset, vc.data() + size_t(r) * kv_dim);
        } else {
            std::memcpy(pk.data() + dst * kv_dim, kc.data() + size_t(r) * kv_dim, size_t(kv_dim) * 4);
            std::memcpy(pv.data() + dst * kv_dim, vc.data() + size_t(r) * kv_dim, size_t(kv_dim) * 4);
        }
    }
    const size_t pool_bytes = capacity * row_bytes(dtype, kv_dim);
    const void * key_data = packed ? static_cast<const void *>(q8k.data()) : pk.data();
    const void * value_data = packed ? static_cast<const void *>(q8v.data()) : pv.data();
    Buffer qb, kb, vb, ob, tb, ot;
    bool ok = alloc(device, qb, q.size() * 4) && alloc(device, kb, pool_bytes) &&
              alloc(device, vb, pool_bytes) &&
              alloc(device, ob, size_t(rows) * heads * head_dim * 4) &&
              alloc(device, ot, size_t(rows) * heads * head_dim * 4) &&
              alloc(device, tb, block_table.size() * 4) && upload_vec(qb, q) &&
              check(yuuka_memcpy_h2d(kb.buf, key_data, pool_bytes), "upload attention keys") &&
              check(yuuka_memcpy_h2d(vb.buf, value_data, pool_bytes), "upload attention values") &&
              upload_vec(tb, block_table);
    std::vector<float> got(size_t(rows) * heads * head_dim);
    std::vector<float> got_tiled(size_t(rows) * heads * head_dim);
    std::vector<float> again(size_t(rows) * heads * head_dim);
    for (int pass = 0; pass < 2 && ok; ++pass) {
        yuuka_plan * plan = nullptr;
        ok = check(yuuka_plan_create(device, &plan), "plan create");
        if (ok) {
            yuuka_op op{};
            op.code = YUUKA_OP_ATTENTION_PREFILL_PAGED;
            op.buf[0] = qb.buf;
            op.buf[1] = kb.buf;
            op.buf[2] = vb.buf;
            op.buf[3] = ob.buf;
            op.buf[4] = tb.buf;
            op.n[0] = kv_heads;
            op.n[1] = heads;
            op.n[2] = head_dim;
            op.n[3] = kv_dim;
            op.n[4] = rows;
            op.u[0] = block_size;
            op.u[1] = capacity;
            op.u[2] = dtype;
            op.u[3] = dense;
            op.f[0] = scale;
            ok = check(yuuka_plan_add(plan, &op), "add attention prefill mma");
            if (ok) {
                op.buf[3] = ot.buf;
                op.flags = YUUKA_OP_FLAG_PREFILL_ATTENTION_TILED;
                ok = check(yuuka_plan_add(plan, &op), "add attention prefill tiled");
            }
        }
        if (ok) ok = execute(plan, stream, event, position);
        if (ok) {
            ok = download_vec(ob, pass == 0 ? got : again) && download_vec(ot, got_tiled);
        }
        if (plan != nullptr) yuuka_plan_destroy(plan);
    }
    if (ok) {
        double max_abs = 0.0;
        double max_tiled = 0.0;
        for (int r = 0; r < rows && ok; ++r) {
            const int pos = position + r;
            for (int h = 0; h < heads && ok; ++h) {
                const int kvh = h / (heads / kv_heads);
                const float * qp = q.data() + size_t(r * heads + h) * query_width;
                std::vector<float> scores(size_t(pos + 1));
                float max_s = -INFINITY;
                for (int p = 0; p <= pos; ++p) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += qp[d] * kc[size_t(p * kv_dim + kvh * head_dim + d)];
                    }
                    scores[size_t(p)] = dot * scale;
                    max_s = std::max(max_s, scores[size_t(p)]);
                }
                float total = 0;
                for (float & x : scores) {
                    x = std::exp(x - max_s);
                    total += x;
                }
                for (int d = 0; d < head_dim; ++d) {
                    float value = 0;
                    for (int p = 0; p <= pos; ++p) {
                        value += scores[size_t(p)] / total * vc[size_t(p * kv_dim + kvh * head_dim + d)];
                    }
                    if (!dense) value /= 1.0f + std::exp(-qp[head_dim + d]);
                    const size_t idx = size_t(r * heads + h) * head_dim + d;
                    max_abs = std::max(max_abs, double(std::fabs(got[idx] - value)));
                    max_tiled = std::max(max_tiled, double(std::fabs(got[idx] - got_tiled[idx])));
                    const float tolerance = 5e-3f * (1.0f + std::fabs(value));
                    if (!std::isfinite(got[idx]) || !std::isfinite(got_tiled[idx]) ||
                        std::fabs(got[idx] - value) > tolerance ||
                        std::fabs(got[idx] - got_tiled[idx]) > tolerance) {
                        std::fprintf(stderr, "attention mma mismatch [%d,%d,%d]: %.6f vs %.6f\n",
                                     r, h, d, got[idx], value);
                        ok = false;
                        break;
                    }
                    if (got[idx] != again[idx]) {
                        std::fprintf(stderr, "attention mma not reproducible [%d,%d,%d]\n", r, h, d);
                        ok = false;
                        break;
                    }
                }
            }
        }
        std::printf("mma attention dtype=%u width=%d dense=%d rows=%d position=%d: %s "
                    "(max abs vs CPU %.3g, vs tiled %.3g)\n",
                    dtype, head_dim, dense, rows, position, ok ? "ok" : "FAILED", max_abs, max_tiled);
    }
    release(tb);
    release(ot);
    release(ob);
    release(vb);
    release(kb);
    release(qb);
    return ok;
}

// test_attention_prefill_mma_gqa exercises the 8:1 grouped-head WMMA path on
// the qwen35moe geometry: 16 query heads, 2 physical KV heads, width 256, F16
// pool. The query chunk is not a multiple of the 16-row tile and starts at a
// nonzero position. A second op reads the last query row through the crop
// displacement the final chunk uses. K and V are rounded to half once, and the
// CPU reference reads exactly those stored values.
bool test_attention_prefill_mma_gqa(yuuka_device * device, yuuka_stream * stream,
                                    yuuka_event * event, std::mt19937 & rng) {
    constexpr int heads = 16, kv_heads = 2, head_dim = 256;
    constexpr int kv_dim = kv_heads * head_dim;
    constexpr int block_size = 16, capacity = 288, position = 220, rows = 37;
    const int blocks = (capacity + block_size - 1) / block_size;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> q(size_t(rows) * heads * 2 * head_dim);
    for (float & x : q) x = dist(rng);
    std::vector<uint16_t> key(size_t(capacity) * kv_dim), value(key.size());
    for (uint16_t & x : key) x = float_to_half(dist(rng));
    for (uint16_t & x : value) x = float_to_half(dist(rng));
    const float scale = 1.0f / std::sqrt(float(head_dim));
    std::vector<int32_t> block_table(static_cast<size_t>(blocks), 0);
    for (int i = 0; i < blocks; ++i) block_table[size_t(i)] = blocks - 1 - i;
    std::vector<uint16_t> pk(key.size()), pv(value.size());
    for (int r = 0; r < capacity; ++r) {
        const int64_t dst = int64_t(block_table[size_t(r / block_size)]) * block_size +
                            (r % block_size);
        std::memcpy(&pk[size_t(dst) * kv_dim], &key[size_t(r) * kv_dim],
                    size_t(kv_dim) * sizeof(uint16_t));
        std::memcpy(&pv[size_t(dst) * kv_dim], &value[size_t(r) * kv_dim],
                    size_t(kv_dim) * sizeof(uint16_t));
    }
    Buffer qb, kb, vb, ob, tb, ot;
    bool ok = alloc(device, qb, q.size() * 4) && alloc(device, kb, pk.size() * 2) &&
              alloc(device, vb, pv.size() * 2) &&
              alloc(device, ob, size_t(rows) * heads * head_dim * 4) &&
              alloc(device, ot, size_t(heads) * head_dim * 4) &&
              alloc(device, tb, block_table.size() * 4) && upload_vec(qb, q) &&
              upload_vec(kb, pk) && upload_vec(vb, pv) && upload_vec(tb, block_table);
    std::vector<float> got(size_t(rows) * heads * head_dim);
    std::vector<float> again(got.size());
    std::vector<float> got_last(size_t(heads) * head_dim);
    std::vector<float> again_last(got_last.size());
    for (int pass = 0; pass < 2 && ok; ++pass) {
        yuuka_plan * plan = nullptr;
        ok = check(yuuka_plan_create(device, &plan), "gqa attention plan create");
        if (ok) {
            yuuka_op op{};
            op.code = YUUKA_OP_ATTENTION_PREFILL_PAGED;
            op.buf[0] = qb.buf;
            op.buf[1] = kb.buf;
            op.buf[2] = vb.buf;
            op.buf[3] = ob.buf;
            op.buf[4] = tb.buf;
            op.n[0] = kv_heads;
            op.n[1] = heads;
            op.n[2] = head_dim;
            op.n[3] = kv_dim;
            op.n[4] = rows;
            op.u[0] = block_size;
            op.u[1] = capacity;
            op.u[2] = YUUKA_DTYPE_F16;
            op.f[0] = scale;
            ok = check(yuuka_plan_add(plan, &op), "add gqa attention prefill mma");
            if (ok) {
                op.buf[3] = ot.buf;
                op.off[0] = uint64_t(rows - 1) * uint64_t(heads) * 2 *
                            uint64_t(head_dim) * sizeof(float);
                op.n[4] = 1;
                op.n[5] = rows - 1;
                ok = check(yuuka_plan_add(plan, &op), "add displaced gqa attention");
            }
        }
        if (ok) ok = execute(plan, stream, event, position);
        if (ok) {
            ok = download_vec(ob, pass == 0 ? got : again) &&
                 download_vec(ot, pass == 0 ? got_last : again_last);
        }
        if (plan != nullptr) yuuka_plan_destroy(plan);
    }
    if (ok) {
        std::vector<float> scores(size_t(position + rows));
        auto check_row = [&](int r, const float * out, size_t out_base,
                             const char * what) {
            const int pos = position + r;
            for (int h = 0; h < heads && ok; ++h) {
                const int kvh = h / (heads / kv_heads);
                const float * qp = q.data() + size_t(r * heads + h) * 2 * head_dim;
                float max_s = -INFINITY;
                for (int p = 0; p <= pos; ++p) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += qp[d] * half_to_float(
                            key[size_t(p * kv_dim + kvh * head_dim + d)]);
                    }
                    scores[size_t(p)] = dot * scale;
                    max_s = std::max(max_s, scores[size_t(p)]);
                }
                float total = 0.0f;
                for (int p = 0; p <= pos; ++p) {
                    scores[size_t(p)] = std::exp(scores[size_t(p)] - max_s);
                    total += scores[size_t(p)];
                }
                for (int d = 0; d < head_dim; ++d) {
                    float want = 0.0f;
                    for (int p = 0; p <= pos; ++p) {
                        want += scores[size_t(p)] / total *
                                half_to_float(value[size_t(p * kv_dim +
                                                            kvh * head_dim + d)]);
                    }
                    want *= 1.0f / (1.0f + std::exp(-qp[head_dim + d]));
                    const float have = out[out_base + size_t(h * head_dim + d)];
                    if (!std::isfinite(have) ||
                        std::fabs(have - want) > 5e-3f * (1.0f + std::fabs(want))) {
                        std::fprintf(stderr,
                                     "gqa mma %s mismatch row %d head %d dim %d: %.6f vs %.6f\n",
                                     what, r, h, d, have, want);
                        ok = false;
                        break;
                    }
                }
            }
        };
        for (int r = 0; r < rows && ok; ++r) {
            check_row(r, got.data(), size_t(r * heads) * head_dim, "chunk");
        }
        check_row(rows - 1, got_last.data(), 0, "displaced");
        if (got != again || got_last != again_last) {
            std::fprintf(stderr, "gqa mma attention is not reproducible\n");
            ok = false;
        }
    }
    std::printf("gqa 8:1 mma attention prefill: %s\n", ok ? "ok" : "FAILED");
    release(tb);
    release(ot);
    release(ob);
    release(vb);
    release(kb);
    release(qb);
    return ok;
}

bool test_attention_prefill_split(yuuka_device * device, yuuka_stream * stream,
                                  yuuka_event * event, std::mt19937 & rng,
                                  int position, int rows = 17, int auto_splits = -1,
                                  yuuka_dtype dtype = YUUKA_DTYPE_F16) {
    constexpr int heads = 16, kv_heads = 4, head_dim = 256;
    constexpr int kv_dim = kv_heads * head_dim, block_size = 16;
    const int capacity = position + rows;
    const int blocks = (capacity + block_size - 1) / block_size;
    const size_t outputs = rows * heads * head_dim;
    const size_t scratch_bytes = rows * heads * (4 * (head_dim + 2) + 1) * sizeof(float);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> query(2 * outputs);
    for (float & x : query) x = dist(rng);
    const int pool_rows = blocks * block_size;
    const size_t pool_bytes = pool_rows * row_bytes(dtype, kv_dim);
    std::vector<uint8_t> key(pool_bytes), value(pool_bytes);
    if (dtype == YUUKA_DTYPE_F16) {
        for (size_t i = 0; i < pool_bytes; i += 2) put_half(key.data() + i, dist(rng));
        for (size_t i = 0; i < pool_bytes; i += 2) put_half(value.data() + i, dist(rng));
    } else {
        key = pack_weights(dtype, rng, kv_dim, pool_rows);
        value = pack_weights(dtype, rng, kv_dim, pool_rows);
    }
    std::vector<int32_t> table(static_cast<size_t>(blocks));
    for (int i = 0; i < blocks; ++i) table[i] = blocks - i - 1;
    Buffer qb, kb, vb, ob, tb, sb, short_scratch;
    bool ok = alloc(device, qb, query.size() * sizeof(float)) &&
              alloc(device, kb, key.size()) &&
              alloc(device, vb, value.size()) &&
              alloc(device, ob, outputs * sizeof(float)) &&
              alloc(device, tb, table.size() * sizeof(int32_t)) &&
              alloc(device, sb, scratch_bytes) &&
              alloc(device, short_scratch, scratch_bytes - 1) &&
              upload_vec(qb, query) && upload_vec(kb, key) && upload_vec(vb, value) &&
              upload_vec(tb, table);
    const auto run = [&](uint32_t flag, std::vector<float> & out, bool bind_auto = false) {
        yuuka_plan * plan = nullptr;
        if (!check(yuuka_plan_create(device, &plan), "split attention plan")) return false;
        yuuka_op op{};
        op.code = YUUKA_OP_ATTENTION_PREFILL_PAGED;
        op.flags = flag;
        op.buf[0] = qb.buf;
        op.buf[1] = kb.buf;
        op.buf[2] = vb.buf;
        op.buf[3] = ob.buf;
        op.buf[4] = tb.buf;
        op.n[0] = kv_heads;
        op.n[1] = heads;
        op.n[2] = head_dim;
        op.n[3] = kv_dim;
        op.n[4] = rows;
        op.u[0] = block_size;
        op.u[1] = capacity;
        op.u[2] = dtype;
        op.f[0] = 1.0f / std::sqrt(float(head_dim));
        bool success = true;
        if (flag != 0) {
            success = yuuka_plan_add(plan, &op) == YUUKA_ERR_INVALID_ARGUMENT;
            op.buf[6] = short_scratch.buf;
            success = success && yuuka_plan_add(plan, &op) == YUUKA_ERR_INVALID_ARGUMENT;
            op.buf[6] = sb.buf;
            op.off[6] = 1;
            success = success && yuuka_plan_add(plan, &op) == YUUKA_ERR_INVALID_ARGUMENT;
            op.off[6] = 0;
        } else if (bind_auto) {
            op.buf[6] = sb.buf;
        }
        success = success && check(yuuka_plan_add(plan, &op), "add split attention") &&
            execute(plan, stream, event, position) && download_vec(ob, out);
        if (success && flag == YUUKA_OP_FLAG_PREFILL_SPLIT_4 && position == 0) {
            float empty[head_dim + 2] = {};
            success = check(yuuka_memcpy_d2h(empty, sb.buf, sizeof(empty)), "empty attention partial") &&
                empty[0] == -INFINITY;
            for (int i = 1; i < head_dim + 2; ++i) success = success && empty[i] == 0.0f;
        }
        yuuka_plan_destroy(plan);
        return success;
    };
    std::vector<float> control(outputs), got(outputs);
    if (auto_splits >= 0) {
        const uint32_t forced = auto_splits == 4 ? YUUKA_OP_FLAG_PREFILL_SPLIT_4 : 0;
        ok = ok && run(forced, control) && run(0, got, true) && got == control;
        std::printf("auto split attention dtype=%u rows=%d position=%d splits=%d: %s\n",
                    dtype, rows, position, auto_splits, ok ? "ok" : "FAILED");
    } else {
        std::vector<float> repeated(outputs);
        ok = ok && run(0, control);
        int splits = 1;
        for (uint32_t flag : {YUUKA_OP_FLAG_PREFILL_SPLIT_1, YUUKA_OP_FLAG_PREFILL_SPLIT_2,
                              YUUKA_OP_FLAG_PREFILL_SPLIT_4}) {
            if (!ok) break;
            ok = run(flag, got) && run(flag, repeated) && got == repeated;
            double max_abs = 0.0;
            for (size_t i = 0; i < outputs && ok; ++i) {
                const double diff = std::fabs(double(got[i]) - control[i]);
                ok = std::isfinite(diff) && diff <= 5e-3 * (1.0 + std::fabs(control[i]));
                max_abs = std::max(diff, max_abs);
            }
            std::printf("split attention dtype=%u position=%d splits=%d: %s (max abs %.3g)\n",
                        dtype, position, splits, ok ? "ok" : "FAILED", max_abs);
            splits *= 2;
        }
    }
    for (Buffer * buf : {&qb, &kb, &vb, &ob, &tb, &sb, &short_scratch}) release(*buf);
    return ok;
}

enum class RecurrentPath { PerToken, Chunkwise, Prepared };

bool test_recurrent_chunk(yuuka_device * device, yuuka_stream * stream,
                          yuuka_event * event, std::mt19937 & rng, int state,
                          RecurrentPath path) {
    const bool chunked = path == RecurrentPath::Chunkwise;
    const bool prepared = path == RecurrentPath::Prepared;
    const int rows = 128, groups = 16, heads = 32, kernel = 4;
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

    // The chunked path keeps two chunk x chunk matrices per chunk and head.
    const size_t chunks = size_t((rows + 31) / 32);
    constexpr size_t scratch_offset = 16;
    const size_t prepared_width = 2 * groups * state + 2 * heads;
    const size_t scratch_bytes = chunked ? chunks * heads * 2 * 32 * 32 * sizeof(float)
        : prepared ? scratch_offset + rows * prepared_width * sizeof(float) : 0;
    Buffer qb, zb, ab, bb, cwb, dtb, avb, nb, csb, rsb, convb, ob, scb;
    bool ok = alloc(device, qb, qkv.size() * 4) && alloc(device, zb, z.size() * 4) &&
              alloc(device, ab, alpha.size() * 4) && alloc(device, bb, beta.size() * 4) &&
              alloc(device, cwb, conv_w.size() * 4) && alloc(device, dtb, dt.size() * 4) &&
              alloc(device, avb, a.size() * 4) && alloc(device, nb, norm.size() * 4) &&
              alloc(device, csb, conv_state.size() * 4) && alloc(device, rsb, rec_state.size() * 4) &&
              alloc(device, convb, qkv.size() * 4) && alloc(device, ob, z.size() * 4) &&
              (scratch_bytes == 0 || alloc(device, scb, scratch_bytes)) &&
              upload_vec(qb, qkv) && upload_vec(zb, z) && upload_vec(ab, alpha) &&
              upload_vec(bb, beta) && upload_vec(cwb, conv_w) && upload_vec(dtb, dt) &&
              upload_vec(avb, a) && upload_vec(nb, norm) && upload_vec(csb, conv_state) &&
              upload_vec(rsb, rec_state);
    yuuka_plan * plan = nullptr;
    if (ok) ok = check(yuuka_plan_create(device, &plan), "plan create");
    const auto add_chunk = [&](int start, int count) {
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
        if (chunked) {
            op.buf[12] = scb.buf;
        }
        if (prepared) {
            op.buf[13] = scb.buf;
            op.off[13] = scratch_offset;
        }
        op.off[2] = size_t(start) * conv_dim * sizeof(float);
        op.off[3] = size_t(start) * inner * sizeof(float);
        op.off[4] = size_t(start) * heads * sizeof(float);
        op.off[5] = op.off[4];
        op.off[10] = op.off[3];
        op.off[11] = op.off[2];
        op.n[0] = count;
        op.n[1] = groups;
        op.n[2] = heads;
        op.n[3] = state;
        op.n[4] = conv_dim;
        op.n[5] = inner;
        op.u[0] = kernel;
        op.f[0] = 1e-6f;
        op.f[1] = 1.0f / std::sqrt(float(state));
        if (prepared && start == 0) {
            yuuka_op invalid = op;
            invalid.off[13] = scratch_bytes;
            if (yuuka_plan_add(plan, &invalid) != YUUKA_ERR_INVALID_ARGUMENT) return false;
            invalid = op;
            ++invalid.off[13];
            if (yuuka_plan_add(plan, &invalid) != YUUKA_ERR_INVALID_ARGUMENT) return false;
            invalid = op;
            invalid.buf[12] = scb.buf;
            if (yuuka_plan_add(plan, &invalid) != YUUKA_ERR_INVALID_ARGUMENT) return false;
            invalid = op;
            invalid.buf[13] = convb.buf;
            if (yuuka_plan_add(plan, &invalid) != YUUKA_ERR_INVALID_ARGUMENT) return false;
            invalid = op;
            invalid.n[1] += uint64_t(1) << 56;
            if (yuuka_plan_add(plan, &invalid) != YUUKA_ERR_INVALID_ARGUMENT) return false;
        }
        return check(yuuka_plan_add(plan, &op), "add recurrent chunk");
    };
    if (ok) {
        constexpr int first_chunk = 57;
        ok = prepared ? add_chunk(0, first_chunk) && add_chunk(first_chunk, rows - first_chunk)
                      : add_chunk(0, rows);
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
                const double diff = std::fabs(double(got[i]) - double(expected[i]));
                if (!std::isfinite(diff)) {
                    std::fprintf(stderr, "nonfinite recurrent output at index %zu\n", i);
                    ok = false;
                    break;
                }
                row_max = std::max(row_max, diff);
            }
            max_abs = std::max(max_abs, row_max);
            if (row_max > 2e-4 * (1.0 + std::fabs(double(expected[size_t(rr * inner)])))) {
                std::fprintf(stderr, "recurrent chunk output mismatch at row %d: %.6g\n",
                             rr, row_max);
                ok = false;
                break;
            }
        }
        // The conv scratch holds the silu'd convolution; the chunked path also
        // normalizes the q and k segments in place, which `conv` already holds.
        double conv_max = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < conv_dim; ++c) {
                const size_t i = size_t(r * conv_dim + c);
                const float want = chunked ? conv[i] : raw_conv[i];
                conv_max = std::max(conv_max, double(std::fabs(got_conv[i] - want)));
            }
        }
        if (conv_max > 1e-5) {
            std::fprintf(stderr, "recurrent conv scratch mismatch: %.3g\n", conv_max);
            ok = false;
        }
        for (size_t i = 0; i < conv_state.size() && ok; ++i) {
            if (std::fabs(got_conv_state[i] - conv_state[i]) > 1e-5f) {
                std::fprintf(stderr, "recurrent conv state mismatch\n");
                ok = false;
            }
        }
        for (size_t i = 0; i < rec_state.size(); ++i) {
            const double diff = std::fabs(double(got_rec_state[i]) - double(rec_state[i]));
            if (!std::isfinite(diff) || diff > 2e-4 * (1.0 + std::fabs(double(rec_state[i])))) {
                std::fprintf(stderr, "recurrent matrix state mismatch at %zu\n", i);
                ok = false;
                break;
            }
        }
        // Second execution with the original state must be bitwise identical.
        // The chunked path consumes alpha and beta in place, so restore them.
        bool deterministic = false;
        if (upload_vec(csb, conv_state0) && upload_vec(rsb, rec_state0) &&
            upload_vec(ab, alpha) && upload_vec(bb, beta) &&
            execute(plan, stream, event, 0)) {
            std::vector<float> again(size_t(rows * inner));
            std::vector<float> conv_again(size_t(rows * conv_dim));
            std::vector<float> state_again(rec_state.size()), history_again(conv_state.size());
            deterministic = download_vec(ob, again) && download_vec(convb, conv_again) &&
                download_vec(rsb, state_again) && download_vec(csb, history_again) &&
                again == got && conv_again == got_conv && state_again == got_rec_state &&
                history_again == got_conv_state;
        }
        if (!deterministic) std::fprintf(stderr, "recurrent chunk repeatability failed\n");
        ok = ok && deterministic;
        std::printf("recurrent chunk state=%d %s: %s (max abs %.3g)\n", state,
                    prepared ? "prepared" : chunked ? "chunked" : "per-token",
                    ok ? "ok" : "FAILED", max_abs);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(scb);
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
        {256, 64, 1},
        {256, 96, 7},
        {512, 128, 16},
        {768, 160, 33},
        {256, 64, 64},
        {512, 128, 70},
        {768, 160, 70},
        {768, 64, 70},
        {512, 128, 128},
    };
    for (const auto & c : cases) {
        for (const auto & s : shapes) {
            ok = test_gemm(device, stream, event, c.dtype, c.name, s.n_in, s.n_out,
                           s.rows, rng) && ok;
        }
    }
    const struct {
        yuuka_dtype dtype;
        const char * name;
    } fused_cases[] = {
        {YUUKA_DTYPE_F32, "fused swiglu F32"},
        {YUUKA_DTYPE_F16, "fused swiglu F16"},
        {YUUKA_DTYPE_Q4_K, "fused swiglu Q4_K"},
        {YUUKA_DTYPE_Q6_K, "fused swiglu Q6_K"},
    };
    for (const auto & c : fused_cases) {
        for (int64_t rows : {33, 70}) {
            ok = test_gemm_swiglu(device, stream, event, c.dtype, c.name, 256, 96,
                                  rows, rng) && ok;
        }
    }
    for (yuuka_dtype dtype : {YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q5_K, YUUKA_DTYPE_Q6_K}) {
        for (int64_t rows : {17, 18, 32, 33, 64, 65, 96, 127}) {
            for (int64_t width : {256, 512}) {
                const int64_t channels = width / 2;
                ok = test_gemm(device, stream, event, dtype, "short K-quant WMMA",
                               width, channels, rows, rng, true) && ok;
                ok = test_gemm_swiglu(device, stream, event, dtype, "short K-quant SwiGLU",
                                      width, channels, rows, rng, true) && ok;
            }
        }
    }
    const struct {
        yuuka_dtype dtype;
        const char * name;
    } mmq_cases[] = {
        {YUUKA_DTYPE_Q4_K, "mmq Q4_K"},
        {YUUKA_DTYPE_Q5_K, "mmq Q5_K"},
        {YUUKA_DTYPE_Q6_K, "mmq Q6_K"},
    };
    const struct {
        int64_t n_in;
        int64_t n_out;
        int64_t rows;
    } mmq_shapes[] = {
        {512, 64, 128},
        {512, 128, 130},
        {768, 70, 129},
        {2560, 128, 128},
    };
    for (const auto & c : mmq_cases) {
        for (const auto & s : mmq_shapes) {
            ok = test_gemm_mmq(device, stream, event, c.dtype, c.name, s.n_in,
                               s.n_out, s.rows, rng) && ok;
        }
    }
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q4_K, "mmq Q4_K FFN",
                       2560, 9216, 512, rng) && ok;
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q4_K, "mmq Q4_K direct K",
                       2560, 1024, 128, rng) && ok;
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q5_K, "mmq Q5_K direct SSM out",
                       4096, 2560, 128, rng) && ok;
    // The qwen35moe dense Q5_K cells selected at the release chunk. The first
    // call is the CPU/Q8_1 oracle for one selected cell; the rest compare the
    // complete 512-row matrix against the generic tile on 513 rows.
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q5_K,
                       "mmq Q5_K MoE QKV", 2048, 8192, 512, rng) && ok;
    const struct {
        const char * name;
        int64_t n_in;
        int64_t n_out;
    } moe_dense_q5[] = {
        {"mmq Q5_K MoE QKV direct/generic", 2048, 8192},
        {"mmq Q5_K MoE gate direct/generic", 2048, 4096},
        {"mmq Q5_K MoE KV direct/generic", 2048, 512},
        {"mmq Q5_K MoE out direct/generic", 4096, 2048},
    };
    for (const auto & c : moe_dense_q5) {
        ok = test_gemm_mmq_direct_control(device, stream, event, YUUKA_DTYPE_Q5_K,
                                          c.name, c.n_in, c.n_out, rng) && ok;
    }
    std::mt19937 tail_rng(137);
    for (int64_t rows : {129, 192, 511}) {
        ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q4_K, "mmq Q4_K partial K",
                           2560, 1024, rows, tail_rng) && ok;
    }
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q4_K, "mmq Q4_K partial gate",
                       2560, 4096, 129, tail_rng) && ok;
    ok = test_gemm_mmq(device, stream, event, YUUKA_DTYPE_Q5_K, "mmq Q5_K partial SSM out",
                       4096, 2560, 140, tail_rng) && ok;
    ok = test_gemm_mmq_row_tail(device, stream, event, YUUKA_DTYPE_Q4_K, 1024, 256, tail_rng) && ok;
    ok = test_gemm_mmq_row_tail(device, stream, event, YUUKA_DTYPE_Q4_K, 9216, 512, tail_rng) && ok;
    ok = test_gemm_mmq_row_tail(device, stream, event, YUUKA_DTYPE_Q5_K, 8192, 512, tail_rng) && ok;
    std::mt19937 q6_rng(732);
    ok = test_direct_q6_mmq(device, stream, event, 9216, 2560, q6_rng) && ok;
    ok = test_direct_q6_mmq(device, stream, event, 2560, 1024, q6_rng) && ok;
    ok = test_blas(device, stream, event, rng) && ok;
    const struct {
        yuuka_dtype dtype;
        const char * name;
    } blasq_cases[] = {
        {YUUKA_DTYPE_Q4_K, "blas-dequant Q4_K"},
        {YUUKA_DTYPE_Q5_K, "blas-dequant Q5_K"},
        {YUUKA_DTYPE_Q6_K, "blas-dequant Q6_K"},
    };
    const struct {
        int64_t n_in;
        int64_t n_out;
        int64_t rows;
    } blasq_shapes[] = {
        {2560, 9216, 256},
        {2560, 9216, 512},
        {9216, 2560, 64},
        {9216, 2560, 512},
        {2560, 8192, 256},
        {768, 160, 70},
    };
    for (const auto & c : blasq_cases) {
        ok = test_dequant_f16(device, stream, event, c.dtype, c.name, 2560, 256,
                              rng) && ok;
        for (const auto & s : blasq_shapes) {
            ok = test_gemm_blas_dequant(device, stream, event, c.dtype, c.name,
                                        s.n_in, s.n_out, s.rows, rng) && ok;
        }
    }
    ok = test_lm_head(device, stream, event, rng) && ok;
    ok = test_silu_mul_quant_q8_1(device, stream, event, rng) && ok;
    yuuka_device_info info{};
    ok = check(yuuka_device_info_get(device, &info), "device info for Q8_0 pair") && ok;
    if (std::strncmp(info.gcn_arch, "gfx1151", 7) == 0) {
        auto pair_rng = rng;
        for (int64_t rows : {128, 131, 512}) {
            ok = test_prefill_q8_pair(device, stream, event, rows, false, pair_rng) && ok;
            ok = test_prefill_q8_pair(device, stream, event, rows, true, pair_rng) && ok;
        }
    }
    ok = test_norm_and_rope(device, stream, event, rng) && ok;
    ok = test_embedding_and_kv(device, stream, event, rng) && ok;
    ok = test_attention_prefill(device, stream, event, rng) && ok;
    ok = test_attention_prefill_mma(device, stream, event, rng) && ok;
    ok = test_attention_prefill_mma_gqa(device, stream, event, rng) && ok;
    for (int width : {64, 128, 256}) {
        ok = test_attention_prefill_mma(device, stream, event, rng,
                                       YUUKA_DTYPE_Q8_0, width, false, 512) && ok;
        ok = test_attention_prefill_mma(device, stream, event, rng,
                                       YUUKA_DTYPE_Q8_0, width, true, 527, 19) && ok;
    }
    ok = test_attention_prefill_split(device, stream, event, rng, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 270) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 6143, 512, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 6144, 512, 4) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 14336, 512, 4) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 14337, 512, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 7679, 64, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 7680, 64, 4) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 14336, 511, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 7679, 513, 0) && ok;
    ok = test_attention_prefill_split(device, stream, event, rng, 7680, 513, 4) && ok;
    for (yuuka_dtype dtype : {YUUKA_DTYPE_F32, YUUKA_DTYPE_Q8_0}) {
        ok = test_attention_prefill_split(device, stream, event, rng, 0, 17, -1, dtype) && ok;
        ok = test_attention_prefill_split(device, stream, event, rng, 527, 17, -1, dtype) && ok;
        const int start = dtype == YUUKA_DTYPE_Q8_0 ? 8192 : 6144;
        ok = test_attention_prefill_split(device, stream, event, rng, start - 1, 512, 0, dtype) && ok;
        ok = test_attention_prefill_split(device, stream, event, rng, start, 512, 4, dtype) && ok;
    }
    ok = test_recurrent_chunk(device, stream, event, rng, 128, RecurrentPath::PerToken) && ok;
    ok = test_recurrent_chunk(device, stream, event, rng, 128, RecurrentPath::Chunkwise) && ok;
    ok = test_recurrent_chunk(device, stream, event, rng, 64, RecurrentPath::PerToken) && ok;
    ok = test_recurrent_chunk(device, stream, event, rng, 128, RecurrentPath::Prepared) && ok;

    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
