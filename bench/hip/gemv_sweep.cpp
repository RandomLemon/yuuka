#include "yuuka_hip.h"

#include <chrono>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

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

uint64_t row_bytes(uint32_t dtype, int64_t n_in) {
    switch (dtype) {
        case YUUKA_DTYPE_F16: return uint64_t(n_in) * 2;
        case YUUKA_DTYPE_Q4_0: return uint64_t(n_in / 32) * 18;
        case YUUKA_DTYPE_Q5_0: return uint64_t(n_in / 32) * 22;
        case YUUKA_DTYPE_Q8_0: return uint64_t(n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return uint64_t(n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return uint64_t(n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return uint64_t(n_in / 256) * 210;
        default: return 0;
    }
}

void fill_weights(std::vector<uint8_t> & weights, uint32_t dtype, int64_t n_in,
                  int64_t n_out, std::mt19937 & rng) {
    const uint64_t bytes = row_bytes(dtype, n_in);
    for (int64_t row = 0; row < n_out; ++row) {
        uint8_t * base = weights.data() + row * bytes;
        for (int64_t i = 0; i < n_in; ++i) {
            if (dtype == YUUKA_DTYPE_F16) {
                put_half(base + i * 2, float(int(rng() % 200) - 100) / 1000.0f);
            }
        }
        if (dtype == YUUKA_DTYPE_Q4_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                uint8_t * blk = base + b * 18;
                put_half(blk, 0.01f);
                for (int i = 0; i < 16; ++i) blk[2 + i] = uint8_t(rng() % 256);
            }
        } else if (dtype == YUUKA_DTYPE_Q5_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                uint8_t * blk = base + b * 22;
                put_half(blk, 0.01f);
                for (int i = 0; i < 20; ++i) blk[2 + i] = uint8_t(rng() % 256);
            }
        } else if (dtype == YUUKA_DTYPE_Q8_0) {
            for (int64_t b = 0; b < n_in / 32; ++b) {
                uint8_t * blk = base + b * 34;
                put_half(blk, 0.01f);
                for (int i = 0; i < 32; ++i) blk[2 + i] = uint8_t(rng() % 256);
            }
        } else if (dtype == YUUKA_DTYPE_Q4_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * blk = base + b * 144;
                put_half(blk, 0.01f);
                put_half(blk + 2, 0.005f);
                for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() % 256);
                for (int i = 0; i < 128; ++i) blk[16 + i] = uint8_t(rng() % 256);
            }
        } else if (dtype == YUUKA_DTYPE_Q5_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * blk = base + b * 176;
                put_half(blk, 0.01f);
                put_half(blk + 2, 0.005f);
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
}

yuuka_op make_op(uint32_t code, yuuka_buffer * w, yuuka_buffer * x,
                 yuuka_buffer * out, uint64_t n_in, uint64_t n_out,
                 uint64_t batch, uint64_t offset, uint32_t dtype) {
    yuuka_op op{};
    op.code = code;
    op.buf[0] = w;
    op.buf[1] = x;
    op.buf[2] = out;
    op.off[1] = offset * n_in * sizeof(float);
    op.off[2] = offset * n_out * sizeof(float);
    op.n[0] = n_in;
    op.n[1] = n_out;
    op.n[2] = batch;
    op.u[0] = dtype;
    return op;
}

double time_plan(yuuka_plan * plan, yuuka_stream * stream, yuuka_event * completion, int reps) {
    for (int i = 0; i < 2; ++i) {
        if (!check(yuuka_plan_execute(plan, stream, 0, 0, 1, completion), "warmup") ||
            !check(yuuka_event_wait(completion), "warmup wait")) {
            return 0;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (!check(yuuka_plan_execute(plan, stream, 0, 0, 1, completion), "execute") ||
            !check(yuuka_event_wait(completion), "wait")) {
            return 0;
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    return std::chrono::duration<double, std::milli>(elapsed).count() / reps;
}

void destroy_plan(yuuka_plan *& plan) {
    yuuka_plan_destroy(plan);
    plan = nullptr;
}

} // namespace

int main() {
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * completion = nullptr;
    if (!check(yuuka_stream_create(device, &stream), "stream") ||
        !check(yuuka_event_create(device, &completion), "event")) return 1;

    const int64_t n_in = 2560;
    const int64_t n_out = 2560;
    const uint32_t dtypes[] = {YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q5_K,
                               YUUKA_DTYPE_Q6_K, YUUKA_DTYPE_Q8_0,
                               YUUKA_DTYPE_Q4_0, YUUKA_DTYPE_Q5_0,
                               YUUKA_DTYPE_F16};
    const int batches[] = {1, 2, 4, 8, 16};
    const int reps = 10;
    std::mt19937 rng(1234);

    std::printf("dtype,n_in,n_out,batch,per_row_ms,batched_ms,speedup,weight_mib\n");
    for (uint32_t dtype : dtypes) {
        const uint64_t bytes = row_bytes(dtype, n_in);
        std::vector<uint8_t> weights(size_t(bytes) * n_out);
        fill_weights(weights, dtype, n_in, n_out, rng);
        yuuka_buffer * wbuf = nullptr;
        if (!check(yuuka_buffer_alloc(device, weights.size(), &wbuf), "alloc weights") ||
            !check(yuuka_memcpy_h2d(wbuf, weights.data(), weights.size()), "h2d weights")) {
            return 1;
        }
        for (int batch : batches) {
            const size_t xbytes = size_t(n_in) * batch * sizeof(float);
            const size_t obytes = size_t(n_out) * batch * sizeof(float);
            std::vector<float> x(size_t(n_in) * batch);
            for (float & value : x) value = float(int(rng() % 200) - 100) / 100.0f;
            yuuka_buffer * xbuf = nullptr;
            yuuka_buffer * obuf = nullptr;
            if (!check(yuuka_buffer_alloc(device, xbytes, &xbuf), "alloc x") ||
                !check(yuuka_buffer_alloc(device, obytes, &obuf), "alloc out") ||
                !check(yuuka_memcpy_h2d(xbuf, x.data(), xbytes), "h2d x")) {
                return 1;
            }

            yuuka_plan * per_row = nullptr;
            yuuka_plan * batched = nullptr;
            check(yuuka_plan_create(device, &per_row), "plan per-row");
            check(yuuka_plan_create(device, &batched), "plan batched");
            for (int row = 0; row < batch; ++row) {
                yuuka_op op = make_op(YUUKA_OP_GEMV, wbuf, xbuf, obuf, n_in, n_out, 1, row, dtype);
                check(yuuka_plan_add(per_row, &op), "add per-row");
            }
            yuuka_op op = make_op(YUUKA_OP_GEMV_BATCH, wbuf, xbuf, obuf, n_in, n_out, batch, 0, dtype);
            check(yuuka_plan_add(batched, &op), "add batched");

            const double per_row_ms = time_plan(per_row, stream, completion, reps);
            const double batched_ms = time_plan(batched, stream, completion, reps);
            std::printf("%s,%lld,%lld,%d,%.3f,%.3f,%.2f,%.1f\n",
                        dtype == YUUKA_DTYPE_Q4_K ? "Q4_K" :
                        dtype == YUUKA_DTYPE_Q5_K ? "Q5_K" :
                        dtype == YUUKA_DTYPE_Q6_K ? "Q6_K" :
                        dtype == YUUKA_DTYPE_Q8_0 ? "Q8_0" :
                        dtype == YUUKA_DTYPE_Q4_0 ? "Q4_0" :
                        dtype == YUUKA_DTYPE_Q5_0 ? "Q5_0" : "F16",
                        static_cast<long long>(n_in), static_cast<long long>(n_out),
                        batch, per_row_ms, batched_ms,
                        batched_ms > 0 ? per_row_ms / batched_ms : 0.0,
                        double(bytes * n_out) / (1024.0 * 1024.0));
            destroy_plan(per_row);
            destroy_plan(batched);
            yuuka_buffer_free(obuf);
            yuuka_buffer_free(xbuf);
        }
        yuuka_buffer_free(wbuf);
    }
    yuuka_event_destroy(completion);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return 0;
}
