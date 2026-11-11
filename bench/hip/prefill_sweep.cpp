// prefill_sweep compares prefill matrix implementations on gfx1151:
// per-row GEMV, batched GEMV, custom rocWMMA GEMM, and the rocBLAS FP16
// baseline. It is an offline dispatch-table input, not a hot-path tool.

#include "yuuka_hip.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Buffer {
    yuuka_buffer * buf = nullptr;
    uint64_t bytes = 0;
};

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

bool alloc(yuuka_device * device, Buffer & b, uint64_t bytes) {
    b.bytes = bytes;
    return check(yuuka_buffer_alloc(device, bytes, &b.buf), "alloc");
}

void release(Buffer & b) {
    if (b.buf != nullptr) {
        yuuka_buffer_free(b.buf);
        b.buf = nullptr;
    }
}

uint64_t row_bytes(yuuka_dtype dtype, uint64_t n) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return n * 4;
        case YUUKA_DTYPE_F16: return n * 2;
        case YUUKA_DTYPE_Q8_0: return (n / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return (n / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return (n / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return (n / 256) * 210;
        default: return 0;
    }
}

const char * dtype_name(yuuka_dtype dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return "F32";
        case YUUKA_DTYPE_F16: return "F16";
        case YUUKA_DTYPE_Q8_0: return "Q8_0";
        case YUUKA_DTYPE_Q4_K: return "Q4_K";
        case YUUKA_DTYPE_Q5_K: return "Q5_K";
        case YUUKA_DTYPE_Q6_K: return "Q6_K";
        default: return "?";
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct Result {
    const char * name;
    double ms;
    bool ok;
};

Result bench_rows(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                  yuuka_plan * plan, const char * name, int iterations,
                  std::function<void()> submit, std::function<bool()> wait) {
    for (int i = 0; i < 2; ++i) {
        submit();
        if (!wait()) return {name, 0, false};
    }
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit();
        if (!wait()) return {name, 0, false};
    }
    const auto stop = Clock::now();
    (void)device;
    (void)stream;
    (void)event;
    (void)plan;
    return {name, elapsed_ms(start, stop) / iterations, true};
}

void run_shape(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
               yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    const uint64_t rb = row_bytes(dtype, uint64_t(n_in));
    std::mt19937 rng(12345);
    std::vector<uint8_t> weights(rb * uint64_t(n_out));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<uint8_t>(rng() & 0xff);
    }
    // Keep the scale halfwords in a sane range for all K-quants.
    if (dtype == YUUKA_DTYPE_Q4_K || dtype == YUUKA_DTYPE_Q5_K) {
        for (int64_t row = 0; row < n_out; ++row) {
            weights[row * rb + 0] = 0x11;
            weights[row * rb + 1] = 0x1c;
        }
    }
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = 0.25f;

    Buffer wb, xb, ob, x16, rowbuf;
    bool ok = alloc(device, wb, weights.size()) &&
              alloc(device, xb, x.size() * 4) && alloc(device, ob, size_t(n_out * rows) * 4) &&
              alloc(device, x16, x.size() * 2) && alloc(device, rowbuf, size_t(n_in) * 4);
    if (ok) {
        ok = check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()), "h2d w") &&
             check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4), "h2d x");
    }

    yuuka_plan * plan = nullptr;
    if (ok) {
        ok = check(yuuka_plan_create(device, &plan), "plan create");
    }
    if (ok) {
        for (int64_t row = 0; row < rows; ++row) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMV;
            op.buf[0] = wb.buf;
            op.buf[1] = xb.buf;
            op.buf[2] = ob.buf;
            op.off[1] = uint64_t(row * n_in) * 4;
            op.off[2] = uint64_t(row * n_out) * 4;
            op.n[0] = uint64_t(n_in);
            op.n[1] = uint64_t(n_out);
            op.u[0] = dtype;
            if (!check(yuuka_plan_add(plan, &op), "add per-row gemv")) {
                ok = false;
                break;
            }
        }
    }

    std::vector<Result> results;
    auto wait_event = [&]() { return check(yuuka_event_wait(event), "wait"); };
    if (ok) {
        results.push_back(bench_rows(device, stream, event, plan, "row-gemv", 5, [&]() {
            (void)yuuka_plan_execute(plan, stream, 0, 0, 1, event);
        }, wait_event));
    }
    if (ok && rows <= 16) {
        results.push_back(bench_rows(device, stream, event, plan, "batch-gemv", 5, [&]() {
            (void)yuuka_gemv_batch(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                   dtype, event);
        }, wait_event));
    }
    if (ok) {
        results.push_back(bench_rows(device, stream, event, plan, "wmma", 5, [&]() {
            (void)yuuka_gemm_wmma(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                  dtype, event);
        }, wait_event));
    }
    if (ok && dtype == YUUKA_DTYPE_F16) {
        results.push_back(bench_rows(device, stream, event, plan, "rocblas", 5, [&]() {
            (void)yuuka_cast_f32_to_f16(stream, x16.buf, xb.buf, n_in * rows, event);
            (void)yuuka_gemm_blas(stream, ob.buf, x16.buf, wb.buf, n_in, n_out, rows,
                                  event);
        }, wait_event));
    }
    if (ok) {
        const double weight_gb = double(weights.size()) / 1e9;
        const double flops = 2.0 * double(n_in) * double(n_out) * double(rows);
        std::printf("%s M=%4lld N=%5lld K=%5lld", dtype_name(dtype),
                    static_cast<long long>(rows), static_cast<long long>(n_out),
                    static_cast<long long>(n_in));
        for (const Result & r : results) {
            if (!r.ok) {
                std::printf("  %s=FAIL", r.name);
                continue;
            }
            std::printf("  %s=%.3fms %.1fGB/s %.1fTF/s", r.name, r.ms,
                        weight_gb / (r.ms / 1000.0),
                        flops / (r.ms / 1000.0) / 1e12);
        }
        std::printf("\n");
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(rowbuf);
    release(x16);
    release(ob);
    release(xb);
    release(wb);
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
    const yuuka_dtype dtypes[] = {YUUKA_DTYPE_F16, YUUKA_DTYPE_Q8_0, YUUKA_DTYPE_Q4_K,
                                  YUUKA_DTYPE_Q5_K, YUUKA_DTYPE_Q6_K};
    const int64_t rows[] = {1, 4, 8, 16, 32, 64, 128};
    const struct {
        int64_t n_in;
        int64_t n_out;
    } shapes[] = {
        {2560, 9216}, {9216, 2560}, {2560, 8192},
    };
    for (yuuka_dtype dtype : dtypes) {
        for (const auto & shape : shapes) {
            for (int64_t m : rows) {
                run_shape(device, stream, event, dtype, shape.n_in, shape.n_out, m);
            }
        }
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return 0;
}
