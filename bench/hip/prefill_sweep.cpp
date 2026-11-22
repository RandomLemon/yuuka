// prefill_sweep compares prefill matrix implementations on gfx1151:
// per-row GEMV, batched GEMV, custom rocWMMA GEMM, the rocBLAS route, and the
// dequant-plus-rocBLAS route for K-quant weights. It is an offline
// dispatch-table input, not a hot-path tool.

#include "yuuka_hip.h"

#include <algorithm>
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

    const bool kquant = dtype == YUUKA_DTYPE_Q4_K || dtype == YUUKA_DTYPE_Q5_K ||
                        dtype == YUUKA_DTYPE_Q6_K;
    Buffer wb, xb, ob, x16, o16, w16, rowbuf, qsb, dsb;
    bool ok = alloc(device, wb, weights.size()) &&
              alloc(device, xb, x.size() * 4) && alloc(device, ob, size_t(n_out * rows) * 4) &&
              alloc(device, x16, x.size() * 2) &&
              alloc(device, o16, size_t(n_out * rows) * 2) &&
              alloc(device, rowbuf, size_t(n_in) * 4);
    if (ok && kquant) {
        ok = alloc(device, w16, size_t(n_in * n_out) * 2) &&
             alloc(device, qsb, uint64_t(n_in) * uint64_t(rows)) &&
             alloc(device, dsb, uint64_t(rows) * uint64_t(n_in / 32) * 8);
    }
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
            (void)yuuka_gemm_blas(stream, ob.buf, o16.buf, x16.buf, wb.buf, n_in, n_out,
                                  rows, event);
        }, wait_event));
    }
    // The K-quant route expands the packed matrix into FP16 on every call, so
    // the dequantization cost is part of the measured time.
    if (ok && kquant) {
        results.push_back(bench_rows(device, stream, event, plan, "blasq", 5, [&]() {
            (void)yuuka_dequant_f16(stream, w16.buf, wb.buf, n_in, n_out, dtype, event);
            (void)yuuka_cast_f32_to_f16(stream, x16.buf, xb.buf, n_in * rows, event);
            (void)yuuka_gemm_blas(stream, ob.buf, o16.buf, x16.buf, w16.buf, n_in, n_out,
                                  rows, event);
        }, wait_event));
    }
    // The MMQ route quantizes the activations on every call, so the Q8_1 pass is
    // part of the measured time.
    if (ok && kquant) {
        results.push_back(bench_rows(device, stream, event, plan, "mmq", 5, [&]() {
            (void)yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf, n_in, rows,
                                      event);
            (void)yuuka_gemm_mmq(stream, ob.buf, qsb.buf, dsb.buf, wb.buf, n_in, n_out,
                                 rows, dtype, event);
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
    release(dsb);
    release(qsb);
    release(w16);
    release(o16);
    release(x16);
    release(ob);
    release(xb);
    release(wb);
}

// run_lm_head compares the streaming LM head against one GEMV submission per
// row for the shapes the prefill tail actually asks for. The output matrix is
// the widest in the model and is read once, so the crossover between the two
// routes is a bandwidth question and is measured per dtype here rather than
// assumed.
void run_lm_head(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                 yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    const uint64_t rb = row_bytes(dtype, uint64_t(n_in));
    std::vector<uint8_t> weights(rb * uint64_t(n_out), 0x31);
    std::vector<float> x(size_t(n_in * rows), 0.25f);
    Buffer wb, xb, ob;
    bool ok = alloc(device, wb, weights.size()) &&
              alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()), "h2d w") &&
              check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4), "h2d x");
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
    auto wait_event = [&]() { return check(yuuka_event_wait(event), "wait"); };
    const Result row_gemv = ok
        ? bench_rows(device, stream, event, plan, "row-gemv", 5, [&]() {
              (void)yuuka_plan_execute(plan, stream, 0, 0, 1, event);
          }, wait_event)
        : Result{"row-gemv", 0, false};
    const Result lm_head = ok
        ? bench_rows(device, stream, event, plan, "lmhead", 5, [&]() {
              (void)yuuka_lm_head(stream, ob.buf, xb.buf, wb.buf, n_in, n_out,
                                  rows, dtype, event);
          }, wait_event)
        : Result{"lmhead", 0, false};
    if (ok) {
        const double weight_gb = double(weights.size()) / 1e9;
        const double row_rate = row_gemv.ok ? weight_gb / (row_gemv.ms / 1000.0) : 0.0;
        const double head_rate = lm_head.ok ? weight_gb / (lm_head.ms / 1000.0) : 0.0;
        std::printf("LMHEAD %s K=%lld N=%lld M=%lld  row-gemv=%.3fms %.1fGB/s"
                    "  lmhead=%.3fms %.1fGB/s\n",
                    dtype_name(dtype), static_cast<long long>(n_in),
                    static_cast<long long>(n_out), static_cast<long long>(rows),
                    row_gemv.ms, row_rate, lm_head.ms, head_rate);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(ob);
    release(xb);
    release(wb);
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

// pack_dense_weights builds valid packed K-quant rows with the same scale and
// minimum layout tests/hip/prefill_test.cpp uses, so the sweep measures the
// production arithmetic instead of a deltas-only shortcut.
std::vector<uint8_t> pack_dense_weights(yuuka_dtype dtype, int64_t n_in,
                                        int64_t n_out, std::mt19937 & rng) {
    std::vector<uint8_t> packed(row_bytes(dtype, uint64_t(n_in)) * uint64_t(n_out));
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    const uint64_t bytes = row_bytes(dtype, uint64_t(n_in));
    for (int64_t row = 0; row < n_out; ++row) {
        uint8_t * dst = packed.data() + uint64_t(row) * bytes;
        if (dtype == YUUKA_DTYPE_F32) {
            for (int64_t i = 0; i < n_in; ++i) {
                const float value = dist(rng);
                std::memcpy(dst + i * 4, &value, 4);
            }
        } else if (dtype == YUUKA_DTYPE_Q5_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * block = dst + b * 176;
                put_half(block, 0.002f);
                put_half(block + 2, 0.001f);
                for (int i = 0; i < 12; ++i) block[4 + i] = uint8_t(rng() & 0xff);
                for (int i = 0; i < 160; ++i) block[16 + i] = uint8_t(rng() & 0xff);
            }
        } else if (dtype == YUUKA_DTYPE_Q6_K) {
            for (int64_t b = 0; b < n_in / 256; ++b) {
                uint8_t * block = dst + b * 210;
                for (int i = 0; i < 192; ++i) block[i] = uint8_t(rng() & 0xff);
                for (int i = 0; i < 16; ++i) {
                    block[192 + i] =
                        uint8_t(int8_t(std::uniform_int_distribution<int>(-10, 10)(rng)));
                }
                put_half(block + 208, 0.002f);
            }
        }
    }
    return packed;
}

int run_moe_dense_cells(yuuka_device * device, yuuka_stream * stream,
                        yuuka_event * event) {
    constexpr int64_t rows = 512;
    const struct {
        yuuka_dtype dtype;
        int64_t n_in;
        int64_t n_out;
        const char * name;
    } cells[] = {
        {YUUKA_DTYPE_F32, 2048, 256, "router"},
        {YUUKA_DTYPE_Q5_K, 2048, 8192, "attn q/qkv"},
        {YUUKA_DTYPE_Q5_K, 2048, 4096, "recurrent gate"},
        {YUUKA_DTYPE_Q5_K, 2048, 512, "kv and shared gate/up"},
        {YUUKA_DTYPE_Q5_K, 4096, 2048, "attention out"},
        {YUUKA_DTYPE_Q6_K, 4096, 2048, "ssm out"},
        {YUUKA_DTYPE_Q6_K, 512, 2048, "shared down"},
    };
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    bool ok = true;
    for (const auto & cell : cells) {
        if (!ok) break;
        const bool kquant = cell.dtype != YUUKA_DTYPE_F32;
        std::vector<uint8_t> weights =
            pack_dense_weights(cell.dtype, cell.n_in, cell.n_out, rng);
        std::vector<float> x(size_t(cell.n_in * rows));
        for (float & value : x) value = dist(rng);
        Buffer wb, xb, ob, qsb, dsb;
        ok = alloc(device, wb, weights.size()) &&
             alloc(device, xb, x.size() * 4) &&
             alloc(device, ob, size_t(cell.n_out * rows) * 4);
        if (ok && kquant) {
            ok = alloc(device, qsb, uint64_t(cell.n_in) * uint64_t(rows)) &&
                 alloc(device, dsb, uint64_t(rows) * uint64_t(cell.n_in / 32) * 8);
        }
        if (ok) {
            ok = check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()),
                       "h2d dense sweep weights") &&
                 check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4),
                       "h2d dense sweep x");
        }
        auto submit = [&]() {
            if (cell.dtype == YUUKA_DTYPE_F32) {
                return check(yuuka_gemm(stream, ob.buf, xb.buf, wb.buf, cell.n_in,
                                        cell.n_out, rows, YUUKA_DTYPE_F32, event),
                             "dense sweep gemm f32");
            }
            return check(yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf,
                                             cell.n_in, rows, event),
                         "dense sweep quantize") &&
                   check(yuuka_gemm_mmq(stream, ob.buf, qsb.buf, dsb.buf, wb.buf,
                                        cell.n_in, cell.n_out, rows, cell.dtype,
                                        event),
                         "dense sweep gemm mmq");
        };
        auto wait = [&]() { return check(yuuka_event_wait(event), "dense sweep wait"); };
        std::vector<double> samples;
        samples.reserve(5);
        for (int i = 0; ok && i < 7; ++i) {
            const auto start = Clock::now();
            ok = submit();
            if (ok) ok = wait();
            const double ms = elapsed_ms(start, Clock::now());
            if (ok && i >= 2) samples.push_back(ms);
        }
        if (ok && samples.size() == 5) {
            std::sort(samples.begin(), samples.end());
            const double weight_gb = double(weights.size()) / 1e9;
            const double flops = 2.0 * double(cell.n_in) * double(cell.n_out) *
                                 double(rows);
            const double median = samples[2];
            std::printf("DENSE %-6s M=%lld K=%lld N=%lld %-22s median=%.3fms"
                        " %.1fGB/s %.1fTF/s samples=",
                        dtype_name(cell.dtype), static_cast<long long>(rows),
                        static_cast<long long>(cell.n_in),
                        static_cast<long long>(cell.n_out), cell.name, median,
                        weight_gb / (median / 1000.0),
                        flops / (median / 1000.0) / 1e12);
            for (size_t i = 0; i < samples.size(); ++i) {
                std::printf("%s%.3f", i == 0 ? "" : ",", samples[i]);
            }
            std::printf("\n");
        }
        release(dsb);
        release(qsb);
        release(ob);
        release(xb);
        release(wb);
    }
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    bool dense = false;
    if (argc == 2 && std::strcmp(argv[1], "--moe-dense") == 0) {
        dense = true;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--moe-dense]\n", argc == 0 ? "prefill_sweep" : argv[0]);
        return 2;
    }
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
    if (dense) {
        const int status = run_moe_dense_cells(device, stream, event);
        yuuka_event_destroy(event);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
        return status;
    }
    const yuuka_dtype dtypes[] = {YUUKA_DTYPE_F16, YUUKA_DTYPE_Q8_0, YUUKA_DTYPE_Q4_K,
                                  YUUKA_DTYPE_Q5_K, YUUKA_DTYPE_Q6_K};
    const int64_t rows[] = {1, 4, 8, 16, 32, 64, 128, 256, 512};
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
    // Prefill tail shapes: every supported output-matrix dtype at the real
    // vocabulary width, plus two smaller widths that show the crossover.
    const yuuka_dtype lm_dtypes[] = {YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q6_K,
                                     YUUKA_DTYPE_Q8_0};
    const int64_t lm_out[] = {512, 1024, 2048, 4096, 32768, 248320};
    for (yuuka_dtype dtype : lm_dtypes) {
        for (int64_t n_out : lm_out) {
            for (int64_t m : {1, 16}) {
                run_lm_head(device, stream, event, dtype, 2560, n_out, m);
            }
        }
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return 0;
}
