// gemv_decode_sweep measures the decode matvec family at the exact shapes one
// Qwen3.5-4B Q4_K_M token dispatches, with the matrix resident, so the numbers
// are the hot-matrix upper bound of docs/benchmarks/qwen3.5_4b.md. It compares
// the per-row GEMV the decode plan uses today with the batched form and the
// streaming LM head kernel, and it is the native target for scripts/
// decode_profile.sh because rocprofv3 cannot collect from a Go process.
//
// Cells run in a fixed order: every shape is submitted kSubmissionsPerCell times
// in each of the three forms, one dispatch per submission, so the dispatch order
// of a kernel trace lines up with the printed rows. Keep this in sync with
// kSubmissionsPerCell in scripts/decode_profile.py.
//
// --threshold runs a separate two-form sweep at the decode width instead: the
// per-row GEMV against the streaming kernel for a range of output counts, which
// is the measurement behind the lmHeadMinOutputs crossover. It prints its own
// schema and is not traced.
//
// --int8 measures the int8 decode matvec against the FP32
// streaming kernel on the same matrix, both with a pre-quantized activation and
// with the block-fused quantizer. It prints its own schema and is not traced.
//
// --recurrent is the isolated RECURRENT cell of the decode attribution: the
// delta-net unit of one recurrent layer at the exact decode shape, built from the
// five separate operators against the fused pair, so the time and state
// bandwidth comparison is reproducible from one command. It prints its own schema
// and is not traced.

#include "yuuka_hip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// Five warmups and twenty-five timed submissions per cell. A kernel trace of
// this binary therefore sees a multiple of kSubmissionsPerCell dispatches per
// cell, and the percentile below has enough samples to reject interference.
constexpr int kWarmupSubmissions = 5;
constexpr int kTimedSubmissions = 25;
constexpr int kSubmissionsPerCell = kWarmupSubmissions + kTimedSubmissions;

// The batch the batched-GEMV column uses. Four rows is the smallest batch that
// amortizes the per-submission floor without leaving the small-row regime.
constexpr uint64_t kBatchRows = 4;

using Clock = std::chrono::steady_clock;

// Encode a float as an IEEE-754 binary16 value. The sweep only needs a sane
// non-zero scale in the packed blocks, so the slow path never runs.
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

const char * dtype_name(uint32_t dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F16: return "F16";
        case YUUKA_DTYPE_Q4_0: return "Q4_0";
        case YUUKA_DTYPE_Q5_0: return "Q5_0";
        case YUUKA_DTYPE_Q8_0: return "Q8_0";
        case YUUKA_DTYPE_Q4_K: return "Q4_K";
        case YUUKA_DTYPE_Q5_K: return "Q5_K";
        case YUUKA_DTYPE_Q6_K: return "Q6_K";
        default: return "?";
    }
}

// fill_weights writes deterministic packed blocks with the scale fields in a
// plausible range, so the kernels execute the same instruction mix a real GGUF
// row does and no value saturates.
void fill_weights(std::vector<uint8_t> & weights, uint32_t dtype, int64_t n_in,
                  int64_t n_out, std::mt19937 & rng) {
    const uint64_t bytes = row_bytes(dtype, n_in);
    for (int64_t row = 0; row < n_out; ++row) {
        uint8_t * base = weights.data() + row * bytes;
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
        } else if (dtype == YUUKA_DTYPE_F16) {
            for (int64_t i = 0; i < n_in; ++i) {
                put_half(base + i * 2, float(int(rng() % 200) - 100) / 1000.0f);
            }
        }
    }
}

// Shape is one decode projection: the name matches the tensor role the
// attribution uses, so the CSV joins with the per-tensor table of
// docs/benchmarks/qwen3.5_4b.md.
struct Shape {
    const char * name;
    int64_t n_in;
    int64_t n_out;
    uint32_t dtype;
};

// One form of the matvec: the plan operation code and the batch it submits.
struct Form {
    const char * name;
    uint32_t code;
    uint64_t batch;
};

yuuka_op make_op(uint32_t code, yuuka_buffer * w, yuuka_buffer * x,
                 yuuka_buffer * out, uint64_t n_in, uint64_t n_out,
                 uint64_t rows, uint32_t dtype) {
    yuuka_op op{};
    op.code = code;
    op.buf[0] = w;
    op.buf[1] = x;
    op.buf[2] = out;
    op.n[0] = n_in;
    op.n[1] = n_out;
    op.n[2] = rows;
    op.u[0] = dtype;
    return op;
}

// ns_per_row in the output is the submission time divided by the output rows it
// produced, that is n_out times the batch, so the batched form is comparable
// per row rather than per submission.
//
// A submission of one of these cells takes 10 to 350 microseconds, which is
// short enough that a host deschedule during it moves the measurement by a
// factor of three: the APU shares memory and CPU with the process, and the
// driver reclaims pages between cells. The recorded cell time is therefore the
// median of the timed submissions, and a cell whose median moves by more than a
// few percent between runs is reported as noise rather than as a result. The
// figure is a host timing of the submit and the wait because the C ABI has no
// event elapsed-time call; the kernel time without the launch cost comes from
// the kernel trace that scripts/decode_profile.sh reduces.
double time_plan(yuuka_plan * plan, yuuka_stream * stream, yuuka_event * completion) {
    for (int i = 0; i < kWarmupSubmissions; ++i) {
        if (!check(yuuka_plan_execute(plan, stream, 0, 0, 1, completion), "warmup") ||
            !check(yuuka_event_wait(completion), "warmup wait")) {
            return 0;
        }
    }
    std::vector<double> samples;
    samples.reserve(kTimedSubmissions);
    for (int i = 0; i < kTimedSubmissions; ++i) {
        const auto started = Clock::now();
        if (!check(yuuka_plan_execute(plan, stream, 0, 0, 1, completion), "execute") ||
            !check(yuuka_event_wait(completion), "wait")) {
            return 0;
        }
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Cell is one device-resident matrix with its activation and output rows.
struct Cell {
    yuuka_buffer * weights = nullptr;
    yuuka_buffer * x = nullptr;
    yuuka_buffer * out = nullptr;
    uint64_t weight_bytes = 0;
};

// alloc_cell allocates and fills one packed matrix plus rows activation rows and
// rows output rows. The matrix stays resident for every form measured on it.
bool alloc_cell(yuuka_device * device, const Shape & shape, uint64_t rows,
                std::mt19937 & rng, Cell & cell) {
    const uint64_t bytes_per_row = row_bytes(shape.dtype, shape.n_in);
    if (bytes_per_row == 0) return false;
    cell.weight_bytes = bytes_per_row * uint64_t(shape.n_out);
    std::vector<uint8_t> weights(static_cast<size_t>(cell.weight_bytes), uint8_t(0));
    fill_weights(weights, shape.dtype, shape.n_in, shape.n_out, rng);
    if (!check(yuuka_buffer_alloc(device, cell.weight_bytes, &cell.weights), "alloc weights") ||
        !check(yuuka_memcpy_h2d(cell.weights, weights.data(), cell.weight_bytes),
               "h2d weights")) {
        return false;
    }
    std::vector<float> x(size_t(shape.n_in) * size_t(rows));
    for (float & value : x) {
        value = float(int(rng() % 200) - 100) / 100.0f;
    }
    const uint64_t x_bytes = uint64_t(x.size()) * sizeof(float);
    const uint64_t out_bytes = uint64_t(shape.n_out) * rows * sizeof(float);
    return check(yuuka_buffer_alloc(device, x_bytes, &cell.x), "alloc x") &&
           check(yuuka_memcpy_h2d(cell.x, x.data(), x_bytes), "h2d x") &&
           check(yuuka_buffer_alloc(device, out_bytes, &cell.out), "alloc out");
}

void free_cell(Cell & cell) {
    if (cell.out != nullptr) yuuka_buffer_free(cell.out);
    if (cell.x != nullptr) yuuka_buffer_free(cell.x);
    if (cell.weights != nullptr) yuuka_buffer_free(cell.weights);
    cell = Cell{};
}

// submit measures one plan operation on the cell and returns the median
// submission in milliseconds.
double submit(yuuka_device * device, yuuka_stream * stream, yuuka_event * completion,
              const Cell & cell, const Shape & shape, uint32_t code, uint64_t rows) {
    yuuka_plan * plan = nullptr;
    if (!check(yuuka_plan_create(device, &plan), "plan")) {
        return 0;
    }
    yuuka_op op = make_op(code, cell.weights, cell.x, cell.out, uint64_t(shape.n_in),
                          uint64_t(shape.n_out), rows, shape.dtype);
    if (!check(yuuka_plan_add(plan, &op), "plan add")) {
        yuuka_plan_destroy(plan);
        return 0;
    }
    const double ms = time_plan(plan, stream, completion);
    yuuka_plan_destroy(plan);
    return ms;
}

// The recurrent layer geometry of Qwen3.5-4B: 32 heads in 16 groups, a 128-wide
// state, and a 4-tap convolution over 8192 qkv channels whose first 4096 are the
// q and k groups the L2 norm covers.
constexpr int64_t kRecGroups = 16;
constexpr int64_t kRecHeads = 32;
constexpr int64_t kRecState = 128;
constexpr int64_t kRecKernel = 4;
constexpr int64_t kRecInner = kRecHeads * kRecState;
constexpr int64_t kRecKey = kRecGroups * kRecState;
constexpr int64_t kRecConvDim = 2 * kRecKey + kRecInner;

bool alloc_floats(yuuka_device * device, int64_t count, yuuka_buffer ** out) {
    std::vector<float> host(static_cast<size_t>(count), 0.01f);
    if (!check(yuuka_buffer_alloc(device, uint64_t(count) * sizeof(float), out), "alloc")) {
        return false;
    }
    return check(yuuka_memcpy_h2d(*out, host.data(), uint64_t(count) * sizeof(float)), "h2d");
}

yuuka_op make_elementwise(uint32_t code, uint64_t n) {
    yuuka_op op{};
    op.code = code;
    op.n[0] = n;
    return op;
}

// run_recurrent times one recurrent layer's delta-net unit in the unfused and
// fused plans. Both submit the same arithmetic: the separate operators are the
// CONV1D, two L2NORM, GDN, and GATED_RMSNORM operations of the legacy plan, and
// the fused plan is CONV_L2NORM plus RECURRENT_TOKEN.
int run_recurrent(yuuka_device * device, yuuka_stream * stream, yuuka_event * completion) {
    yuuka_buffer * conv_state = nullptr;
    yuuka_buffer * conv_w = nullptr;
    yuuka_buffer * qkv = nullptr;
    yuuka_buffer * conv_out = nullptr;
    yuuka_buffer * rec_state = nullptr;
    yuuka_buffer * alpha = nullptr;
    yuuka_buffer * beta = nullptr;
    yuuka_buffer * dt = nullptr;
    yuuka_buffer * sa = nullptr;
    yuuka_buffer * gdn_out = nullptr;
    yuuka_buffer * gate = nullptr;
    yuuka_buffer * norm = nullptr;
    yuuka_buffer * gdn_norm = nullptr;
    if (!alloc_floats(device, kRecConvDim * (kRecKernel - 1), &conv_state) ||
        !alloc_floats(device, kRecConvDim * kRecKernel, &conv_w) ||
        !alloc_floats(device, kRecConvDim, &qkv) ||
        !alloc_floats(device, kRecConvDim, &conv_out) ||
        !alloc_floats(device, kRecHeads * kRecState * kRecState, &rec_state) ||
        !alloc_floats(device, kRecHeads, &alpha) || !alloc_floats(device, kRecHeads, &beta) ||
        !alloc_floats(device, kRecHeads, &dt) || !alloc_floats(device, kRecHeads, &sa) ||
        !alloc_floats(device, kRecInner, &gdn_out) || !alloc_floats(device, kRecInner, &gate) ||
        !alloc_floats(device, kRecState, &norm) || !alloc_floats(device, kRecInner, &gdn_norm)) {
        return 1;
    }
    const float eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(float(kRecState));

    yuuka_plan * legacy = nullptr;
    yuuka_plan * fused = nullptr;
    if (!check(yuuka_plan_create(device, &legacy), "legacy plan") ||
        !check(yuuka_plan_create(device, &fused), "fused plan")) {
        return 1;
    }
    int status = 0;
    yuuka_op op{};
    op.code = YUUKA_OP_CONV1D;
    op.buf[0] = conv_state;
    op.buf[1] = conv_w;
    op.buf[2] = qkv;
    op.buf[3] = conv_out;
    op.n[0] = uint64_t(kRecConvDim);
    op.n[1] = uint64_t(kRecKernel);
    status |= !check(yuuka_plan_add(legacy, &op), "add conv1d");
    for (int pass = 0; pass < 2; ++pass) {
        yuuka_op norm_op{};
        norm_op.code = YUUKA_OP_L2NORM;
        norm_op.buf[0] = conv_out;
        norm_op.buf[2] = conv_out;
        norm_op.off[0] = uint64_t(pass == 0 ? 0 : kRecKey) * sizeof(float);
        norm_op.off[2] = norm_op.off[0];
        norm_op.n[0] = uint64_t(kRecGroups);
        norm_op.n[1] = uint64_t(kRecState);
        norm_op.n[2] = uint64_t(kRecState);
        norm_op.f[0] = eps;
        norm_op.f[1] = pass == 0 ? q_scale : 1.0f;
        status |= !check(yuuka_plan_add(legacy, &norm_op), "add l2norm");
    }
    yuuka_op gdn{};
    gdn.code = YUUKA_OP_GDN;
    gdn.buf[0] = rec_state;
    gdn.buf[1] = conv_out;
    gdn.buf[2] = conv_out;
    gdn.buf[3] = conv_out;
    gdn.buf[4] = alpha;
    gdn.buf[5] = beta;
    gdn.buf[6] = dt;
    gdn.buf[7] = sa;
    gdn.buf[8] = gdn_out;
    gdn.off[2] = uint64_t(kRecKey) * sizeof(float);
    gdn.off[3] = uint64_t(2 * kRecKey) * sizeof(float);
    gdn.n[0] = uint64_t(kRecGroups);
    gdn.n[1] = uint64_t(kRecHeads);
    gdn.n[2] = uint64_t(kRecState);
    status |= !check(yuuka_plan_add(legacy, &gdn), "add gdn");
    yuuka_op gnorm{};
    gnorm.code = YUUKA_OP_GATED_RMSNORM;
    gnorm.buf[0] = gdn_out;
    gnorm.buf[1] = norm;
    gnorm.buf[2] = gate;
    gnorm.buf[3] = gdn_norm;
    gnorm.n[0] = uint64_t(kRecState);
    gnorm.n[1] = uint64_t(kRecHeads);
    gnorm.n[2] = uint64_t(kRecState);
    gnorm.f[0] = eps;
    status |= !check(yuuka_plan_add(legacy, &gnorm), "add gated rmsnorm");

    yuuka_op conv_l2{};
    conv_l2.code = YUUKA_OP_CONV_L2NORM;
    conv_l2.buf[0] = conv_state;
    conv_l2.buf[1] = conv_w;
    conv_l2.buf[2] = qkv;
    conv_l2.buf[3] = conv_out;
    conv_l2.n[0] = uint64_t(kRecConvDim);
    conv_l2.n[1] = uint64_t(kRecKernel);
    conv_l2.n[2] = uint64_t(2 * kRecKey);
    conv_l2.n[3] = uint64_t(kRecKey);
    conv_l2.n[4] = uint64_t(kRecState);
    conv_l2.f[0] = eps;
    conv_l2.f[1] = q_scale;
    status |= !check(yuuka_plan_add(fused, &conv_l2), "add conv l2norm");
    yuuka_op token{};
    token.code = YUUKA_OP_RECURRENT_TOKEN;
    token.buf[0] = rec_state;
    token.buf[1] = conv_out;
    token.buf[2] = conv_out;
    token.buf[3] = conv_out;
    token.buf[4] = alpha;
    token.buf[5] = beta;
    token.buf[6] = dt;
    token.buf[7] = sa;
    token.buf[8] = gdn_norm;
    token.buf[9] = norm;
    token.buf[10] = gate;
    token.off[2] = uint64_t(kRecKey) * sizeof(float);
    token.off[3] = uint64_t(2 * kRecKey) * sizeof(float);
    token.n[0] = uint64_t(kRecGroups);
    token.n[1] = uint64_t(kRecHeads);
    token.n[2] = uint64_t(kRecState);
    token.f[0] = eps;
    status |= !check(yuuka_plan_add(fused, &token), "add recurrent token");

    const double state_bytes = double(kRecHeads) * kRecState * kRecState * sizeof(float);
    std::printf("layers,%lld,groups,%lld,heads,%lld,state,%lld,conv_channels,%lld\n",
                static_cast<long long>(kRecConvDim > 0 ? 24 : 0),
                static_cast<long long>(kRecGroups), static_cast<long long>(kRecHeads),
                static_cast<long long>(kRecState), static_cast<long long>(kRecConvDim));
    std::printf("variant,ops,ms_per_layer,ms_24_layers,state_GB/s,state_MB_per_token\n");
    struct Form {
        const char * name;
        yuuka_plan * plan;
        int ops;
        double traversals;
    } forms[] = {
        {"legacy_5_op", legacy, 5, 5.0},
        {"fused_2_op", fused, 2, 2.0},
    };
    for (const Form & form : forms) {
        if (status != 0) {
            break;
        }
        const double ms = time_plan(form.plan, stream, completion);
        if (ms <= 0) {
            std::printf("%s,failed\n", form.name);
            status = 1;
            break;
        }
        const double bytes = form.traversals * state_bytes;
        std::printf("%s,%d,%.4f,%.1f,%.1f,%.1f\n", form.name, form.ops, ms, ms * 24.0,
                    bytes / (ms * 1.0e6), bytes * 24.0 / 1.0e6);
    }
    yuuka_plan_destroy(legacy);
    yuuka_plan_destroy(fused);
    yuuka_buffer_free(conv_state);
    yuuka_buffer_free(conv_w);
    yuuka_buffer_free(qkv);
    yuuka_buffer_free(conv_out);
    yuuka_buffer_free(rec_state);
    yuuka_buffer_free(alpha);
    yuuka_buffer_free(beta);
    yuuka_buffer_free(dt);
    yuuka_buffer_free(sa);
    yuuka_buffer_free(gdn_out);
    yuuka_buffer_free(gate);
    yuuka_buffer_free(norm);
    yuuka_buffer_free(gdn_norm);
    return status;
}

} // namespace

int main(int argc, char ** argv) {
    bool threshold = false;
    bool int8 = false;
    bool recurrent = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threshold") {
            threshold = true;
        } else if (arg == "--int8") {
            int8 = true;
        } else if (arg == "--recurrent") {
            recurrent = true;
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--threshold|--int8|--recurrent]\n"
                        "  (no option)  the twelve decode shapes in the per-row, batched,\n"
                        "               and streaming forms, traced by scripts/decode_profile.sh\n"
                        "  --threshold  the rows = 1 output projection crossover per dtype\n"
                        "  --int8       the int8 decode matvec against the FP32 streaming kernel\n"
                        "  --recurrent  the delta-net unit of one recurrent layer, five separate\n"
                        "               operators against the fused pair\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * completion = nullptr;
    if (!check(yuuka_stream_create(device, &stream), "stream") ||
        !check(yuuka_event_create(device, &completion), "event")) {
        return 1;
    }
    std::mt19937 rng(1234);
    int status = 0;

    if (int8) {
        // The int8 route at the twelve decode shapes: the FP32
        // streaming kernel, the same matvec over a pre-quantized activation, and
        // the fused variant that quantizes inside the block.
        const Shape shapes[] = {
            {"ffn_gate", 2560, 9216, YUUKA_DTYPE_Q4_K},
            {"ffn_up", 2560, 9216, YUUKA_DTYPE_Q4_K},
            {"ffn_down", 9216, 2560, YUUKA_DTYPE_Q6_K},
            {"ffn_down_q4k", 9216, 2560, YUUKA_DTYPE_Q4_K},
            {"attn_q", 2560, 8192, YUUKA_DTYPE_Q4_K},
            {"attn_k", 2560, 1024, YUUKA_DTYPE_Q4_K},
            {"attn_v", 2560, 1024, YUUKA_DTYPE_Q6_K},
            {"attn_v_q4k", 2560, 1024, YUUKA_DTYPE_Q4_K},
            {"attn_output", 4096, 2560, YUUKA_DTYPE_Q4_K},
            {"attn_gate", 2560, 4096, YUUKA_DTYPE_Q4_K},
            {"attn_qkv", 2560, 8192, YUUKA_DTYPE_Q5_K},
            {"ssm_out", 4096, 2560, YUUKA_DTYPE_Q5_K},
            {"ssm_beta", 2560, 32, YUUKA_DTYPE_Q8_0},
            {"lm_head", 2560, 248320, YUUKA_DTYPE_Q6_K},
        };
        std::printf("name,dtype,n_in,n_out,fp32_ms,fp32_GB/s,prequant_ms,"
                    "prequant_GB/s,fused_ms,fused_GB/s\n");
        for (const Shape & shape : shapes) {
            const uint64_t bytes_per_row = row_bytes(shape.dtype, shape.n_in);
            if (bytes_per_row == 0) {
                std::fprintf(stderr, "SKIP shape %s: unsupported dtype %u\n", shape.name,
                             shape.dtype);
                continue;
            }
            if (!(shape.dtype == YUUKA_DTYPE_Q4_K || shape.dtype == YUUKA_DTYPE_Q5_K ||
                  shape.dtype == YUUKA_DTYPE_Q6_K || shape.dtype == YUUKA_DTYPE_Q8_0)) {
                // Only these four have an int8 row decoder.
                continue;
            }
            Cell cell;
            if (!alloc_cell(device, shape, 1, rng, cell)) {
                free_cell(cell);
                status = 1;
                break;
            }
            // The digits the pre-quantized form reads, produced once like the
            // plan does for a group of matvecs.
            yuuka_buffer * qs = nullptr;
            yuuka_buffer * ds = nullptr;
            if (!check(yuuka_buffer_alloc(device, uint64_t(shape.n_in), &qs), "alloc q8") ||
                !check(yuuka_buffer_alloc(device, uint64_t(shape.n_in / 32) * 8, &ds),
                       "alloc q8 scales")) {
                status = 1;
            }
            yuuka_plan * quant = nullptr;
            if (status == 0 && check(yuuka_plan_create(device, &quant), "quant plan")) {
                yuuka_op op{};
                op.code = YUUKA_OP_QUANT_Q8_1;
                op.buf[0] = cell.x;
                op.buf[1] = qs;
                op.buf[2] = ds;
                op.n[0] = uint64_t(shape.n_in);
                op.n[1] = 1;
                if (!check(yuuka_plan_add(quant, &op), "add quantize") ||
                    !check(yuuka_plan_execute(quant, stream, 0, 0, 1, completion),
                           "execute quantize") ||
                    !check(yuuka_event_wait(completion), "quantize wait")) {
                    status = 1;
                }
                yuuka_plan_destroy(quant);
            }
            const double bytes = double(bytes_per_row) * double(shape.n_out);
            const double fp32_ms = status == 0
                ? submit(device, stream, completion, cell, shape, YUUKA_OP_LM_HEAD, 1) : 0;
            double pre_ms = 0;
            double fused_ms = 0;
            if (status == 0) {
                yuuka_plan * plan = nullptr;
                if (check(yuuka_plan_create(device, &plan), "int8 plan")) {
                    yuuka_op op{};
                    op.code = YUUKA_OP_GEMV_Q8_1;
                    op.buf[0] = cell.weights;
                    op.buf[1] = qs;
                    op.buf[2] = ds;
                    op.buf[3] = cell.out;
                    op.n[0] = uint64_t(shape.n_in);
                    op.n[1] = uint64_t(shape.n_out);
                    op.u[0] = shape.dtype;
                    op.u[1] = 0;
                    if (check(yuuka_plan_add(plan, &op), "add int8")) {
                        pre_ms = time_plan(plan, stream, completion);
                    } else {
                        status = 1;
                    }
                    yuuka_plan_destroy(plan);
                }
            }
            if (status == 0) {
                yuuka_plan * plan = nullptr;
                if (check(yuuka_plan_create(device, &plan), "fused plan")) {
                    yuuka_op op{};
                    op.code = YUUKA_OP_GEMV_Q8_1;
                    op.buf[0] = cell.weights;
                    op.buf[1] = cell.x;
                    op.buf[3] = cell.out;
                    op.n[0] = uint64_t(shape.n_in);
                    op.n[1] = uint64_t(shape.n_out);
                    op.u[0] = shape.dtype;
                    op.u[1] = 1;
                    if (check(yuuka_plan_add(plan, &op), "add fused")) {
                        fused_ms = time_plan(plan, stream, completion);
                    } else {
                        status = 1;
                    }
                    yuuka_plan_destroy(plan);
                }
            }
            if (status == 0 && (fp32_ms <= 0 || pre_ms <= 0 || fused_ms <= 0)) {
                status = 1;
            }
            if (status == 0) {
                std::printf("%s,%s,%lld,%lld,%.4f,%.1f,%.4f,%.1f,%.4f,%.1f\n", shape.name,
                            dtype_name(shape.dtype),
                            static_cast<long long>(shape.n_in),
                            static_cast<long long>(shape.n_out), fp32_ms,
                            bytes / (fp32_ms * 1e6), pre_ms, bytes / (pre_ms * 1e6),
                            fused_ms, bytes / (fused_ms * 1e6));
                std::fflush(stdout);
            }
            if (ds != nullptr) yuuka_buffer_free(ds);
            if (qs != nullptr) yuuka_buffer_free(qs);
            free_cell(cell);
            if (status != 0) {
                break;
            }
        }
        yuuka_event_destroy(completion);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
        return status;
    }

    if (recurrent) {
        status = run_recurrent(device, stream, completion);
        yuuka_event_destroy(completion);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
        return status;
    }

    if (threshold) {
        // The crossover behind lmHeadMinOutputs, at the decode width of
        // Qwen3.5-4B (n_in = 2560) and one row, over the dtypes the dispatch
        // rule accepts. N ascends so the printed order is the decision order.
        const uint32_t dtypes[] = {YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q5_K,
                                   YUUKA_DTYPE_Q6_K, YUUKA_DTYPE_Q8_0,
                                   YUUKA_DTYPE_F16};
        const int64_t outputs[] = {256, 512, 768, 1024, 1536, 2048};
        std::printf("dtype,n_in,n_out,gemv_ms,gemv_GB/s,lmhead_ms,lmhead_GB/s,winner\n");
        for (uint32_t dtype : dtypes) {
            // The crossover is the smallest output count at which the streaming
            // kernel wins and keeps winning, so a cell that flips back to GEMV
            // inside the launch-dominated range does not move the threshold.
            int64_t sustained = 0;
            for (int64_t n_out : outputs) {
                const Shape shape = {"threshold", 2560, n_out, dtype};
                Cell cell;
                if (!alloc_cell(device, shape, 1, rng, cell)) {
                    free_cell(cell);
                    status = 1;
                    break;
                }
                const double gemv_ms = submit(device, stream, completion, cell, shape,
                                             YUUKA_OP_GEMV, 1);
                const double lm_ms = submit(device, stream, completion, cell, shape,
                                           YUUKA_OP_LM_HEAD, 1);
                free_cell(cell);
                if (gemv_ms <= 0 || lm_ms <= 0) {
                    status = 1;
                    break;
                }
                const char * winner = lm_ms < gemv_ms ? "lm_head" : "gemv";
                if (gemv_ms <= lm_ms) {
                    sustained = 0;
                } else if (sustained == 0) {
                    sustained = n_out;
                }
                const double bytes = double(row_bytes(dtype, shape.n_in)) *
                                     double(shape.n_out);
                std::printf("%s,%lld,%lld,%.4f,%.1f,%.4f,%.1f,%s\n", dtype_name(dtype),
                            static_cast<long long>(shape.n_in),
                            static_cast<long long>(n_out), gemv_ms,
                            bytes / (gemv_ms * 1e6), lm_ms, bytes / (lm_ms * 1e6),
                            winner);
                std::fflush(stdout);
            }
            if (status == 0) {
                if (sustained == 0) {
                    std::printf("crossover,%s,none in 256..2048 outputs\n",
                                dtype_name(dtype));
                } else {
                    std::printf("crossover,%s,streaming kernel wins from %lld outputs"
                                " upward\n", dtype_name(dtype),
                                static_cast<long long>(sustained));
                }
            }
        }
        yuuka_event_destroy(completion);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
        return status;
    }

    // The decode shapes of Qwen3.5-4B-Q4_K_M, one row per tensor role. attn_v
    // and ffn_down carry two dtypes because the GGUF is mixed inside the
    // preset: attn_v is Q6_K in five layers and Q4_K in three, ffn_down is Q6_K
    // in sixteen and Q4_K in sixteen.
    const Shape shapes[] = {
        {"ffn_gate", 2560, 9216, YUUKA_DTYPE_Q4_K},
        {"ffn_up", 2560, 9216, YUUKA_DTYPE_Q4_K},
        {"ffn_down", 9216, 2560, YUUKA_DTYPE_Q6_K},
        {"ffn_down_q4k", 9216, 2560, YUUKA_DTYPE_Q4_K},
        {"attn_q", 2560, 8192, YUUKA_DTYPE_Q4_K},
        {"attn_k", 2560, 1024, YUUKA_DTYPE_Q4_K},
        {"attn_v", 2560, 1024, YUUKA_DTYPE_Q6_K},
        {"attn_v_q4k", 2560, 1024, YUUKA_DTYPE_Q4_K},
        {"attn_output", 4096, 2560, YUUKA_DTYPE_Q4_K},
        {"attn_gate", 2560, 4096, YUUKA_DTYPE_Q4_K},
        {"attn_qkv", 2560, 8192, YUUKA_DTYPE_Q5_K},
        {"ssm_out", 4096, 2560, YUUKA_DTYPE_Q5_K},
        {"ssm_beta", 2560, 32, YUUKA_DTYPE_Q8_0},
        {"lm_head", 2560, 248320, YUUKA_DTYPE_Q6_K},
    };
    const Form forms[] = {
        {"gemv", YUUKA_OP_GEMV, 1},
        {"gemv_batch4", YUUKA_OP_GEMV_BATCH, kBatchRows},
        {"lm_head", YUUKA_OP_LM_HEAD, 1},
    };

    std::printf("name,dtype,n_in,n_out,ms,GB/s,ns_per_row,form,batch\n");
    for (const Shape & shape : shapes) {
        Cell cell;
        if (row_bytes(shape.dtype, shape.n_in) == 0) {
            std::fprintf(stderr, "SKIP shape %s: unsupported dtype %u\n", shape.name,
                         shape.dtype);
            continue;
        }
        if (!alloc_cell(device, shape, kBatchRows, rng, cell)) {
            free_cell(cell);
            status = 1;
            break;
        }
        for (const Form & form : forms) {
            const double ms = submit(device, stream, completion, cell, shape, form.code,
                                     form.batch);
            if (ms <= 0) {
                status = 1;
                break;
            }
            // GB/s counts the packed weights the submission must read; the
            // activation and the output are three orders of magnitude smaller
            // for every cell here.
            std::printf("%s,%s,%lld,%lld,%.4f,%.1f,%.1f,%s,%llu\n", shape.name,
                        dtype_name(shape.dtype), static_cast<long long>(shape.n_in),
                        static_cast<long long>(shape.n_out), ms,
                        double(cell.weight_bytes) / (ms * 1e6),
                        ms * 1e6 / (double(shape.n_out) * double(form.batch)),
                        form.name, static_cast<unsigned long long>(form.batch));
            std::fflush(stdout);
        }
        free_cell(cell);
        if (status != 0) {
            break;
        }
    }

    yuuka_event_destroy(completion);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return status;
}
