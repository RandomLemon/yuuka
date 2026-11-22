// Pins the IQ2_XXS, IQ2_S, and IQ3_S expert path of the grouped int8 WMMA (MMQ)
// route against the Q8_1 dp4a route and against a host reference over the same
// decoded weights.
//
// The route stages IQ expert rows through the dot8 decoders of
// kernels/hip/iq_decode.cuh and applies the codebook scale of a 16-value or a
// 32-value group inside the int8 WMMA tile. The tests below pin the three
// properties that decoder relies on:
//   - the staged digits and the scale slot of every codebook, by comparing the
//     MMQ projections element by element against the dp4a route at the same
//     slot positions (both consume the same Q8_1 digits, so only a decode or a
//     scale-granularity error separates them);
//   - the per-16 scale of IQ2_S, by building its scale bytes with the two
//     nibbles of one 32-value group far apart: applying the group scale per 32
//     values instead of per 16 would move the projection by more than the
//     tolerance;
//   - the whole pipeline against a host computation over the weights decoded by
//     yuuka_dequant_f16 and the host reproduction of the device Q8_1 pass.
//
// The routing arrays are written by the host so the bucket sizes are exact: a
// bucket of 1, J-1, J, J+1, or 3J+5 assignments cannot come from one top-k chunk
// because an expert that owns more than half of the assignments cannot be paired
// with a distinct expert in every row. moe_route itself is pinned by
// tests/hip/iq_moe_test.cpp and tests/hip/moe_test.cpp.

#include "yuuka_hip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);    \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

#define CHECK_STATUS(call)                                                               \
    do {                                                                                 \
        const yuuka_status st__ = (call);                                                \
        if (st__ != YUUKA_OK) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__,      \
                         #call, int(st__), yuuka_last_error());                          \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

struct context {
    yuuka_device * device = nullptr;
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
};

int setup(context & ctx) {
    if (yuuka_device_open(0, &ctx.device) != YUUKA_OK) {
        return 77;
    }
    CHECK_STATUS(yuuka_stream_create(ctx.device, &ctx.stream));
    CHECK_STATUS(yuuka_event_create(ctx.device, &ctx.event));
    return 0;
}

void teardown(context & ctx) {
    if (ctx.event != nullptr) yuuka_event_destroy(ctx.event);
    if (ctx.stream != nullptr) yuuka_stream_destroy(ctx.stream);
    if (ctx.device != nullptr) yuuka_device_close(ctx.device);
}

yuuka_buffer * upload(context & ctx, const void * data, uint64_t bytes) {
    yuuka_buffer * buffer = nullptr;
    if (yuuka_buffer_alloc(ctx.device, bytes, &buffer) != YUUKA_OK) {
        return nullptr;
    }
    if (data != nullptr && yuuka_memcpy_h2d(buffer, data, bytes) != YUUKA_OK) {
        yuuka_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

float half_to_float(uint16_t h) {
    const bool negative = (h & 0x8000u) != 0;
    const int exponent = (h >> 10) & 0x1F;
    const int mantissa = h & 0x3FF;
    if (exponent == 0x1F) {
        return negative ? -INFINITY : INFINITY;
    }
    const double value = exponent == 0 ? std::ldexp(double(mantissa), -24)
                                      : std::ldexp(double(mantissa | 0x400), exponent - 25);
    return float(negative ? -value : value);
}

std::vector<uint8_t> from_hex(const char * text) {
    auto nibble = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return uint8_t(c - '0');
        return uint8_t((c | 0x20) - 'a' + 10);
    };
    std::vector<uint8_t> out;
    for (const char * p = text; p[0] != '\0' && p[1] != '\0'; p += 2) {
        out.push_back(uint8_t((nibble(p[0]) << 4) | nibble(p[1])));
    }
    return out;
}

uint64_t block_bytes(uint32_t dtype) {
    return dtype == YUUKA_DTYPE_IQ2_XXS ? 66u
         : dtype == YUUKA_DTYPE_IQ2_S   ? 82u
                                        : 110u;
}

// ref_block returns the first block of each codebook from
// Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf, the same fixtures tests/hip/iq_moe_test.cpp
// and tensor/quant_test.go pin against ggml.
std::vector<uint8_t> ref_block(uint32_t dtype) {
    if (dtype == YUUKA_DTYPE_IQ2_XXS) {
        return from_hex(
            "2f0c070067cad0a4b8adba52aa0456f20f7b013a0c004f7332f58f6651311d43d68b23eba5d2"
            "ece74da21c8c5623474cc08e59d71e0d6377b69c22ec15786a8338bb");
    }
    if (dtype == YUUKA_DTYPE_IQ3_S) {
        return from_hex(
            "76051fd85942b4e46e38b10d380fbc2ca7c5c61373867fed8a59c099a3df55e617fe8e574dd8"
            "282e7e8d3f9c002844e737e27987199ddc6f21c6d1f4a1213f94106899a002a7dd78845721b9"
            "846cec02ee3f5e044832ff7cabb771788225979696dd19b7214cf9b6a9638b8a999f");
    }
    return from_hex(
        "57099f38722503000d3a00c3f51de39e4c4a0e193e06d87f1b3f001d4f7f4900b602489e7a4b"
        "8a3a5ea690784a9e256b50e3007be615268e7d57c371164b0f34398a00c024c1000d70b0999b"
        "a959fb9c8ee7");
}

// packed_row returns one expert weight row: the codebook fixture with its FP16
// block scale nudged per row, so neighbouring rows decode different values. For
// IQ2_S the eight group scale bytes are replaced by 0x0F, whose low and high
// nibble are the extreme scale indices of the 32-value group: the first sixteen
// values then decode against a scale that is 31 times smaller than the second
// sixteen, which is what makes a per-32 scale application visible.
std::vector<uint8_t> packed_row(uint32_t dtype, const std::vector<uint8_t> & block,
                                int row) {
    std::vector<uint8_t> out = block;
    out[0] = uint8_t((out[0] + 7 * row) & 0xFF);
    if (dtype == YUUKA_DTYPE_IQ2_S) {
        for (int i = 0; i < 8; ++i) {
            out[74 + i] = 0x0F;
        }
    }
    return out;
}

void fill_matrix(std::vector<uint8_t> & packed, uint32_t dtype,
                 const std::vector<uint8_t> & block, int rows) {
    const uint64_t bytes = block_bytes(dtype);
    packed.resize(size_t(rows) * bytes);
    for (int row = 0; row < rows; ++row) {
        const std::vector<uint8_t> value = packed_row(dtype, block, row);
        std::memcpy(packed.data() + size_t(row) * bytes, value.data(), bytes);
    }
}

// fill_matrix_wide builds a packed matrix whose rows span n_in values and whose
// 256-value block scales change across K blocks. The codebook fixture supplies
// the digits; only the FP16 scale mantissa and the IQ2_S group-scale bytes are
// varied, so every block stays finite and valid.
void fill_matrix_wide(std::vector<uint8_t> & packed, uint32_t dtype,
                      const std::vector<uint8_t> & block, int rows, int n_in,
                      int salt) {
    const int blocks = n_in / 256;
    const uint64_t bytes = block_bytes(dtype);
    packed.resize(size_t(rows) * size_t(blocks) * bytes);
    for (int row = 0; row < rows; ++row) {
        for (int b = 0; b < blocks; ++b) {
            std::vector<uint8_t> value = packed_row(dtype, block, row + salt + 5 * b);
            value[0] = uint8_t((value[0] + 17 * b + salt) & 0xFF);
            if (dtype == YUUKA_DTYPE_IQ2_S) {
                for (int i = 0; i < 8; ++i) {
                    value[74 + i] = 0x0F;
                }
            }
            std::memcpy(packed.data() + (size_t(row) * blocks + b) * bytes,
                        value.data(), bytes);
        }
    }
}

// dequant runs yuuka_dequant_f16 over one packed matrix and returns the decoded
// values as FP32. An empty result means the call failed.
std::vector<float> dequant(context & ctx, const std::vector<uint8_t> & packed,
                           uint64_t n_in, uint64_t n_out, uint32_t dtype) {
    const uint64_t elements = n_in * n_out;
    std::vector<float> out;
    yuuka_buffer * src = upload(ctx, packed.data(), packed.size());
    yuuka_buffer * dst = upload(ctx, nullptr, elements * 2);
    if (src == nullptr || dst == nullptr) {
        if (src != nullptr) yuuka_buffer_free(src);
        if (dst != nullptr) yuuka_buffer_free(dst);
        return out;
    }
    yuuka_status status =
        yuuka_dequant_f16(ctx.stream, dst, src, n_in, n_out, dtype, ctx.event);
    if (status == YUUKA_OK) {
        status = yuuka_event_wait(ctx.event);
    }
    if (status == YUUKA_OK) {
        std::vector<uint16_t> halves(elements);
        status = yuuka_memcpy_d2h(halves.data(), dst, elements * 2);
        if (status == YUUKA_OK) {
            out.resize(elements);
            for (uint64_t i = 0; i < elements; ++i) {
                out[i] = half_to_float(halves[i]);
            }
        }
    }
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "dequant dtype %u failed: %s\n", dtype, yuuka_last_error());
        out.clear();
    }
    yuuka_buffer_free(src);
    yuuka_buffer_free(dst);
    return out;
}

// quantize_q8_1_host reproduces the device Q8_1 quantizer for a matrix of whole
// 32-value blocks: the scale is amax/127 and each value becomes digit * scale.
// That is the value both routes see; the IQ codebooks do not use the block sum
// that the K-quant minimum term reads.
std::vector<float> quantize_q8_1_host(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t base = 0; base + 32 <= x.size(); base += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::fmax(amax, std::fabs(x[base + i]));
        }
        const float scale = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float digit =
                std::fmax(-127.0f, std::fmin(127.0f, std::rint(x[base + i] * inv)));
            out[base + i] = digit * scale;
        }
    }
    return out;
}

struct buffers {
    yuuka_buffer * norm = nullptr;
    yuuka_buffer * norm_q8 = nullptr;
    yuuka_buffer * norm_ds = nullptr;
    yuuka_buffer * weights = nullptr;
    yuuka_buffer * counts = nullptr;
    yuuka_buffer * assign_rows = nullptr;
    yuuka_buffer * assign_slots = nullptr;
    yuuka_buffer * gate_w = nullptr;
    yuuka_buffer * up_w = nullptr;
    yuuka_buffer * down_w = nullptr;
    yuuka_buffer * gate_out = nullptr;
    yuuka_buffer * up_out = nullptr;
    yuuka_buffer * act_q8 = nullptr;
    yuuka_buffer * act_ds = nullptr;
    yuuka_buffer * expert_out = nullptr;
    yuuka_buffer * x = nullptr;
};

void free_buffers(buffers & b) {
    yuuka_buffer * all[] = {b.norm, b.norm_q8, b.norm_ds, b.weights, b.counts,
                            b.assign_rows, b.assign_slots, b.gate_w, b.up_w, b.down_w,
                            b.gate_out, b.up_out, b.act_q8, b.act_ds, b.expert_out,
                            b.x};
    for (yuuka_buffer * buffer : all) {
        if (buffer != nullptr) yuuka_buffer_free(buffer);
    }
    b = buffers{};
}

bool alloc_buffers(context & ctx, buffers & b, int rows, int experts, int used,
                   int hidden, int ffn, const std::vector<uint8_t> & gate_packed,
                   const std::vector<uint8_t> & up_packed,
                   const std::vector<uint8_t> & down_packed,
                   const std::vector<float> & norm, const std::vector<float> & weights,
                   const std::vector<int32_t> & counts,
                   const std::vector<int32_t> & assign_rows,
                   const std::vector<int32_t> & assign_slots) {
    const int slots = rows * used;
    b.norm = upload(ctx, norm.data(), norm.size() * 4);
    b.norm_q8 = upload(ctx, nullptr, uint64_t(rows) * hidden);
    b.norm_ds = upload(ctx, nullptr, uint64_t(rows) * (hidden / 32) * 8);
    b.weights = upload(ctx, weights.data(), weights.size() * 4);
    b.counts = upload(ctx, counts.data(), counts.size() * 4);
    b.assign_rows = upload(ctx, assign_rows.data(), assign_rows.size() * 4);
    b.assign_slots = upload(ctx, assign_slots.data(), assign_slots.size() * 4);
    b.gate_w = upload(ctx, gate_packed.data(), gate_packed.size());
    b.up_w = upload(ctx, up_packed.data(), up_packed.size());
    b.down_w = upload(ctx, down_packed.data(), down_packed.size());
    b.gate_out = upload(ctx, nullptr, uint64_t(slots) * ffn * 4);
    b.up_out = upload(ctx, nullptr, uint64_t(slots) * ffn * 4);
    b.act_q8 = upload(ctx, nullptr, uint64_t(slots) * ffn);
    b.act_ds = upload(ctx, nullptr, uint64_t(slots) * (ffn / 32) * 8);
    b.expert_out = upload(ctx, nullptr, uint64_t(slots) * hidden * 4);
    b.x = upload(ctx, nullptr, uint64_t(rows) * hidden * 4);
    return b.norm != nullptr && b.norm_q8 != nullptr && b.norm_ds != nullptr &&
           b.weights != nullptr && b.counts != nullptr && b.assign_rows != nullptr &&
           b.assign_slots != nullptr && b.gate_w != nullptr && b.up_w != nullptr &&
           b.down_w != nullptr && b.gate_out != nullptr && b.up_out != nullptr &&
           b.act_q8 != nullptr && b.act_ds != nullptr && b.expert_out != nullptr &&
           b.x != nullptr;
}

// route selects the expert projections the plan runs: the int8 WMMA route or the
// Q8_1 dp4a route the former replaces. Both consume the same Q8_1 activation
// tile and the same routing arrays.
enum route { kRouteMMQ, kRouteQ8 };

yuuka_status add_op(yuuka_plan * plan, yuuka_op & op) {
    const yuuka_status status = yuuka_plan_add(plan, &op);
    op = yuuka_op{};
    return status;
}

// run_route executes the routed-expert pipeline over a zero residual row and
// returns the weighted expert output, plus the gate and up tiles when gate_up is
// set. With gate_up false only the down projection runs, over whatever Q8_1
// SwiGLU tile is already in the buffers, which is how the two down kernels are
// compared over identical activation digits.
int run_route(context & ctx, route kind, buffers & b, int rows, int experts, int used,
              int hidden, int ffn, int tile_j, uint32_t dtype, uint32_t down_dtype,
              bool gate_up, std::vector<float> & gate, std::vector<float> & up,
              std::vector<float> & out) {
    const int slots = rows * used;
    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    // Every op of the pipeline is added in order; the first rejection is
    // reported instead of executing a partial plan.
    yuuka_status add_status = YUUKA_OK;
    auto add = [&](yuuka_op & op) {
        if (add_status == YUUKA_OK) {
            add_status = add_op(plan, op);
        }
    };
    yuuka_op op{};
    if (gate_up) {
        op.code = YUUKA_OP_QUANT_Q8_1;
        op.buf[0] = b.norm;
        op.buf[1] = b.norm_q8;
        op.buf[2] = b.norm_ds;
        op.n[0] = hidden;
        op.n[1] = rows;
        add(op);

        if (kind == kRouteMMQ) {
            op.code = YUUKA_OP_MOE_GATE_UP_MMQ;
            op.n[5] = tile_j;
        } else {
            op.code = YUUKA_OP_MOE_GATE_UP_Q8;
        }
        op.buf[0] = b.norm_q8;
        op.buf[1] = b.norm_ds;
        op.buf[2] = b.gate_w;
        op.buf[3] = b.up_w;
        op.buf[4] = b.counts;
        op.buf[5] = b.assign_rows;
        op.buf[6] = b.assign_slots;
        op.buf[7] = b.gate_out;
        op.buf[8] = b.up_out;
        op.n[0] = hidden;
        op.n[1] = ffn;
        op.n[2] = rows;
        op.n[3] = experts;
        op.n[4] = used;
        op.u[0] = dtype;
        add(op);

        op.code = YUUKA_OP_SILU_MUL_QUANT_Q8_1;
        op.buf[0] = b.gate_out;
        op.buf[1] = b.up_out;
        op.buf[2] = b.act_q8;
        op.buf[3] = b.act_ds;
        op.n[0] = ffn;
        op.n[1] = slots;
        add(op);
    }

    if (kind == kRouteMMQ) {
        op.code = YUUKA_OP_MOE_DOWN_MMQ;
        op.n[5] = tile_j;
    } else {
        op.code = YUUKA_OP_MOE_DOWN_Q8;
    }
    op.buf[0] = b.act_q8;
    op.buf[1] = b.act_ds;
    op.buf[2] = b.down_w;
    op.buf[3] = b.counts;
    op.buf[4] = b.assign_rows;
    op.buf[5] = b.assign_slots;
    op.buf[6] = b.expert_out;
    op.n[0] = ffn;
    op.n[1] = hidden;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = down_dtype;
    add(op);

    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = b.x;
    op.buf[1] = b.expert_out;
    op.buf[2] = b.weights;
    op.n[0] = hidden;
    op.n[1] = rows;
    op.n[2] = used;
    add(op);

    if (add_status != YUUKA_OK) {
        std::fprintf(stderr, "plan build failed: %s\n", yuuka_last_error());
        yuuka_plan_destroy(plan);
        return 1;
    }
    std::vector<float> zero(size_t(rows) * hidden, 0.0f);
    CHECK_STATUS(yuuka_memcpy_h2d(b.x, zero.data(), zero.size() * 4));
    yuuka_status status = yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event);
    if (status == YUUKA_OK) status = yuuka_event_wait(ctx.event);
    yuuka_plan_destroy(plan);
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "plan execute failed: %s\n", yuuka_last_error());
        return 1;
    }
    if (gate_up) {
        gate.resize(size_t(slots) * ffn);
        up.resize(size_t(slots) * ffn);
        CHECK_STATUS(yuuka_memcpy_d2h(gate.data(), b.gate_out, gate.size() * 4));
        CHECK_STATUS(yuuka_memcpy_d2h(up.data(), b.up_out, up.size() * 4));
    }
    out.resize(size_t(slots) * hidden);
    CHECK_STATUS(yuuka_memcpy_d2h(out.data(), b.expert_out, out.size() * 4));
    return 0;
}

// worst_diff returns the largest absolute difference and the largest reference
// magnitude, so a failure reports the scale the tolerance acts on.
void worst_diff(const std::vector<float> & got, const std::vector<float> & want,
                float & worst, float & magnitude) {
    worst = 0.0f;
    magnitude = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::fmax(worst, std::fabs(got[i] - want[i]));
        magnitude = std::fmax(magnitude, std::fabs(want[i]));
    }
}

// check_case runs one (dtype, tile_j) cell end to end.
int check_case(context & ctx, uint32_t dtype, int tile_j) {
    const int experts = 6, used = 1, hidden = 256, ffn = 256;
    // One bucket per size, plus an empty bucket: every planned launch includes
    // experts that received no assignment at all, so their blocks must return
    // without staging or writing anything.
    const int counts_host[6] = {1, tile_j - 1, tile_j, tile_j + 1, 3 * tile_j + 5, 0};
    int rows = 0;
    for (int e = 0; e < experts; ++e) {
        rows += counts_host[e];
    }
    const int slots = rows * used;
    const uint64_t block = block_bytes(dtype);
    const std::vector<uint8_t> fixture = ref_block(dtype);
    CHECK(uint64_t(fixture.size()) == block);

    std::vector<std::vector<uint8_t>> packed(3);
    fill_matrix(packed[0], dtype, fixture, experts * ffn);
    fill_matrix(packed[1], dtype, fixture, experts * ffn);
    fill_matrix(packed[2], dtype, fixture, experts * hidden);

    const std::vector<float> gate_host = dequant(ctx, packed[0], hidden, experts * ffn,
                                                dtype);
    const std::vector<float> up_host = dequant(ctx, packed[1], hidden, experts * ffn,
                                              dtype);
    const std::vector<float> down_host = dequant(ctx, packed[2], ffn, experts * hidden,
                                                dtype);
    CHECK(!gate_host.empty() && !up_host.empty() && !down_host.empty());

    std::mt19937 rng(4242u + uint32_t(dtype) * 31u + uint32_t(tile_j));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> norm(size_t(rows) * hidden);
    for (float & v : norm) v = dist(rng);

    // The routing arrays: bucket e holds counts[e] assignments whose activation
    // rows are a shuffled permutation of the chunk, so the gather is not a
    // contiguous walk and no two assignments share an activation row (which
    // would make two slots collide).
    std::vector<int32_t> tile_rows(rows);
    for (int i = 0; i < rows; ++i) tile_rows[i] = i;
    std::shuffle(tile_rows.begin(), tile_rows.end(), rng);
    std::vector<int32_t> counts(experts, 0), assign_rows(size_t(experts) * slots, 0),
        assign_slots(size_t(experts) * slots, 0);
    int next_row = 0;
    for (int e = 0; e < experts; ++e) {
        counts[e] = counts_host[e];
        for (int k = 0; k < counts_host[e]; ++k) {
            const int64_t j = int64_t(e) * rows * used + k;
            assign_rows[j] = tile_rows[next_row++];
        }
    }
    CHECK(next_row == rows);
    std::vector<float> weights(size_t(rows) * used, 1.0f);

    buffers b;
    if (!alloc_buffers(ctx, b, rows, experts, used, hidden, ffn, packed[0], packed[1],
                       packed[2], norm, weights, counts, assign_rows, assign_slots)) {
        std::fprintf(stderr, "buffer allocation failed\n");
        free_buffers(b);
        return 1;
    }

    // The dp4a pipeline runs first, so both routes read the same Q8_1 activation
    // tile; the WMMA run then leaves its own SwiGLU Q8_1 tile in the buffers for
    // the down comparison. The dp4a pipeline's own expert output is not
    // compared: it reads the SwiGLU tile that pipeline requantized from slightly
    // different gate/up sums, so its last stage is not the same input problem.
    std::vector<float> gate_mmq, up_mmq, out_mmq, gate_q8, up_q8, out_q8_pipeline;
    if (run_route(ctx, kRouteQ8, b, rows, experts, used, hidden, ffn, tile_j, dtype,
                  dtype, true, gate_q8, up_q8, out_q8_pipeline) != 0 ||
        run_route(ctx, kRouteMMQ, b, rows, experts, used, hidden, ffn, tile_j, dtype,
                  dtype, true, gate_mmq, up_mmq, out_mmq) != 0) {
        free_buffers(b);
        return 1;
    }

    // The dp4a route and the WMMA route decode the same digits and consume the
    // same Q8_1 values, so an element-by-element comparison at the same slot
    // positions pins the staged digits, the scale slot, and the scatter.
    std::printf("  cell dtype %u rows %d tile_j %d\n", dtype, rows, tile_j);
    std::fflush(stdout);
    float gate_route = 0.0f, gate_scale = 1.0f, up_route = 0.0f, up_scale = 1.0f;
    worst_diff(gate_mmq, gate_q8, gate_route, gate_scale);
    worst_diff(up_mmq, up_q8, up_route, up_scale);
    std::printf("  gate q8 %.4g/%.4g up q8 %.4g/%.4g\n", gate_route, gate_scale,
                up_route, up_scale);
    std::fflush(stdout);
    CHECK(gate_route <= 1e-5f * gate_scale);
    CHECK(up_route <= 1e-5f * up_scale);

    // Host reference over the decoded weights and the host Q8_1 values. The
    // reference weights are the F16 decode of the same blocks, which is the
    // documented 2e-2 bound of the Q8_1 expert route.
    std::vector<float> gate_ref(size_t(slots) * ffn, 0.0f);
    std::vector<float> up_ref(size_t(slots) * ffn, 0.0f);
    for (int e = 0; e < experts; ++e) {
        for (int k = 0; k < counts[e]; ++k) {
            const int64_t j = int64_t(e) * rows * used + k;
            const int slot = assign_rows[j] * used + assign_slots[j];
            const std::vector<float> x = quantize_q8_1_host(
                std::vector<float>(norm.begin() + int64_t(assign_rows[j]) * hidden,
                                   norm.begin() + int64_t(assign_rows[j] + 1) * hidden));
            for (int n = 0; n < ffn; ++n) {
                const float * gate_row = &gate_host[(int64_t(e) * ffn + n) * hidden];
                const float * up_row = &up_host[(int64_t(e) * ffn + n) * hidden];
                float g = 0.0f;
                float u = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    g += gate_row[i] * x[i];
                    u += up_row[i] * x[i];
                }
                gate_ref[int64_t(slot) * ffn + n] = g;
                up_ref[int64_t(slot) * ffn + n] = u;
            }
        }
    }
    float gate_ref_diff = 0.0f, gate_ref_scale = 1.0f, up_ref_diff = 0.0f,
          up_ref_scale = 1.0f;
    worst_diff(gate_mmq, gate_ref, gate_ref_diff, gate_ref_scale);
    worst_diff(up_mmq, up_ref, up_ref_diff, up_ref_scale);
    CHECK(gate_ref_diff <= 2e-2f * gate_ref_scale);
    CHECK(up_ref_diff <= 2e-2f * up_ref_scale);

    // The down projection is compared over the Q8_1 SwiGLU tile the WMMA run
    // produced: the dp4a down kernel is run again over that tile, so both kernels
    // consume identical activation digits and the comparison covers the down
    // staging and the slot activation read without the requantization step in
    // between.
    std::vector<int8_t> act_q8(size_t(slots) * ffn);
    std::vector<float> act_ds(size_t(slots) * (ffn / 32) * 2);
    CHECK_STATUS(yuuka_memcpy_d2h(act_q8.data(), b.act_q8, act_q8.size()));
    CHECK_STATUS(yuuka_memcpy_d2h(act_ds.data(), b.act_ds, act_ds.size() * 4));
    std::vector<float> down_q8;
    if (run_route(ctx, kRouteQ8, b, rows, experts, used, hidden, ffn, tile_j, dtype,
                  dtype, false, gate_q8, up_q8, down_q8) != 0) {
        free_buffers(b);
        return 1;
    }
    float down_route = 0.0f, down_scale = 1.0f;
    worst_diff(out_mmq, down_q8, down_route, down_scale);
    CHECK(down_route <= 1e-5f * down_scale);

    std::vector<float> out_ref(size_t(slots) * hidden, 0.0f);
    for (int e = 0; e < experts; ++e) {
        for (int k = 0; k < counts[e]; ++k) {
            const int64_t j = int64_t(e) * rows * used + k;
            const int slot = assign_rows[j] * used + assign_slots[j];
            for (int n = 0; n < hidden; ++n) {
                const float * row = &down_host[(int64_t(e) * hidden + n) * ffn];
                float value = 0.0f;
                for (int i = 0; i < ffn; ++i) {
                    value += float(act_q8[int64_t(slot) * ffn + i]) *
                             act_ds[int64_t(slot) * (ffn / 32) * 2 + (i / 32) * 2] *
                             row[i];
                }
                out_ref[int64_t(slot) * hidden + n] = value;
            }
        }
    }
    float down_ref_diff = 0.0f, down_ref_scale = 1.0f;
    worst_diff(out_mmq, out_ref, down_ref_diff, down_ref_scale);
    free_buffers(b);
    CHECK(down_ref_diff <= 2e-2f * down_ref_scale);

    std::printf("  dtype %u rows %d tile_j %d buckets %d/%d/%d/%d/%d: "
                "gate q8 %.3g/%.3g ref %.3g/%.3g up q8 %.3g/%.3g ref %.3g/%.3g "
                "down q8 %.3g/%.3g ref %.3g/%.3g\n",
                dtype, rows, tile_j, counts_host[0], counts_host[1], counts_host[2],
                counts_host[3], counts_host[4], gate_route, gate_scale, gate_ref_diff,
                gate_ref_scale, up_route, up_scale, up_ref_diff, up_ref_scale,
                down_route, down_scale, down_ref_diff, down_ref_scale);
    std::fflush(stdout);
    return 0;
}

// check_direct_case runs the measured qwen35moe gate/up cell through the direct
// IQ2_XXS layout: 512 tokens, 17 experts, top-8, hidden 2048, expert FFN 512,
// and 32-assignment tiles. Expert 16 is empty; the replaced slots make bucket
// sizes 1, 15, 31, 32, 33, 63, 97, and 127, which covers exact tile boundaries,
// tails, and multi-tile buckets. Every bucket is shuffled with a fixed seed.
// All slots are compared with dp4a on the same Q8_1 digits, and sampled
// channels are compared with the independent host arithmetic over the decoded
// weights.
int check_direct_case(context & ctx, uint32_t down_dtype) {
    constexpr int experts = 17, used = 8, hidden = 2048, ffn = 512, rows = 512;
    constexpr int tile_j = 32;
    const uint32_t gate_dtype = YUUKA_DTYPE_IQ2_XXS;
    const int slots = rows * used;
    CHECK(down_dtype == YUUKA_DTYPE_IQ2_S || down_dtype == YUUKA_DTYPE_IQ3_S);

    std::vector<std::vector<uint8_t>> packed(3);
    fill_matrix_wide(packed[0], gate_dtype, ref_block(gate_dtype), experts * ffn,
                     hidden, 11);
    fill_matrix_wide(packed[1], gate_dtype, ref_block(gate_dtype), experts * ffn,
                     hidden, 23);
    fill_matrix_wide(packed[2], down_dtype, ref_block(down_dtype), experts * hidden,
                     ffn, 37);

    const std::vector<float> gate_host =
        dequant(ctx, packed[0], hidden, experts * ffn, gate_dtype);
    const std::vector<float> up_host =
        dequant(ctx, packed[1], hidden, experts * ffn, gate_dtype);
    const std::vector<float> down_host =
        dequant(ctx, packed[2], ffn, experts * hidden, down_dtype);
    CHECK(!gate_host.empty() && !up_host.empty() && !down_host.empty());

    std::mt19937 rng(9001u + uint32_t(down_dtype));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> norm(size_t(rows) * hidden);
    for (float & v : norm) v = dist(rng);

    const int replace_after[used] = {1, 15, 31, 32, 33, 63, 97, 127};
    std::vector<int32_t> counts(experts, 0);
    std::vector<int32_t> assign_rows(size_t(experts) * slots, 0);
    std::vector<int32_t> assign_slots(size_t(experts) * slots, 0);
    for (int r = 0; r < rows; ++r) {
        for (int s = 0; s < used; ++s) {
            int e = (r + s) % 8;
            if (r < replace_after[s]) e = 8 + s;
            const int64_t j = int64_t(e) * slots + counts[e]++;
            assign_rows[j] = r;
            assign_slots[j] = s;
        }
    }
    CHECK(counts[16] == 0);
    for (int e = 0; e < experts; ++e) {
        std::vector<std::pair<int32_t, int32_t>> entries;
        entries.reserve(counts[e]);
        for (int k = 0; k < counts[e]; ++k) {
            const int64_t j = int64_t(e) * slots + k;
            entries.emplace_back(assign_rows[j], assign_slots[j]);
        }
        std::shuffle(entries.begin(), entries.end(), rng);
        for (int k = 0; k < counts[e]; ++k) {
            const int64_t j = int64_t(e) * slots + k;
            assign_rows[j] = entries[k].first;
            assign_slots[j] = entries[k].second;
        }
    }
    std::vector<float> weights(size_t(rows) * used, 1.0f);

    buffers b;
    if (!alloc_buffers(ctx, b, rows, experts, used, hidden, ffn, packed[0], packed[1],
                       packed[2], norm, weights, counts, assign_rows, assign_slots)) {
        std::fprintf(stderr, "direct IQ2 buffer allocation failed\n");
        free_buffers(b);
        return 1;
    }

    std::vector<float> gate_mmq, up_mmq, out_mmq, gate_q8, up_q8, out_q8_pipeline;
    if (run_route(ctx, kRouteQ8, b, rows, experts, used, hidden, ffn, tile_j,
                  gate_dtype, down_dtype, true, gate_q8, up_q8, out_q8_pipeline) != 0 ||
        run_route(ctx, kRouteMMQ, b, rows, experts, used, hidden, ffn, tile_j,
                  gate_dtype, down_dtype, true, gate_mmq, up_mmq, out_mmq) != 0) {
        free_buffers(b);
        return 1;
    }
    float gate_route = 0.0f, gate_scale = 1.0f, up_route = 0.0f, up_scale = 1.0f;
    worst_diff(gate_mmq, gate_q8, gate_route, gate_scale);
    worst_diff(up_mmq, up_q8, up_route, up_scale);
    CHECK(gate_route <= 1e-5f * gate_scale);
    CHECK(up_route <= 1e-5f * up_scale);

    std::vector<int8_t> act_q8(size_t(slots) * ffn);
    std::vector<float> act_ds(size_t(slots) * (ffn / 32) * 2);
    CHECK_STATUS(yuuka_memcpy_d2h(act_q8.data(), b.act_q8, act_q8.size()));
    CHECK_STATUS(yuuka_memcpy_d2h(act_ds.data(), b.act_ds, act_ds.size() * 4));
    std::vector<float> down_q8;
    if (run_route(ctx, kRouteQ8, b, rows, experts, used, hidden, ffn, tile_j,
                  gate_dtype, down_dtype, false, gate_q8, up_q8, down_q8) != 0) {
        free_buffers(b);
        return 1;
    }
    float down_route = 0.0f, down_scale = 1.0f;
    worst_diff(out_mmq, down_q8, down_route, down_scale);
    CHECK(down_route <= 1e-5f * down_scale);

    // Sample the first, middle, and last assignment of every nonempty bucket
    // and check four output channels against the independent host product.
    const int gate_channels[4] = {0, 1, 7, ffn - 1};
    const int down_channels[4] = {0, 1, 7, hidden - 1};
    int sampled = 0;
    for (int e = 0; e < experts; ++e) {
        if (counts[e] == 0) {
            continue;
        }
        const int picks[3] = {0, counts[e] / 2, counts[e] - 1};
        for (int pick : picks) {
            const int64_t j = int64_t(e) * slots + pick;
            const int row = assign_rows[j];
            const int slot = row * used + assign_slots[j];
            const std::vector<float> x = quantize_q8_1_host(
                std::vector<float>(norm.begin() + int64_t(row) * hidden,
                                   norm.begin() + int64_t(row + 1) * hidden));
            for (int c : gate_channels) {
                const float * grow = &gate_host[(int64_t(e) * ffn + c) * hidden];
                const float * urow = &up_host[(int64_t(e) * ffn + c) * hidden];
                float g = 0.0f;
                float u = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    g += grow[i] * x[i];
                    u += urow[i] * x[i];
                }
                CHECK(std::fabs(gate_mmq[int64_t(slot) * ffn + c] - g) <=
                      2e-2f * (1.0f + std::fabs(g)));
                CHECK(std::fabs(up_mmq[int64_t(slot) * ffn + c] - u) <=
                      2e-2f * (1.0f + std::fabs(u)));
            }
            for (int c : down_channels) {
                const float * drow = &down_host[(int64_t(e) * hidden + c) * ffn];
                float value = 0.0f;
                for (int i = 0; i < ffn; ++i) {
                    value += float(act_q8[int64_t(slot) * ffn + i]) *
                             act_ds[int64_t(slot) * (ffn / 32) * 2 + (i / 32) * 2] *
                             drow[i];
                }
                CHECK(std::fabs(out_mmq[int64_t(slot) * hidden + c] - value) <=
                      2e-2f * (1.0f + std::fabs(value)));
            }
            ++sampled;
        }
    }
    free_buffers(b);
    std::printf("  direct IQ2_XXS gate/up, %s down, M=%d buckets "
                "1/15/31/32/33/63/97/127 empty=16, %d sampled: "
                "gate q8 %.3g up %.3g down q8 %.3g\n",
                down_dtype == YUUKA_DTYPE_IQ2_S ? "IQ2_S" : "IQ3_S", rows, sampled,
                gate_route, up_route, down_route);
    std::fflush(stdout);
    return 0;
}

} // namespace

int main() {
    context ctx;
    if (setup(ctx) != 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }

    std::printf("iq moe MMQ grouped expert route\n");
    for (uint32_t dtype : {uint32_t(YUUKA_DTYPE_IQ2_XXS), uint32_t(YUUKA_DTYPE_IQ2_S),
                           uint32_t(YUUKA_DTYPE_IQ3_S)}) {
        for (int tile_j : {16, 32, 64, 128}) {
            if (check_case(ctx, dtype, tile_j) != 0) {
                teardown(ctx);
                return 1;
            }
        }
    }
    for (uint32_t down_dtype :
         {uint32_t(YUUKA_DTYPE_IQ2_S), uint32_t(YUUKA_DTYPE_IQ3_S)}) {
        if (check_direct_case(ctx, down_dtype) != 0) {
            teardown(ctx);
            return 1;
        }
    }
    teardown(ctx);
    std::printf("iq moe MMQ grouped expert route: pass\n");
    return 0;
}
