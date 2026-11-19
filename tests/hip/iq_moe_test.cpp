// Pins the device IQ2_XXS, IQ2_S, and IQ3_S block decoding and the selected and
// grouped MoE expert pipelines against the host reference.
//
// The three packed blocks are the first block of each type from
// Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf, the same fixtures that
// tensor/quant_test.go:TestDequantizeIQReference pins against ggml. The MoE
// pipeline then dequantizes the same packed expert matrices with
// yuuka_dequant_f16 and compares the grouped and selected kernels with a host
// computation over those decoded values.

#include "yuuka_hip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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

// variant_block copies one 256-value block and replaces the low byte of its
// FP16 scale so every row of a test matrix decodes different values while the
// host reference and the kernel read the same bytes.
std::vector<uint8_t> variant_block(const std::vector<uint8_t> & block, int row) {
    std::vector<uint8_t> out = block;
    out[0] = uint8_t((out[0] + 7 * row) & 0xFF);
    return out;
}

// repeat_row appends count copies of one block, which is one packed row.
void append_block(std::vector<uint8_t> & packed, const std::vector<uint8_t> & block,
                  int row) {
    const std::vector<uint8_t> variant = variant_block(block, row);
    packed.insert(packed.end(), variant.begin(), variant.end());
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

float dot(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

struct moe_buffers {
    yuuka_buffer * norm = nullptr;
    yuuka_buffer * ids = nullptr;
    yuuka_buffer * weights = nullptr;
    yuuka_buffer * counts = nullptr;
    yuuka_buffer * assign_rows = nullptr;
    yuuka_buffer * assign_slots = nullptr;
    yuuka_buffer * assign_weights = nullptr;
    yuuka_buffer * gate = nullptr;
    yuuka_buffer * up = nullptr;
    yuuka_buffer * down = nullptr;
    yuuka_buffer * gate_out = nullptr;
    yuuka_buffer * up_out = nullptr;
    yuuka_buffer * slot_act = nullptr;
    yuuka_buffer * expert_out = nullptr;
    yuuka_buffer * x = nullptr;
    yuuka_buffer * logits = nullptr;
};

bool alloc_buffers(context & ctx, moe_buffers & b, int rows, int experts, int used,
                   int hidden, int ffn, const std::vector<uint8_t> & gate_packed,
                   const std::vector<uint8_t> & up_packed,
                   const std::vector<uint8_t> & down_packed,
                   const std::vector<float> & norm, const std::vector<float> & logits) {
    const int slots = rows * used;
    b.norm = upload(ctx, norm.data(), norm.size() * 4);
    b.logits = upload(ctx, logits.data(), logits.size() * 4);
    b.ids = upload(ctx, nullptr, slots * 4);
    b.weights = upload(ctx, nullptr, slots * 4);
    b.counts = upload(ctx, nullptr, experts * 4);
    b.assign_rows = upload(ctx, nullptr, experts * slots * 4);
    b.assign_slots = upload(ctx, nullptr, experts * slots * 4);
    b.assign_weights = upload(ctx, nullptr, experts * slots * 4);
    b.gate = upload(ctx, gate_packed.data(), gate_packed.size());
    b.up = upload(ctx, up_packed.data(), up_packed.size());
    b.down = upload(ctx, down_packed.data(), down_packed.size());
    b.gate_out = upload(ctx, nullptr, slots * ffn * 4);
    b.up_out = upload(ctx, nullptr, slots * ffn * 4);
    b.slot_act = upload(ctx, nullptr, slots * ffn * 4);
    b.expert_out = upload(ctx, nullptr, slots * hidden * 4);
    b.x = upload(ctx, nullptr, rows * hidden * 4);
    return b.norm != nullptr && b.logits != nullptr && b.ids != nullptr &&
           b.weights != nullptr && b.counts != nullptr && b.assign_rows != nullptr &&
           b.assign_slots != nullptr && b.assign_weights != nullptr &&
           b.gate != nullptr && b.up != nullptr && b.down != nullptr &&
           b.gate_out != nullptr && b.up_out != nullptr && b.slot_act != nullptr &&
           b.expert_out != nullptr && b.x != nullptr;
}

void free_buffers(moe_buffers & b) {
    yuuka_buffer * all[] = {b.norm, b.ids, b.weights, b.counts, b.assign_rows,
                            b.assign_slots, b.assign_weights, b.gate, b.up, b.down,
                            b.gate_out, b.up_out, b.slot_act, b.expert_out, b.x,
                            b.logits};
    for (yuuka_buffer * buffer : all) {
        if (buffer != nullptr) yuuka_buffer_free(buffer);
    }
    b = moe_buffers{};
}

yuuka_status add_route(yuuka_plan * plan, moe_buffers & b, int rows, int experts,
                       int used) {
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_ROUTE;
    op.buf[0] = b.logits;
    op.buf[1] = b.ids;
    op.buf[2] = b.weights;
    op.buf[3] = b.counts;
    op.buf[4] = b.assign_rows;
    op.buf[5] = b.assign_slots;
    op.buf[6] = b.assign_weights;
    op.n[0] = rows;
    op.n[1] = experts;
    op.n[2] = used;
    return yuuka_plan_add(plan, &op);
}

yuuka_status add_reduce(yuuka_plan * plan, moe_buffers & b, int hidden, int rows,
                        int used) {
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = b.x;
    op.buf[1] = b.expert_out;
    op.buf[2] = b.weights;
    op.n[0] = hidden;
    op.n[1] = rows;
    op.n[2] = used;
    return yuuka_plan_add(plan, &op);
}

yuuka_status add_grouped(yuuka_plan * plan, moe_buffers & b, int hidden, int ffn,
                         int rows, int experts, int used, uint32_t gate_dtype,
                         uint32_t down_dtype) {
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_GATE_UP;
    op.buf[0] = b.norm;
    op.buf[1] = b.gate;
    op.buf[2] = b.up;
    op.buf[3] = b.counts;
    op.buf[4] = b.assign_rows;
    op.buf[5] = b.assign_slots;
    op.buf[6] = b.gate_out;
    op.buf[7] = b.up_out;
    op.n[0] = hidden;
    op.n[1] = ffn;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = gate_dtype;
    yuuka_status status = yuuka_plan_add(plan, &op);
    if (status != YUUKA_OK) return status;

    op = yuuka_op{};
    op.code = YUUKA_OP_SILU_MUL;
    op.buf[0] = b.gate_out;
    op.buf[1] = b.up_out;
    op.buf[2] = b.slot_act;
    op.n[0] = rows * used * ffn;
    status = yuuka_plan_add(plan, &op);
    if (status != YUUKA_OK) return status;

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_DOWN;
    op.buf[0] = b.slot_act;
    op.buf[1] = b.down;
    op.buf[2] = b.counts;
    op.buf[3] = b.assign_rows;
    op.buf[4] = b.assign_slots;
    op.buf[5] = b.expert_out;
    op.n[0] = ffn;
    op.n[1] = hidden;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = down_dtype;
    return yuuka_plan_add(plan, &op);
}

yuuka_status add_selected(yuuka_plan * plan, moe_buffers & b, int hidden, int ffn,
                          int rows, int experts, int used, uint32_t gate_dtype,
                          uint32_t down_dtype) {
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_GATE_UP_SELECTED;
    op.buf[0] = b.norm;
    op.buf[1] = b.ids;
    op.buf[2] = b.gate;
    op.buf[3] = b.up;
    op.buf[4] = b.slot_act;
    op.n[0] = hidden;
    op.n[1] = ffn;
    op.n[2] = rows;
    op.n[3] = used;
    op.n[4] = experts;
    op.u[0] = gate_dtype;
    yuuka_status status = yuuka_plan_add(plan, &op);
    if (status != YUUKA_OK) return status;

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_DOWN_SELECTED;
    op.buf[0] = b.slot_act;
    op.buf[1] = b.ids;
    op.buf[2] = b.down;
    op.buf[3] = b.expert_out;
    op.n[0] = ffn;
    op.n[1] = hidden;
    op.n[2] = rows;
    op.n[3] = used;
    op.n[4] = experts;
    op.u[0] = down_dtype;
    return yuuka_plan_add(plan, &op);
}

// run_pipeline executes one MoE plan over a zero residual row and returns the
// result plus the routing the device selected.
int run_pipeline(context & ctx, moe_buffers & b, bool selected, int hidden, int ffn,
                 int rows, int experts, int used, uint32_t gate_dtype,
                 uint32_t down_dtype, std::vector<float> & out,
                 std::vector<int32_t> & ids, std::vector<float> & weights) {
    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_status status = add_route(plan, b, rows, experts, used);
    if (status == YUUKA_OK) {
        status = selected ? add_selected(plan, b, hidden, ffn, rows, experts, used,
                                         gate_dtype, down_dtype)
                          : add_grouped(plan, b, hidden, ffn, rows, experts, used,
                                        gate_dtype, down_dtype);
    }
    if (status == YUUKA_OK) {
        status = add_reduce(plan, b, hidden, rows, used);
    }
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "plan build failed: %s\n", yuuka_last_error());
        yuuka_plan_destroy(plan);
        return 1;
    }
    std::vector<float> zero(rows * hidden, 0.0f);
    CHECK_STATUS(yuuka_memcpy_h2d(b.x, zero.data(), zero.size() * 4));
    status = yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event);
    if (status == YUUKA_OK) status = yuuka_event_wait(ctx.event);
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "plan execute failed: %s\n", yuuka_last_error());
        yuuka_plan_destroy(plan);
        return 1;
    }
    yuuka_plan_destroy(plan);
    out.resize(rows * hidden);
    ids.resize(rows * used);
    weights.resize(rows * used);
    CHECK_STATUS(yuuka_memcpy_d2h(out.data(), b.x, out.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(ids.data(), b.ids, ids.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(weights.data(), b.weights, weights.size() * 4));
    return 0;
}

} // namespace

int main() {
    context ctx;
    if (setup(ctx) != 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }

    // The three reference blocks and the first eight decoded values plus the
    // block sum are the fixtures pinned by the CPU dequantization test.
    struct block_case {
        const char * name;
        uint32_t dtype;
        const char * hex;
        float first[8];
        float sum;
    };
    const block_case blocks[] = {
        {"IQ2_XXS", YUUKA_DTYPE_IQ2_XXS,
         "2f0c070067cad0a4b8adba52aa0456f20f7b013a0c004f7332f58f6651311d43d68b23eba5d2"
         "ece74da21c8c5623474cc08e59d71e0d6377b69c22ec15786a8338bb",
         {0.005362272262573242f, 0.005362272262573242f, 0.028822213411331177f,
          0.005362272262573242f, -0.005362272262573242f, 0.005362272262573242f,
          -0.005362272262573242f, 0.005362272262573242f},
         -0.09559527039527893f},
        {"IQ3_S", YUUKA_DTYPE_IQ3_S,
         "76051fd85942b4e46e38b10d380fbc2ca7c5c61373867fed8a59c099a3df55e617fe8e574dd8"
         "282e7e8d3f9c002844e737e27987199ddc6f21c6d1f4a1213f94106899a002a7dd78845721b9"
         "846cec02ee3f5e044832ff7cabb771788225979696dd19b7214cf9b6a9638b8a999f",
         {-0.0057495832443237305f, 0.013415694236755371f, 0.0019165277481079102f,
          0.013415694236755371f, 0.0019165277481079102f, -0.028747916221618652f,
          0.0019165277481079102f, 0.00958263874053955f},
         -0.0661618709564209f},
        {"IQ2_S", YUUKA_DTYPE_IQ2_S,
         "57099f38722503000d3a00c3f51de39e4c4a0e193e06d87f1b3f001d4f7f4900b602489e7a4b"
         "8a3a5ea690784a9e256b50e3007be615268e7d57c371164b0f34398a00c024c1000d70b0999b"
         "a959fb9c8ee7",
         {0.016642197966575623f, 0.0030962228775024414f, 0.0030962228775024414f,
          -0.0030962228775024414f, 0.0030962228775024414f, 0.016642197966575623f,
          -0.0030962228775024414f, 0.0030962228775024414f},
         -0.14647985994815826f},
    };
    std::vector<uint8_t> ref_blocks[3];
    for (int i = 0; i < 3; ++i) {
        ref_blocks[i] = from_hex(blocks[i].hex);
        CHECK(ref_blocks[i].size() == (i == 0 ? 66u : (i == 1 ? 110u : 82u)));
        const std::vector<float> got =
            dequant(ctx, ref_blocks[i], 256, 1, blocks[i].dtype);
        CHECK(!got.empty());
        float first_diff = 0.0f;
        for (int j = 0; j < 8; ++j) {
            first_diff = std::fmax(first_diff, std::fabs(got[j] - blocks[i].first[j]));
        }
        float sum = 0.0f;
        for (float value : got) sum += value;
        const float sum_diff = std::fabs(sum - blocks[i].sum);
        std::printf("dequant %s: first8 max abs diff %.3g, block sum diff %.3g\n",
                    blocks[i].name, first_diff, sum_diff);
        CHECK(first_diff <= 2e-5f);
        CHECK(sum_diff <= 5e-3f);
    }

    // The MoE pipeline needs 256-value blocks in every expert row, so the
    // expert FFN width and the hidden width are both one block.
    const int hidden = 256, ffn = 256, experts = 2, used = 1, rows = 1;
    const int slots = rows * used;
    std::mt19937 rng(20260306);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> norm(hidden), logits(experts);
    for (float & value : norm) value = dist(rng);
    logits[0] = 0.5f;
    logits[1] = 0.1f;

    std::vector<uint8_t> gate_packed, up_packed;
    for (int row = 0; row < experts * ffn; ++row) {
        append_block(gate_packed, ref_blocks[0], row);
        append_block(up_packed, ref_blocks[0], row + 1);
    }
    const std::vector<float> gate_host =
        dequant(ctx, gate_packed, hidden, uint64_t(experts) * ffn, YUUKA_DTYPE_IQ2_XXS);
    const std::vector<float> up_host =
        dequant(ctx, up_packed, hidden, uint64_t(experts) * ffn, YUUKA_DTYPE_IQ2_XXS);
    CHECK(gate_host.size() == uint64_t(hidden) * experts * ffn);
    CHECK(up_host.size() == uint64_t(hidden) * experts * ffn);

    const uint32_t down_dtypes[2] = {YUUKA_DTYPE_IQ2_S, YUUKA_DTYPE_IQ3_S};
    const std::vector<uint8_t> & down_block_iq2s = ref_blocks[2];
    const std::vector<uint8_t> & down_block_iq3s = ref_blocks[1];
    for (int variant = 0; variant < 2; ++variant) {
        const uint32_t down_dtype = down_dtypes[variant];
        const std::vector<uint8_t> & down_block =
            down_dtype == YUUKA_DTYPE_IQ3_S ? down_block_iq3s : down_block_iq2s;
        std::vector<uint8_t> down_packed;
        for (int row = 0; row < experts * hidden; ++row) {
            append_block(down_packed, down_block, row);
        }
        const std::vector<float> down_host = dequant(
            ctx, down_packed, ffn, uint64_t(experts) * hidden, down_dtype);
        CHECK(!down_host.empty());

        for (int selected = 0; selected < 2; ++selected) {
            moe_buffers b;
            CHECK(alloc_buffers(ctx, b, rows, experts, used, hidden, ffn, gate_packed,
                                up_packed, down_packed, norm, logits));
            std::vector<float> got, weights;
            std::vector<int32_t> ids;
            if (run_pipeline(ctx, b, selected != 0, hidden, ffn, rows, experts, used,
                             YUUKA_DTYPE_IQ2_XXS, down_dtype, got, ids, weights) != 0) {
                free_buffers(b);
                teardown(ctx);
                return 1;
            }
            free_buffers(b);
            CHECK(ids.size() == size_t(slots));
            CHECK(ids[0] == 0);

            // Host reference over the decoded weights and the device routing.
            std::vector<float> want(rows * hidden, 0.0f);
            for (int row = 0; row < rows; ++row) {
                for (int slot = 0; slot < used; ++slot) {
                    const int expert = ids[row * used + slot];
                    CHECK(expert >= 0 && expert < experts);
                    std::vector<float> act(ffn);
                    for (int j = 0; j < ffn; ++j) {
                        const float * gate_row =
                            &gate_host[(uint64_t(expert) * ffn + j) * hidden];
                        const float * up_row =
                            &up_host[(uint64_t(expert) * ffn + j) * hidden];
                        const float * x = &norm[uint64_t(row) * hidden];
                        act[j] = silu(dot(gate_row, x, hidden)) *
                                 dot(up_row, x, hidden);
                    }
                    for (int i = 0; i < hidden; ++i) {
                        const float * down_row =
                            &down_host[(uint64_t(expert) * hidden + i) * ffn];
                        want[uint64_t(row) * hidden + i] +=
                            weights[row * used + slot] * dot(down_row, act.data(), ffn);
                    }
                }
            }
            float worst = 0.0f;
            for (size_t i = 0; i < got.size(); ++i) {
                worst = std::fmax(worst, std::fabs(got[i] - want[i]));
            }
            std::printf("moe %s path, down dtype %u: max abs diff %.3g\n",
                        selected != 0 ? "selected" : "grouped", down_dtype, worst);
            CHECK(worst <= 1e-2f);
        }
    }

    // Shared expert: x += sigmoid(dot(gate, norm[row])) * src[row].
    {
        const int rows2 = 2;
        std::vector<float> x(rows2 * hidden), src(rows2 * hidden), gate(hidden);
        std::vector<float> norm2(rows2 * hidden);
        for (float & value : x) value = dist(rng);
        for (float & value : src) value = dist(rng);
        for (float & value : gate) value = dist(rng);
        for (float & value : norm2) value = dist(rng);
        std::vector<float> want = x;
        for (int row = 0; row < rows2; ++row) {
            const float weight =
                1.0f / (1.0f + std::exp(-dot(gate.data(), &norm2[row * hidden], hidden)));
            for (int i = 0; i < hidden; ++i) {
                const uint64_t index = uint64_t(row) * hidden + i;
                want[index] += weight * src[index];
            }
        }
        yuuka_buffer * x_buf = upload(ctx, x.data(), x.size() * 4);
        yuuka_buffer * src_buf = upload(ctx, src.data(), src.size() * 4);
        yuuka_buffer * gate_buf = upload(ctx, gate.data(), gate.size() * 4);
        yuuka_buffer * norm_buf = upload(ctx, norm2.data(), norm2.size() * 4);
        CHECK(x_buf && src_buf && gate_buf && norm_buf);
        yuuka_plan * plan = nullptr;
        CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
        yuuka_op op{};
        op.code = YUUKA_OP_MOE_SHARED_ADD;
        op.buf[0] = x_buf;
        op.buf[1] = src_buf;
        op.buf[2] = gate_buf;
        op.buf[3] = norm_buf;
        op.n[0] = hidden;
        op.n[1] = rows2;
        CHECK_STATUS(yuuka_plan_add(plan, &op));
        CHECK_STATUS(yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event));
        CHECK_STATUS(yuuka_event_wait(ctx.event));
        yuuka_plan_destroy(plan);
        std::vector<float> got(x.size());
        CHECK_STATUS(yuuka_memcpy_d2h(got.data(), x_buf, got.size() * 4));
        yuuka_buffer_free(x_buf);
        yuuka_buffer_free(src_buf);
        yuuka_buffer_free(gate_buf);
        yuuka_buffer_free(norm_buf);
        float worst = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            worst = std::fmax(worst, std::fabs(got[i] - want[i]));
        }
        std::printf("moe shared add: max abs diff %.3g\n", worst);
        CHECK(worst <= 1e-5f);
    }

    teardown(ctx);
    std::printf("iq moe correctness: pass\n");
    return 0;
}
