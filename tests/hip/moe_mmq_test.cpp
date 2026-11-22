// Pins the grouped int8 WMMA (MMQ) expert kernels against a host reference over
// the same decoded weights and the same Q8_1 activation values.
//
// The route stages K-quant expert rows through the shared MMQ tile staging and
// dots them with 16x16x16 int8 WMMA over an assignment tile of tile_j rows, so
// the test covers the tile widths the plan can select (16, 32, 64, 128), a
// bucket that does not fill its tile, and the Q4_K and Q6_K scale layouts: one
// weight scale per 32 values with a per-row minimum for Q4_K, and one per 16
// values without a minimum for Q6_K. The reference dequantizes the same packed
// matrices with an independent host decoder and reproduces the device Q8_1
// quantizer, so a decode or scale-slot error cannot hide behind the quantized
// comparison.

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

void put_half(uint8_t * dst, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = int32_t((bits >> 23) & 0xFF) - 127;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    uint16_t half = 0;
    if (exponent > 15) {
        half = uint16_t(sign | 0x7C00u);
    } else if (exponent < -24) {
        half = uint16_t(sign);
    } else if (exponent < -14) {
        const uint32_t shift = uint32_t(-exponent - 14);
        const uint32_t full = mantissa | 0x800000u;
        half = uint16_t(sign | ((full + (1u << (shift + 12))) >> (shift + 13)));
    } else {
        uint32_t rounded = mantissa + 0x1000u;
        if (rounded & 0x800000u) {
            rounded = 0;
            ++exponent;
        }
        half = exponent > 15 ? uint16_t(sign | 0x7C00u)
                             : uint16_t(sign | (uint32_t(exponent + 15) << 10) |
                                        (rounded >> 13));
    }
    *reinterpret_cast<uint16_t *>(dst) = half;
}

uint64_t block_bytes(uint32_t dtype) {
    return dtype == YUUKA_DTYPE_Q4_K ? 144u
         : dtype == YUUKA_DTYPE_Q6_K ? 210u
                                     : 0u;
}

// fill_packed writes a synthetic but well-formed packed matrix. Every block
// carries a finite FP16 scale so the decode stays in range; the quantized
// payload is random because the kernel and the host reference consume the same
// bytes.
void fill_packed(std::vector<uint8_t> & data, uint32_t dtype, std::mt19937 & rng) {
    const uint64_t bytes = block_bytes(dtype);
    for (uint64_t off = 0; off + bytes <= data.size(); off += bytes) {
        uint8_t * blk = data.data() + off;
        if (dtype == YUUKA_DTYPE_Q4_K) {
            put_half(blk, 0.002f);
            put_half(blk + 2, 0.001f);
            for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() & 0xFF);
            for (int i = 0; i < 128; ++i) blk[16 + i] = uint8_t(rng() & 0xFF);
            continue;
        }
        for (int i = 0; i < 192; ++i) blk[i] = uint8_t(rng() & 0xFF);
        for (int i = 0; i < 16; ++i) blk[192 + i] = uint8_t(rng() & 0x0F);
        put_half(blk + 208, 0.002f);
    }
}

// get_scale_min_k4 restates the Q4_K scale/min nibble layout of the format so
// the host reference reads the packed bytes without the device decoder.
void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = uint8_t(q[j] & 63);
        m = uint8_t(q[j + 4] & 63);
        return;
    }
    d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// dequant_ref_value decodes one element of a packed Q4_K or Q6_K row. It is an
// independent statement of the format arithmetic, so a nibble, scale-slot, or
// minimum error in the kernel cannot hide behind the tolerance.
float dequant_ref_value(uint32_t dtype, const uint8_t * row, int64_t index) {
    if (dtype == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (index / 256) * 144;
        const float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
        const float dm = half_to_float(uint16_t(blk[2] | (blk[3] << 8)));
        const int sub = int((index % 256) / 32);
        const int within = int(index & 31);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const uint8_t qs = blk[16 + (sub / 2) * 32 + within];
        const int q = (sub & 1) ? (qs >> 4) : (qs & 0xF);
        return d * float(sc) * float(q) - dm * float(m);
    }
    const uint8_t * blk = row + (index / 256) * 210;
    const int n = int((index % 256) / 128);
    const int l = int(index & 127);
    const uint8_t * ql = blk + n * 64;
    const uint8_t * qh = blk + 128 + n * 32;
    const int8_t * scales = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = half_to_float(uint16_t(blk[208] | (blk[209] << 8)));
    int q = 0;
    int sidx = 0;
    if (l < 32) {
        q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        sidx = l / 16;
    } else if (l < 64) {
        q = (ql[l] & 0xF) | (((qh[l - 32] >> 2) & 3) << 4);
        sidx = (l - 32) / 16 + 2;
    } else if (l < 96) {
        q = (ql[l - 64] >> 4) | (((qh[l - 64] >> 4) & 3) << 4);
        sidx = (l - 64) / 16 + 4;
    } else {
        q = (ql[l - 64] >> 4) | (((qh[l - 96] >> 6) & 3) << 4);
        sidx = (l - 96) / 16 + 6;
    }
    return d * float(scales[n * 8 + sidx]) * float(q - 32);
}

// dequant_host decodes a whole packed matrix to FP32 with the host reference.
std::vector<float> dequant_host(const std::vector<uint8_t> & packed, uint32_t dtype,
                                int n_in, int n_out) {
    const uint64_t row_bytes = uint64_t(n_in / 256) * block_bytes(dtype);
    std::vector<float> out(size_t(n_in) * n_out);
    for (int r = 0; r < n_out; ++r) {
        const uint8_t * row = packed.data() + uint64_t(r) * row_bytes;
        for (int i = 0; i < n_in; ++i) {
            out[size_t(r) * n_in + i] = dequant_ref_value(dtype, row, i);
        }
    }
    return out;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// quantize_q8_1_host reproduces the device Q8_1 quantizer for a matrix of whole
// 32-value blocks: the scale is amax/127 and each value becomes digit * scale.
// That is the value both the WMMA digits and the minimum term of Q4_K see.
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

struct moe_buffers {
    yuuka_buffer * norm = nullptr;
    yuuka_buffer * logits = nullptr;
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
    yuuka_buffer * expert_out = nullptr;
    yuuka_buffer * x = nullptr;
    yuuka_buffer * norm_q8 = nullptr;
    yuuka_buffer * norm_ds = nullptr;
    yuuka_buffer * act_q8 = nullptr;
    yuuka_buffer * act_ds = nullptr;
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
    b.assign_rows = upload(ctx, nullptr, uint64_t(experts) * slots * 4);
    b.assign_slots = upload(ctx, nullptr, uint64_t(experts) * slots * 4);
    b.assign_weights = upload(ctx, nullptr, uint64_t(experts) * slots * 4);
    b.gate = upload(ctx, gate_packed.data(), gate_packed.size());
    b.up = upload(ctx, up_packed.data(), up_packed.size());
    b.down = upload(ctx, down_packed.data(), down_packed.size());
    b.gate_out = upload(ctx, nullptr, uint64_t(slots) * ffn * 4);
    b.up_out = upload(ctx, nullptr, uint64_t(slots) * ffn * 4);
    b.expert_out = upload(ctx, nullptr, uint64_t(slots) * hidden * 4);
    b.x = upload(ctx, nullptr, uint64_t(rows) * hidden * 4);
    b.norm_q8 = upload(ctx, nullptr, uint64_t(rows) * hidden);
    b.norm_ds = upload(ctx, nullptr, uint64_t(rows) * (hidden / 32) * 8);
    b.act_q8 = upload(ctx, nullptr, uint64_t(slots) * ffn);
    b.act_ds = upload(ctx, nullptr, uint64_t(slots) * (ffn / 32) * 8);
    return b.norm != nullptr && b.logits != nullptr && b.ids != nullptr &&
           b.weights != nullptr && b.counts != nullptr && b.assign_rows != nullptr &&
           b.assign_slots != nullptr && b.assign_weights != nullptr &&
           b.gate != nullptr && b.up != nullptr && b.down != nullptr &&
           b.gate_out != nullptr && b.up_out != nullptr && b.expert_out != nullptr &&
           b.x != nullptr && b.norm_q8 != nullptr && b.norm_ds != nullptr &&
           b.act_q8 != nullptr && b.act_ds != nullptr;
}

void free_buffers(moe_buffers & b) {
    yuuka_buffer * all[] = {b.norm, b.logits, b.ids, b.weights, b.counts,
                            b.assign_rows, b.assign_slots, b.assign_weights, b.gate,
                            b.up, b.down, b.gate_out, b.up_out, b.expert_out, b.x,
                            b.norm_q8, b.norm_ds, b.act_q8, b.act_ds};
    for (yuuka_buffer * buffer : all) {
        if (buffer != nullptr) yuuka_buffer_free(buffer);
    }
    b = moe_buffers{};
}

yuuka_status add_op(yuuka_plan * plan, yuuka_op & op) {
    const yuuka_status status = yuuka_plan_add(plan, &op);
    op = yuuka_op{};
    return status;
}

// build_mmq_plan appends the routed-expert int8 WMMA pipeline: routing, the
// Q8_1 activation pass, both expert projections, the quantized SwiGLU product,
// the down projection, and the weighted reduction.
yuuka_status build_mmq_plan(yuuka_plan * plan, moe_buffers & b, int hidden, int ffn,
                            int rows, int experts, int used, int tile_j,
                            uint32_t dtype) {
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
    yuuka_status status = add_op(plan, op);
    if (status != YUUKA_OK) return status;

    op.code = YUUKA_OP_QUANT_Q8_1;
    op.buf[0] = b.norm;
    op.buf[1] = b.norm_q8;
    op.buf[2] = b.norm_ds;
    op.n[0] = hidden;
    op.n[1] = rows;
    status = add_op(plan, op);
    if (status != YUUKA_OK) return status;

    op.code = YUUKA_OP_MOE_GATE_UP_MMQ;
    op.buf[0] = b.norm_q8;
    op.buf[1] = b.norm_ds;
    op.buf[2] = b.gate;
    op.buf[3] = b.up;
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
    op.n[5] = tile_j;
    op.u[0] = dtype;
    status = add_op(plan, op);
    if (status != YUUKA_OK) return status;

    op.code = YUUKA_OP_SILU_MUL_QUANT_Q8_1;
    op.buf[0] = b.gate_out;
    op.buf[1] = b.up_out;
    op.buf[2] = b.act_q8;
    op.buf[3] = b.act_ds;
    op.n[0] = ffn;
    op.n[1] = rows * used;
    status = add_op(plan, op);
    if (status != YUUKA_OK) return status;

    op.code = YUUKA_OP_MOE_DOWN_MMQ;
    op.buf[0] = b.act_q8;
    op.buf[1] = b.act_ds;
    op.buf[2] = b.down;
    op.buf[3] = b.counts;
    op.buf[4] = b.assign_rows;
    op.buf[5] = b.assign_slots;
    op.buf[6] = b.expert_out;
    op.n[0] = ffn;
    op.n[1] = hidden;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.n[5] = tile_j;
    op.u[0] = dtype;
    status = add_op(plan, op);
    if (status != YUUKA_OK) return status;

    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = b.x;
    op.buf[1] = b.expert_out;
    op.buf[2] = b.weights;
    op.n[0] = hidden;
    op.n[1] = rows;
    op.n[2] = used;
    return add_op(plan, op);
}

// run_mmq executes the plan over a zero residual row.
int run_mmq(context & ctx, moe_buffers & b, int hidden, int ffn, int rows,
            int experts, int used, int tile_j, uint32_t dtype,
            std::vector<float> & out) {
    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_status status =
        build_mmq_plan(plan, b, hidden, ffn, rows, experts, used, tile_j, dtype);
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "plan build failed: %s\n", yuuka_last_error());
        yuuka_plan_destroy(plan);
        return 1;
    }
    std::vector<float> zero(size_t(rows) * hidden, 0.0f);
    CHECK_STATUS(yuuka_memcpy_h2d(b.x, zero.data(), zero.size() * 4));
    status = yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event);
    if (status == YUUKA_OK) status = yuuka_event_wait(ctx.event);
    yuuka_plan_destroy(plan);
    if (status != YUUKA_OK) {
        std::fprintf(stderr, "plan execute failed: %s\n", yuuka_last_error());
        return 1;
    }
    out.resize(size_t(rows) * hidden);
    CHECK_STATUS(yuuka_memcpy_d2h(out.data(), b.x, out.size() * 4));
    return 0;
}

// gate_up_reference recomputes both expert projections from the decoded weights
// and the host reproduction of the device Q8_1 activation tile.
void gate_up_reference(const std::vector<float> & gate_host,
                       const std::vector<float> & up_host,
                       const std::vector<float> & norm,
                       const std::vector<int32_t> & ids, int rows, int used,
                       int hidden, int ffn, std::vector<float> & gate,
                       std::vector<float> & up) {
    const std::vector<float> xq = quantize_q8_1_host(norm);
    const int slots = rows * used;
    gate.assign(size_t(slots) * ffn, 0.0f);
    up.assign(size_t(slots) * ffn, 0.0f);
    for (int slot = 0; slot < slots; ++slot) {
        const int row = slot / used;
        const int expert = ids[slot];
        for (int j = 0; j < ffn; ++j) {
            const float * gate_row = gate_host.data() +
                (size_t(expert) * ffn + j) * hidden;
            const float * up_row = up_host.data() +
                (size_t(expert) * ffn + j) * hidden;
            float g = 0.0f;
            float u = 0.0f;
            for (int i = 0; i < hidden; ++i) {
                const float x = xq[size_t(row) * hidden + i];
                g += x * gate_row[i];
                u += x * up_row[i];
            }
            gate[size_t(slot) * ffn + j] = g;
            up[size_t(slot) * ffn + j] = u;
        }
    }
}

// down_reference recomputes the down projection from the decoded weights and the
// device SwiGLU Q8_1 tile, so the stage is checked without the requantization
// noise of reproducing the SwiGLU product on the host.
std::vector<float> down_reference(const std::vector<float> & down_host,
                                  const std::vector<int8_t> & act_q8,
                                  const std::vector<float> & act_ds,
                                  const std::vector<int32_t> & ids,
                                  const std::vector<float> & weights, int rows,
                                  int used, int hidden, int ffn) {
    const int slots = rows * used;
    std::vector<float> out(size_t(rows) * hidden, 0.0f);
    for (int slot = 0; slot < slots; ++slot) {
        const int row = slot / used;
        const int expert = ids[slot];
        const int8_t * digits = act_q8.data() + size_t(slot) * ffn;
        const float * scales = act_ds.data() + size_t(slot) * (ffn / 32) * 2;
        for (int r = 0; r < hidden; ++r) {
            const float * down_row = down_host.data() +
                (size_t(expert) * hidden + r) * ffn;
            float value = 0.0f;
            for (int j = 0; j < ffn; ++j) {
                value += float(digits[j]) * scales[(j / 32) * 2] * down_row[j];
            }
            out[size_t(row) * hidden + r] += weights[slot] * value;
        }
    }
    return out;
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

// check_case runs one (dtype, rows, tile_j) cell end to end.
int check_case(context & ctx, uint32_t dtype, int rows, int tile_j) {
    const int experts = 8, used = 2, hidden = 256, ffn = 256;
    const uint64_t bytes = block_bytes(dtype);
    std::mt19937 rng(1234u + uint32_t(dtype) * 17u + uint32_t(rows));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const uint64_t hidden_row = (hidden / 256) * bytes;
    const uint64_t ffn_row = (ffn / 256) * bytes;

    std::vector<uint8_t> gate_packed(size_t(experts) * ffn * hidden_row);
    std::vector<uint8_t> up_packed(gate_packed.size());
    std::vector<uint8_t> down_packed(size_t(experts) * hidden * ffn_row);
    fill_packed(gate_packed, dtype, rng);
    fill_packed(up_packed, dtype, rng);
    fill_packed(down_packed, dtype, rng);

    std::vector<float> norm(size_t(rows) * hidden);
    std::vector<float> logits(size_t(rows) * experts);
    for (float & v : norm) v = dist(rng);
    for (float & v : logits) v = dist(rng);

    const std::vector<float> gate_host = dequant_host(gate_packed, dtype, hidden,
                                                      experts * ffn);
    const std::vector<float> up_host = dequant_host(up_packed, dtype, hidden,
                                                    experts * ffn);
    const std::vector<float> down_host = dequant_host(down_packed, dtype, ffn,
                                                      experts * hidden);

    moe_buffers b;
    if (!alloc_buffers(ctx, b, rows, experts, used, hidden, ffn, gate_packed,
                       up_packed, down_packed, norm, logits)) {
        std::fprintf(stderr, "buffer allocation failed\n");
        free_buffers(b);
        return 1;
    }
    {
        // Zero the expert outputs so a slot the kernel never writes reads as
        // zero instead of uninitialized device memory.
        std::vector<float> zero_out(size_t(rows) * used * ffn, 0.0f);
        CHECK_STATUS(yuuka_memcpy_h2d(b.gate_out, zero_out.data(), zero_out.size() * 4));
        CHECK_STATUS(yuuka_memcpy_h2d(b.up_out, zero_out.data(), zero_out.size() * 4));
    }
    std::vector<float> got;
    if (run_mmq(ctx, b, hidden, ffn, rows, experts, used, tile_j, dtype, got) != 0) {
        free_buffers(b);
        return 1;
    }
    const int slots = rows * used;
    std::vector<int32_t> ids(static_cast<size_t>(slots));
    std::vector<float> weights(static_cast<size_t>(slots));
    std::vector<float> gate_got(size_t(slots) * ffn);
    std::vector<float> up_got(size_t(slots) * ffn);
    std::vector<int8_t> act_q8(size_t(slots) * ffn);
    std::vector<float> act_ds(size_t(slots) * (ffn / 32) * 2);
    CHECK_STATUS(yuuka_memcpy_d2h(ids.data(), b.ids, ids.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(weights.data(), b.weights, weights.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(gate_got.data(), b.gate_out, gate_got.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(up_got.data(), b.up_out, up_got.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(act_q8.data(), b.act_q8, act_q8.size()));
    CHECK_STATUS(yuuka_memcpy_d2h(act_ds.data(), b.act_ds, act_ds.size() * 4));
    for (int slot = 0; slot < slots; ++slot) {
        CHECK(ids[slot] >= 0 && ids[slot] < experts);
    }
    std::vector<int32_t> counts(static_cast<size_t>(experts));
    CHECK_STATUS(yuuka_memcpy_d2h(counts.data(), b.counts, counts.size() * 4));
    int64_t count_sum = 0;
    for (int e = 0; e < experts; ++e) {
        CHECK(counts[e] >= 0 && counts[e] <= rows * used);
        count_sum += counts[e];
    }
    CHECK(count_sum == int64_t(rows) * used);
    free_buffers(b);

    // The gate and up projections decode the weight tile, apply the per-16 or
    // per-32 scale (and the Q4_K minimum), and gather the activation rows, so
    // they are compared against the host reference over the same Q8_1 values.
    std::vector<float> gate_ref;
    std::vector<float> up_ref;
    gate_up_reference(gate_host, up_host, norm, ids, rows, used, hidden, ffn,
                      gate_ref, up_ref);
    float gate_worst = 0.0f, gate_magnitude = 1.0f;
    float up_worst = 0.0f, up_magnitude = 1.0f;
    worst_diff(gate_got, gate_ref, gate_worst, gate_magnitude);
    worst_diff(up_got, up_ref, up_worst, up_magnitude);
    if (gate_worst > 1e-5f * gate_magnitude ||
        up_worst > 1e-5f * up_magnitude) {
        int shown = 0;
        for (int slot = 0; slot < slots && shown < 6; ++slot) {
            float dw = 0.0f;
            for (int j = 0; j < ffn; ++j) {
                dw = std::fmax(dw, std::fabs(gate_got[size_t(slot) * ffn + j] -
                                             gate_ref[size_t(slot) * ffn + j]));
            }
            if (dw > 1e-4f) {
                std::fprintf(stderr,
                             "  bad slot %d expert %d row %d gate_worst %.4g "
                             "first: got %.6g want %.6g up got %.6g want %.6g\n",
                             slot, ids[slot], slot / used, dw,
                             gate_got[size_t(slot) * ffn], gate_ref[size_t(slot) * ffn],
                             up_got[size_t(slot) * ffn], up_ref[size_t(slot) * ffn]);
                ++shown;
            }
        }
        std::fprintf(stderr,
                     "FAIL gate/up: dtype %u rows %d tile_j %d gate %.6g/%.6g "
                     "up %.6g/%.6g\n",
                     dtype, rows, tile_j, gate_worst, gate_magnitude, up_worst,
                     up_magnitude);
        return 1;
    }

    // The down projection is checked over the Q8_1 SwiGLU tile the device
    // produced, so the comparison pins the tile walk and the scatter without
    // the requantization step that sits between the two projections.
    const std::vector<float> out_ref = down_reference(down_host, act_q8, act_ds, ids,
                                                      weights, rows, used, hidden, ffn);
    float out_worst = 0.0f, out_magnitude = 1.0f;
    worst_diff(got, out_ref, out_worst, out_magnitude);
    CHECK(out_worst <= 1e-5f * out_magnitude);
    std::printf("  dtype %u rows %d tile_j %d: gate %.3g/%.3g up %.3g/%.3g "
                "down %.3g/%.3g (diff/scale)\n",
                dtype, rows, tile_j, gate_worst, gate_magnitude, up_worst,
                up_magnitude, out_worst, out_magnitude);
    return 0;
}

// check_rejects pins the plan validation: a dtype without a decoder in this tile
// layout, a K width that is not a whole number of 256-value blocks, a channel
// tile that is not a multiple of 128, and an unmeasured assignment tile are
// rejected instead of running another route. The IQ codebooks are accepted by
// the same op and are pinned end to end by tests/hip/iq_moe_mmq_test.cpp.
int check_rejects(context & ctx) {
    const int experts = 8, used = 2, hidden = 256, ffn = 256, rows = 16;
    const int slots = rows * used;
    const uint64_t hidden_row = (uint64_t(hidden) / 256) * 144;
    moe_buffers b;
    std::vector<uint8_t> gate(size_t(experts) * ffn * hidden_row, 0);
    std::vector<float> norm(size_t(rows) * hidden, 0.0f);
    std::vector<float> logits(size_t(rows) * experts, 0.0f);
    b.norm = upload(ctx, norm.data(), norm.size() * 4);
    b.norm_q8 = upload(ctx, nullptr, uint64_t(rows) * hidden);
    b.norm_ds = upload(ctx, nullptr, uint64_t(rows) * (hidden / 32) * 8);
    b.gate = upload(ctx, gate.data(), gate.size());
    b.up = upload(ctx, gate.data(), gate.size());
    b.counts = upload(ctx, nullptr, experts * 4);
    b.assign_rows = upload(ctx, nullptr, uint64_t(experts) * slots * 4);
    b.assign_slots = upload(ctx, nullptr, uint64_t(experts) * slots * 4);
    b.gate_out = upload(ctx, nullptr, uint64_t(slots) * ffn * 4);
    b.up_out = b.gate_out;
    CHECK(b.norm && b.norm_q8 && b.norm_ds && b.gate && b.up && b.counts &&
          b.assign_rows && b.assign_slots && b.gate_out);

    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_GATE_UP_MMQ;
    op.buf[0] = b.norm_q8;
    op.buf[1] = b.norm_ds;
    op.buf[2] = b.gate;
    op.buf[3] = b.up;
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
    op.n[5] = 16;
    op.u[0] = YUUKA_DTYPE_Q8_0;
    CHECK(yuuka_plan_add(plan, &op) != YUUKA_OK);
    op.u[0] = YUUKA_DTYPE_Q4_K;
    op.n[1] = 192;  // not a multiple of 128
    CHECK(yuuka_plan_add(plan, &op) != YUUKA_OK);
    op.n[1] = ffn;
    op.n[5] = 24;  // not a measured assignment tile
    CHECK(yuuka_plan_add(plan, &op) != YUUKA_OK);
    op.n[5] = 16;
    CHECK(yuuka_plan_add(plan, &op) == YUUKA_OK);
    op.u[0] = YUUKA_DTYPE_IQ2_S;
    CHECK(yuuka_plan_add(plan, &op) == YUUKA_OK);
    yuuka_plan_destroy(plan);

    yuuka_buffer_free(b.norm);
    yuuka_buffer_free(b.norm_q8);
    yuuka_buffer_free(b.norm_ds);
    yuuka_buffer_free(b.gate);
    yuuka_buffer_free(b.up);
    yuuka_buffer_free(b.counts);
    yuuka_buffer_free(b.assign_rows);
    yuuka_buffer_free(b.assign_slots);
    yuuka_buffer_free(b.gate_out);
    return 0;
}

} // namespace

int main() {
    context ctx;
    if (setup(ctx) != 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }

    std::printf("moe MMQ grouped expert route\n");
    if (check_rejects(ctx) != 0) {
        teardown(ctx);
        return 1;
    }
    // Two bucket shapes: 17 rows average five assignments per expert, so most
    // experts fill only part of a tile, and 64 rows average sixteen, which is
    // exactly one tile at the selected width.
    for (uint32_t dtype : {uint32_t(YUUKA_DTYPE_Q4_K), uint32_t(YUUKA_DTYPE_Q6_K)}) {
        for (int rows : {17, 64}) {
            for (int tile_j : {16, 32, 64, 128}) {
                if (check_case(ctx, dtype, rows, tile_j) != 0) {
                    teardown(ctx);
                    return 1;
                }
            }
        }
    }
    teardown(ctx);
    std::printf("moe MMQ grouped expert route: pass\n");
    return 0;
}
