// Measures the grouped MoE routing and expert pipeline at Qwen3-30B-A3B-like
// shapes. The weights are packed Q4_K and are never dequantized on the host.

#include "yuuka_hip.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

#define CHECK_STATUS(call)                                                               \
    do {                                                                                 \
        const yuuka_status st__ = (call);                                                \
        if (st__ != YUUKA_OK) {                                                          \
            std::fprintf(stderr, "%s -> %d (%s)\n", #call, int(st__),                    \
                         yuuka_last_error());                                            \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

void put_half(uint8_t * dst, float value) {
    const float clamped = value;
    uint32_t bits;
    std::memcpy(&bits, &clamped, 4);
    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = int32_t((bits >> 23) & 0xff) - 127;
    const uint32_t mant = bits & 0x7fffff;
    uint16_t half = 0;
    if (exp > 15) {
        half = uint16_t(sign | 0x7c00);
    } else if (exp < -24) {
        half = uint16_t(sign);
    } else if (exp < -14) {
        const uint32_t shift = uint32_t(-exp - 14);
        const uint32_t full = mant | 0x800000;
        half = uint16_t(sign | ((full + (1u << (shift + 12))) >> (shift + 13)));
    } else {
        uint32_t rounded = mant + 0x1000;
        int32_t e = exp;
        if (rounded & 0x800000) {
            rounded = 0;
            e++;
            if (e > 15) {
                half = uint16_t(sign | 0x7c00);
                *reinterpret_cast<uint16_t *>(dst) = half;
                return;
            }
        }
        half = uint16_t(sign | ((e + 15) << 10) | (rounded >> 13));
    }
    *reinterpret_cast<uint16_t *>(dst) = half;
}

uint64_t q4k_row_bytes(uint64_t width) { return (width / 256) * 144; }

void fill_q4k(uint8_t * data, uint64_t bytes, std::mt19937 & rng) {
    for (uint64_t off = 0; off + 144 <= bytes; off += 144) {
        uint8_t * blk = data + off;
        put_half(blk, 0.02f);
        put_half(blk + 2, 0.01f);
        for (int i = 0; i < 12; ++i) blk[4 + i] = uint8_t(rng() & 0xff);
        for (int i = 0; i < 128; ++i) blk[16 + i] = uint8_t(rng() & 0xff);
    }
}

yuuka_buffer * alloc(yuuka_device * device, uint64_t bytes) {
    yuuka_buffer * buffer = nullptr;
    if (yuuka_buffer_alloc(device, bytes, &buffer) != YUUKA_OK) {
        return nullptr;
    }
    return buffer;
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
    if (yuuka_stream_create(device, &stream) != YUUKA_OK ||
        yuuka_event_create(device, &event) != YUUKA_OK) {
        std::fprintf(stderr, "stream or event creation failed\n");
        return 1;
    }
    const int experts = 128, used = 8, hidden = 2048, ffn = 768;
    const uint64_t hidden_row = q4k_row_bytes(hidden);
    const uint64_t ffn_row = q4k_row_bytes(ffn);
    std::mt19937 rng(7);

    std::vector<uint8_t> gate(experts * ffn * hidden_row);
    std::vector<uint8_t> up(gate.size());
    std::vector<uint8_t> down(experts * hidden * ffn_row);
    fill_q4k(gate.data(), gate.size(), rng);
    fill_q4k(up.data(), up.size(), rng);
    fill_q4k(down.data(), down.size(), rng);

    yuuka_buffer * gate_buf = alloc(device, gate.size());
    yuuka_buffer * up_buf = alloc(device, up.size());
    yuuka_buffer * down_buf = alloc(device, down.size());
    if (gate_buf == nullptr || up_buf == nullptr || down_buf == nullptr) {
        std::fprintf(stderr, "weight allocation failed\n");
        return 1;
    }
    CHECK_STATUS(yuuka_memcpy_h2d(gate_buf, gate.data(), gate.size()));
    CHECK_STATUS(yuuka_memcpy_h2d(up_buf, up.data(), up.size()));
    CHECK_STATUS(yuuka_memcpy_h2d(down_buf, down.data(), down.size()));

    std::printf("device: %s\n", "gfx1151");
    std::printf("shape: experts=%d used=%d hidden=%d ffn=%d dtype=Q4_K\n",
                experts, used, hidden, ffn);
    std::printf("rows,route_ms,gate_up_ms,down_ms,reduce_ms,total_ms,tokens_per_s\n");

    for (int rows : {1, 2, 4, 8, 16, 32}) {
        const uint64_t bucket = uint64_t(experts) * rows * used * 4;
        yuuka_buffer * logits = alloc(device, uint64_t(rows) * experts * 4);
        yuuka_buffer * norm = alloc(device, uint64_t(rows) * hidden * 4);
        yuuka_buffer * ids = alloc(device, uint64_t(rows) * used * 4);
        yuuka_buffer * weights = alloc(device, uint64_t(rows) * used * 4);
        yuuka_buffer * counts = alloc(device, uint64_t(experts) * 4);
        yuuka_buffer * assign_rows = alloc(device, bucket);
        yuuka_buffer * assign_slots = alloc(device, bucket);
        yuuka_buffer * assign_weights = alloc(device, bucket);
        yuuka_buffer * gate_out = alloc(device, uint64_t(rows) * used * ffn * 4);
        yuuka_buffer * up_out = alloc(device, uint64_t(rows) * used * ffn * 4);
        yuuka_buffer * act_out = alloc(device, uint64_t(rows) * used * ffn * 4);
        yuuka_buffer * expert_out = alloc(device, uint64_t(rows) * used * hidden * 4);
        yuuka_buffer * x = alloc(device, uint64_t(rows) * hidden * 4);
        if (logits == nullptr || norm == nullptr || ids == nullptr || weights == nullptr ||
            counts == nullptr || assign_rows == nullptr || assign_slots == nullptr ||
            assign_weights == nullptr || gate_out == nullptr || up_out == nullptr ||
            act_out == nullptr || expert_out == nullptr || x == nullptr) {
            std::fprintf(stderr, "activation allocation failed\n");
            return 1;
        }
        std::vector<float> host_logits(uint64_t(rows) * experts);
        std::vector<float> host_norm(uint64_t(rows) * hidden);
        std::vector<float> host_x(uint64_t(rows) * hidden, 0.0f);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (float & v : host_logits) v = dist(rng);
        for (float & v : host_norm) v = dist(rng);
        CHECK_STATUS(yuuka_memcpy_h2d(logits, host_logits.data(), host_logits.size() * 4));
        CHECK_STATUS(yuuka_memcpy_h2d(norm, host_norm.data(), host_norm.size() * 4));
        CHECK_STATUS(yuuka_memcpy_h2d(x, host_x.data(), host_x.size() * 4));

        yuuka_plan * plan = nullptr;
        CHECK_STATUS(yuuka_plan_create(device, &plan));
        yuuka_op op{};
        op.code = YUUKA_OP_MOE_ROUTE;
        op.buf[0] = logits;
        op.buf[1] = ids;
        op.buf[2] = weights;
        op.buf[3] = counts;
        op.buf[4] = assign_rows;
        op.buf[5] = assign_slots;
        op.buf[6] = assign_weights;
        op.n[0] = rows;
        op.n[1] = experts;
        op.n[2] = used;
        CHECK_STATUS(yuuka_plan_add(plan, &op));

        op = yuuka_op{};
        op.code = YUUKA_OP_MOE_GATE_UP;
        op.buf[0] = norm;
        op.buf[1] = gate_buf;
        op.buf[2] = up_buf;
        op.buf[3] = counts;
        op.buf[4] = assign_rows;
        op.buf[5] = assign_slots;
        op.buf[6] = gate_out;
        op.buf[7] = up_out;
        op.n[0] = hidden;
        op.n[1] = ffn;
        op.n[2] = rows;
        op.n[3] = experts;
        op.n[4] = used;
        op.u[0] = YUUKA_DTYPE_Q4_K;
        CHECK_STATUS(yuuka_plan_add(plan, &op));

        op = yuuka_op{};
        op.code = YUUKA_OP_SILU_MUL;
        op.buf[0] = gate_out;
        op.buf[1] = up_out;
        op.buf[2] = act_out;
        op.n[0] = uint64_t(rows) * used * ffn;
        CHECK_STATUS(yuuka_plan_add(plan, &op));

        op = yuuka_op{};
        op.code = YUUKA_OP_MOE_DOWN;
        op.buf[0] = act_out;
        op.buf[1] = down_buf;
        op.buf[2] = counts;
        op.buf[3] = assign_rows;
        op.buf[4] = assign_slots;
        op.buf[5] = expert_out;
        op.n[0] = ffn;
        op.n[1] = hidden;
        op.n[2] = rows;
        op.n[3] = experts;
        op.n[4] = used;
        op.u[0] = YUUKA_DTYPE_Q4_K;
        CHECK_STATUS(yuuka_plan_add(plan, &op));

        op = yuuka_op{};
        op.code = YUUKA_OP_MOE_REDUCE_ADD;
        op.buf[0] = x;
        op.buf[1] = expert_out;
        op.buf[2] = weights;
        op.n[0] = hidden;
        op.n[1] = rows;
        op.n[2] = used;
        CHECK_STATUS(yuuka_plan_add(plan, &op));

        const int reps = 20;
        for (int i = 0; i < 3; ++i) {
            CHECK_STATUS(yuuka_plan_execute(plan, stream, 0, 0, 1, event));
            CHECK_STATUS(yuuka_event_wait(event));
        }
        uint32_t op_codes[16] = {};
        uint64_t op_times[16] = {};
        uint64_t op_count = 0;
        CHECK_STATUS(yuuka_plan_execute_profiled(plan, stream, 0, 0, 1, op_codes, op_times,
                                                 16, &op_count, event));
        CHECK_STATUS(yuuka_event_wait(event));
        double route_ns = 0, gate_ns = 0, down_ns = 0, reduce_ns = 0;
        for (uint64_t i = 0; i < op_count; ++i) {
            switch (op_codes[i]) {
                case YUUKA_OP_MOE_ROUTE: route_ns += double(op_times[i]); break;
                case YUUKA_OP_MOE_GATE_UP: gate_ns += double(op_times[i]); break;
                case YUUKA_OP_MOE_DOWN: down_ns += double(op_times[i]); break;
                case YUUKA_OP_MOE_REDUCE_ADD: reduce_ns += double(op_times[i]); break;
                default: break;
            }
        }
        double total_ms = 0;
        for (int rep = 0; rep < reps; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            CHECK_STATUS(yuuka_plan_execute(plan, stream, 0, 0, 1, event));
            CHECK_STATUS(yuuka_event_wait(event));
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        const double route_ms = route_ns / 1e6;
        const double gate_ms = gate_ns / 1e6;
        const double down_ms = down_ns / 1e6;
        const double reduce_ms = reduce_ns / 1e6;
        total_ms /= reps;
        const double tokens_per_s = rows > 0 ? (rows * 1000.0) / total_ms : 0.0;
        std::printf("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n", rows, route_ms, gate_ms,
                    down_ms, reduce_ms, total_ms, tokens_per_s);
        yuuka_plan_destroy(plan);
        for (yuuka_buffer * buffer : {logits, norm, ids, weights, counts, assign_rows,
                                      assign_slots, assign_weights, gate_out, up_out,
                                      act_out, expert_out, x}) {
            yuuka_buffer_free(buffer);
        }
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return 0;
}
