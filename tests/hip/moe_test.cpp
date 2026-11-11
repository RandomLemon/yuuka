#include "yuuka_hip.h"

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

void topk(const std::vector<float> & logits, int used, std::vector<int> & ids,
          std::vector<float> & weights) {
    ids.assign(used, 0);
    weights.assign(used, 0.0f);
    for (int slot = 0; slot < used; ++slot) {
        int best = -1;
        for (int e = 0; e < int(logits.size()); ++e) {
            bool taken = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (ids[prior] == e) taken = true;
            }
            if (!taken && (best < 0 || logits[e] > logits[best])) best = e;
        }
        ids[slot] = best;
    }
    const float max = logits[ids[0]];
    float total = 0;
    for (int slot = 0; slot < used; ++slot) {
        weights[slot] = std::exp(logits[ids[slot]] - max);
        total += weights[slot];
    }
    for (float & weight : weights) weight /= total;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

} // namespace

int main() {
    context ctx;
    if (setup(ctx) != 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }
    const int rows = 3, experts = 4, used = 2, hidden = 16, ffn = 8;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> norm(rows * hidden), logits(rows * experts);
    std::vector<float> gate(experts * ffn * hidden), up(experts * ffn * hidden);
    std::vector<float> down(experts * hidden * ffn);
    for (float & v : norm) v = dist(rng);
    for (float & v : logits) v = dist(rng);
    for (float & v : gate) v = dist(rng);
    for (float & v : up) v = dist(rng);
    for (float & v : down) v = dist(rng);

    std::vector<float> want(rows * hidden, 0.0f);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> row_logits(logits.begin() + row * experts,
                                      logits.begin() + (row + 1) * experts);
        std::vector<int> ids;
        std::vector<float> weights;
        topk(row_logits, used, ids, weights);
        for (int slot = 0; slot < used; ++slot) {
            const int e = ids[slot];
            for (int r = 0; r < hidden; ++r) {
                std::vector<float> act(ffn);
                for (int j = 0; j < ffn; ++j) {
                    float g = 0, u = 0;
                    for (int i = 0; i < hidden; ++i) {
                        const float x = norm[row * hidden + i];
                        g += x * gate[(e * ffn + j) * hidden + i];
                        u += x * up[(e * ffn + j) * hidden + i];
                    }
                    act[j] = silu(g) * u;
                }
                float value = 0;
                for (int j = 0; j < ffn; ++j) {
                    value += act[j] * down[(e * hidden + r) * ffn + j];
                }
                want[row * hidden + r] += weights[slot] * value;
            }
        }
    }

    yuuka_buffer * norm_buf = upload(ctx, norm.data(), norm.size() * 4);
    yuuka_buffer * logits_buf = upload(ctx, logits.data(), logits.size() * 4);
    yuuka_buffer * gate_buf = upload(ctx, gate.data(), gate.size() * 4);
    yuuka_buffer * up_buf = upload(ctx, up.data(), up.size() * 4);
    yuuka_buffer * down_buf = upload(ctx, down.data(), down.size() * 4);
    yuuka_buffer * ids_buf = upload(ctx, nullptr, rows * used * 4);
    yuuka_buffer * weights_buf = upload(ctx, nullptr, rows * used * 4);
    yuuka_buffer * counts_buf = upload(ctx, nullptr, experts * 4);
    yuuka_buffer * assign_rows = upload(ctx, nullptr, experts * rows * used * 4);
    yuuka_buffer * assign_slots = upload(ctx, nullptr, experts * rows * used * 4);
    yuuka_buffer * assign_weights = upload(ctx, nullptr, experts * rows * used * 4);
    yuuka_buffer * gate_out = upload(ctx, nullptr, rows * used * ffn * 4);
    yuuka_buffer * up_out = upload(ctx, nullptr, rows * used * ffn * 4);
    yuuka_buffer * act_out = upload(ctx, nullptr, rows * used * ffn * 4);
    yuuka_buffer * expert_out = upload(ctx, nullptr, rows * used * hidden * 4);
    yuuka_buffer * x_buf = upload(ctx, nullptr, rows * hidden * 4);
    CHECK(norm_buf && logits_buf && gate_buf && up_buf && down_buf && ids_buf &&
          weights_buf && counts_buf && assign_rows && assign_slots && assign_weights &&
          gate_out && up_out && act_out && expert_out && x_buf);

    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_ROUTE;
    op.buf[0] = logits_buf;
    op.buf[1] = ids_buf;
    op.buf[2] = weights_buf;
    op.buf[3] = counts_buf;
    op.buf[4] = assign_rows;
    op.buf[5] = assign_slots;
    op.buf[6] = assign_weights;
    op.n[0] = rows;
    op.n[1] = experts;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_GATE_UP;
    op.buf[0] = norm_buf;
    op.buf[1] = gate_buf;
    op.buf[2] = up_buf;
    op.buf[3] = counts_buf;
    op.buf[4] = assign_rows;
    op.buf[5] = assign_slots;
    op.buf[6] = gate_out;
    op.buf[7] = up_out;
    op.n[0] = hidden;
    op.n[1] = ffn;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = YUUKA_DTYPE_F32;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_SILU_MUL;
    op.buf[0] = gate_out;
    op.buf[1] = up_out;
    op.buf[2] = act_out;
    op.n[0] = rows * used * ffn;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_DOWN;
    op.buf[0] = act_out;
    op.buf[1] = down_buf;
    op.buf[2] = counts_buf;
    op.buf[3] = assign_rows;
    op.buf[4] = assign_slots;
    op.buf[5] = expert_out;
    op.n[0] = ffn;
    op.n[1] = hidden;
    op.n[2] = rows;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = YUUKA_DTYPE_F32;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = x_buf;
    op.buf[1] = expert_out;
    op.buf[2] = weights_buf;
    op.n[0] = hidden;
    op.n[1] = rows;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    std::vector<float> zero(rows * hidden, 0.0f);
    CHECK_STATUS(yuuka_memcpy_h2d(x_buf, zero.data(), zero.size() * 4));
    CHECK_STATUS(yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event));
    CHECK_STATUS(yuuka_event_wait(ctx.event));

    std::vector<float> got(rows * hidden);
    CHECK_STATUS(yuuka_memcpy_d2h(got.data(), x_buf, got.size() * 4));

    float max_diff = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_diff = std::fmax(max_diff, std::fabs(got[i] - want[i]));
    }
    std::printf("moe grouped expert path: max abs diff %.6g\n", max_diff);
    CHECK(max_diff < 2e-4f);

    std::vector<int32_t> host_ids(rows * used);
    std::vector<float> host_weights(rows * used);
    CHECK_STATUS(yuuka_memcpy_d2h(host_ids.data(), ids_buf, host_ids.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(host_weights.data(), weights_buf, host_weights.size() * 4));
    for (int row = 0; row < rows; ++row) {
        std::vector<float> row_logits(logits.begin() + row * experts,
                                      logits.begin() + (row + 1) * experts);
        std::vector<int> want_ids;
        std::vector<float> want_weights;
        topk(row_logits, used, want_ids, want_weights);
        for (int slot = 0; slot < used; ++slot) {
            CHECK(host_ids[row * used + slot] == want_ids[slot]);
            CHECK(std::fabs(host_weights[row * used + slot] -
                            want_weights[slot]) < 1e-6f);
        }
    }

    for (yuuka_buffer * buffer : {norm_buf, logits_buf, gate_buf, up_buf, down_buf,
                                  ids_buf, weights_buf, counts_buf, assign_rows,
                                  assign_slots, assign_weights, gate_out, up_out,
                                  act_out, expert_out, x_buf}) {
        yuuka_buffer_free(buffer);
    }
    yuuka_plan_destroy(plan);
    teardown(ctx);
    return 0;
}
