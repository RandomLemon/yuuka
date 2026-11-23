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

// selected_route_case compares the one-wave selected router against the binned
// router on a 256-expert row. The logits include tied winners that span lanes,
// the lowest and highest expert ids, and negative values, so both the
// descending tie-break and the normalization reduction are exercised.
int selected_route_case(context & ctx) {
    const int experts = 256, used = 8;
    std::vector<float> logits(experts);
    for (int e = 0; e < experts; ++e) {
        logits[e] = -1.0f - 0.001f * float(e);
    }
    logits[100] = 3.0f;
    logits[0] = 2.0f;
    logits[255] = 2.0f;
    logits[32] = 1.5f;
    logits[64] = 1.5f;
    logits[128] = 1.0f;
    logits[200] = 0.5f;
    logits[33] = 0.25f;
    std::vector<int> want_ids;
    std::vector<float> want_weights;
    topk(logits, used, want_ids, want_weights);

    yuuka_buffer * logits_buf = upload(ctx, logits.data(), logits.size() * 4);
    yuuka_buffer * ids_binned = upload(ctx, nullptr, used * 4);
    yuuka_buffer * weights_binned = upload(ctx, nullptr, used * 4);
    yuuka_buffer * ids_selected = upload(ctx, nullptr, used * 4);
    yuuka_buffer * weights_selected = upload(ctx, nullptr, used * 4);
    yuuka_buffer * counts = upload(ctx, nullptr, experts * 4);
    yuuka_buffer * assign_rows = upload(ctx, nullptr, experts * used * 4);
    yuuka_buffer * assign_slots = upload(ctx, nullptr, experts * used * 4);
    yuuka_buffer * assign_weights = upload(ctx, nullptr, experts * used * 4);
    CHECK(logits_buf && ids_binned && weights_binned && ids_selected &&
          weights_selected && counts && assign_rows && assign_slots && assign_weights);

    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_ROUTE;
    op.buf[0] = logits_buf;
    op.buf[1] = ids_binned;
    op.buf[2] = weights_binned;
    op.buf[3] = counts;
    op.buf[4] = assign_rows;
    op.buf[5] = assign_slots;
    op.buf[6] = assign_weights;
    op.n[0] = 1;
    op.n[1] = experts;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_ROUTE_SELECTED;
    op.buf[0] = logits_buf;
    op.buf[1] = ids_selected;
    op.buf[2] = weights_selected;
    op.n[0] = 1;
    op.n[1] = experts;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    std::vector<int32_t> binned_ids(used), selected_ids(used), replay_ids(used);
    std::vector<float> binned_weights(used), selected_weights(used), replay_weights(used);
    for (int pass = 0; pass < 2; ++pass) {
        CHECK_STATUS(yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event));
        CHECK_STATUS(yuuka_event_wait(ctx.event));
        CHECK_STATUS(yuuka_memcpy_d2h(binned_ids.data(), ids_binned, used * 4));
        CHECK_STATUS(yuuka_memcpy_d2h(binned_weights.data(), weights_binned, used * 4));
        CHECK_STATUS(yuuka_memcpy_d2h(selected_ids.data(), ids_selected, used * 4));
        CHECK_STATUS(yuuka_memcpy_d2h(selected_weights.data(), weights_selected,
                                      used * 4));
        for (int slot = 0; slot < used; ++slot) {
            CHECK(binned_ids[slot] == want_ids[slot]);
            CHECK(selected_ids[slot] == want_ids[slot]);
            CHECK(std::fabs(binned_weights[slot] - want_weights[slot]) < 1e-6f);
            CHECK(std::memcmp(&selected_weights[slot], &binned_weights[slot],
                              sizeof(float)) == 0);
        }
        if (pass == 0) {
            replay_ids = binned_ids;
            replay_weights = binned_weights;
        } else {
            CHECK(std::memcmp(binned_ids.data(), replay_ids.data(), used * 4) == 0);
            CHECK(std::memcmp(binned_weights.data(), replay_weights.data(),
                              used * 4) == 0);
            CHECK(std::memcmp(selected_ids.data(), replay_ids.data(), used * 4) == 0);
            CHECK(std::memcmp(selected_weights.data(), replay_weights.data(),
                              used * 4) == 0);
        }
    }
    std::printf("selected route: ids match, weights bitwise, replay deterministic\n");

    for (yuuka_buffer * buffer : {logits_buf, ids_binned, weights_binned, ids_selected,
                                  weights_selected, counts, assign_rows, assign_slots,
                                  assign_weights}) {
        yuuka_buffer_free(buffer);
    }
    yuuka_plan_destroy(plan);
    return 0;
}

// grouped_one_row_case keeps the binned route and the grouped expert kernels on
// a one-row prefill tail, where the assignment bins are live.
int grouped_one_row_case(context & ctx) {
    const int experts = 4, used = 2, hidden = 16, ffn = 8;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> norm(hidden), logits(experts);
    std::vector<float> gate(experts * ffn * hidden), up(experts * ffn * hidden);
    std::vector<float> down(experts * hidden * ffn);
    for (float & v : norm) v = dist(rng);
    for (float & v : logits) v = dist(rng);
    for (float & v : gate) v = dist(rng);
    for (float & v : up) v = dist(rng);
    for (float & v : down) v = dist(rng);
    std::vector<int> ids;
    std::vector<float> weights;
    topk(logits, used, ids, weights);
    std::vector<float> want(hidden, 0.0f);
    for (int slot = 0; slot < used; ++slot) {
        const int e = ids[slot];
        std::vector<float> act(ffn);
        for (int j = 0; j < ffn; ++j) {
            float g = 0, u = 0;
            for (int i = 0; i < hidden; ++i) {
                g += norm[i] * gate[(e * ffn + j) * hidden + i];
                u += norm[i] * up[(e * ffn + j) * hidden + i];
            }
            act[j] = silu(g) * u;
        }
        for (int r = 0; r < hidden; ++r) {
            float value = 0;
            for (int j = 0; j < ffn; ++j) {
                value += act[j] * down[(e * hidden + r) * ffn + j];
            }
            want[r] += weights[slot] * value;
        }
    }

    yuuka_buffer * norm_buf = upload(ctx, norm.data(), norm.size() * 4);
    yuuka_buffer * logits_buf = upload(ctx, logits.data(), logits.size() * 4);
    yuuka_buffer * gate_buf = upload(ctx, gate.data(), gate.size() * 4);
    yuuka_buffer * up_buf = upload(ctx, up.data(), up.size() * 4);
    yuuka_buffer * down_buf = upload(ctx, down.data(), down.size() * 4);
    yuuka_buffer * ids_buf = upload(ctx, nullptr, used * 4);
    yuuka_buffer * weights_buf = upload(ctx, nullptr, used * 4);
    yuuka_buffer * counts = upload(ctx, nullptr, experts * 4);
    yuuka_buffer * assign_rows = upload(ctx, nullptr, experts * used * 4);
    yuuka_buffer * assign_slots = upload(ctx, nullptr, experts * used * 4);
    yuuka_buffer * assign_weights = upload(ctx, nullptr, experts * used * 4);
    yuuka_buffer * gate_out = upload(ctx, nullptr, used * ffn * 4);
    yuuka_buffer * up_out = upload(ctx, nullptr, used * ffn * 4);
    yuuka_buffer * act_out = upload(ctx, nullptr, used * ffn * 4);
    yuuka_buffer * expert_out = upload(ctx, nullptr, used * hidden * 4);
    yuuka_buffer * x_buf = upload(ctx, nullptr, hidden * 4);
    CHECK(norm_buf && logits_buf && gate_buf && up_buf && down_buf && ids_buf &&
          weights_buf && counts && assign_rows && assign_slots && assign_weights &&
          gate_out && up_out && act_out && expert_out && x_buf);

    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_ROUTE;
    op.buf[0] = logits_buf;
    op.buf[1] = ids_buf;
    op.buf[2] = weights_buf;
    op.buf[3] = counts;
    op.buf[4] = assign_rows;
    op.buf[5] = assign_slots;
    op.buf[6] = assign_weights;
    op.n[0] = 1;
    op.n[1] = experts;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_GATE_UP;
    op.buf[0] = norm_buf;
    op.buf[1] = gate_buf;
    op.buf[2] = up_buf;
    op.buf[3] = counts;
    op.buf[4] = assign_rows;
    op.buf[5] = assign_slots;
    op.buf[6] = gate_out;
    op.buf[7] = up_out;
    op.n[0] = hidden;
    op.n[1] = ffn;
    op.n[2] = 1;
    op.n[3] = experts;
    op.n[4] = used;
    op.u[0] = YUUKA_DTYPE_F32;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_SILU_MUL;
    op.buf[0] = gate_out;
    op.buf[1] = up_out;
    op.buf[2] = act_out;
    op.n[0] = used * ffn;
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
    op.n[2] = 1;
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
    op.n[1] = 1;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    std::vector<float> zero(hidden, 0.0f);
    CHECK_STATUS(yuuka_memcpy_h2d(x_buf, zero.data(), zero.size() * 4));
    CHECK_STATUS(yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event));
    CHECK_STATUS(yuuka_event_wait(ctx.event));
    std::vector<float> got(hidden);
    CHECK_STATUS(yuuka_memcpy_d2h(got.data(), x_buf, got.size() * 4));
    float max_diff = 0;
    for (int i = 0; i < hidden; ++i) {
        max_diff = std::fmax(max_diff, std::fabs(got[i] - want[i]));
    }
    std::printf("grouped one-row bins: max abs diff %.6g\n", max_diff);
    CHECK(max_diff < 2e-4f);

    for (yuuka_buffer * buffer : {norm_buf, logits_buf, gate_buf, up_buf, down_buf,
                                  ids_buf, weights_buf, counts, assign_rows, assign_slots,
                                  assign_weights, gate_out, up_out, act_out, expert_out,
                                  x_buf}) {
        yuuka_buffer_free(buffer);
    }
    yuuka_plan_destroy(plan);
    return 0;
}

// reduce_specialization_case runs the 2048-column one-row reduction and the
// generic block reduction over the same per-column data. The generic form needs
// a different width to be selected, so its extra final column is ignored; the
// first 2048 residual values must match bitwise.
int reduce_specialization_case(context & ctx) {
    const int used = 8, hidden = 2048;
    std::vector<float> weights(used);
    std::vector<float> x_new(hidden), x_old(hidden + 1);
    std::vector<float> out_new(used * hidden), out_old(used * (hidden + 1));
    for (int slot = 0; slot < used; ++slot) {
        weights[slot] = 0.05f + 0.125f * float(slot + 1);
    }
    for (int i = 0; i < hidden + 1; ++i) {
        x_old[i] = 0.25f + 0.001f * float(i);
        if (i < hidden) x_new[i] = x_old[i];
    }
    for (int slot = 0; slot < used; ++slot) {
        for (int i = 0; i < hidden + 1; ++i) {
            const float value = 0.5f * std::sin(float(slot * 37 + i)) +
                                0.001f * float(i) - 0.125f * float(slot);
            out_old[slot * (hidden + 1) + i] = value;
            if (i < hidden) out_new[slot * hidden + i] = value;
        }
    }
    yuuka_buffer * x_new_buf = upload(ctx, x_new.data(), x_new.size() * 4);
    yuuka_buffer * x_old_buf = upload(ctx, x_old.data(), x_old.size() * 4);
    yuuka_buffer * out_new_buf = upload(ctx, out_new.data(), out_new.size() * 4);
    yuuka_buffer * out_old_buf = upload(ctx, out_old.data(), out_old.size() * 4);
    yuuka_buffer * weights_buf = upload(ctx, weights.data(), weights.size() * 4);
    CHECK(x_new_buf && x_old_buf && out_new_buf && out_old_buf && weights_buf);

    yuuka_plan * plan = nullptr;
    CHECK_STATUS(yuuka_plan_create(ctx.device, &plan));
    yuuka_op op{};
    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = x_new_buf;
    op.buf[1] = out_new_buf;
    op.buf[2] = weights_buf;
    op.n[0] = hidden;
    op.n[1] = 1;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    op = yuuka_op{};
    op.code = YUUKA_OP_MOE_REDUCE_ADD;
    op.buf[0] = x_old_buf;
    op.buf[1] = out_old_buf;
    op.buf[2] = weights_buf;
    op.n[0] = hidden + 1;
    op.n[1] = 1;
    op.n[2] = used;
    CHECK_STATUS(yuuka_plan_add(plan, &op));

    CHECK_STATUS(yuuka_plan_execute(plan, ctx.stream, 0, 0, 1, ctx.event));
    CHECK_STATUS(yuuka_event_wait(ctx.event));
    std::vector<float> got_new(hidden), got_old(hidden + 1);
    CHECK_STATUS(yuuka_memcpy_d2h(got_new.data(), x_new_buf, got_new.size() * 4));
    CHECK_STATUS(yuuka_memcpy_d2h(got_old.data(), x_old_buf, got_old.size() * 4));
    CHECK(std::memcmp(got_new.data(), got_old.data(), hidden * sizeof(float)) == 0);
    std::printf("reduce specialization: first %d columns bitwise equal\n", hidden);

    for (yuuka_buffer * buffer : {x_new_buf, x_old_buf, out_new_buf, out_old_buf,
                                  weights_buf}) {
        yuuka_buffer_free(buffer);
    }
    yuuka_plan_destroy(plan);
    return 0;
}

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

    if (selected_route_case(ctx) != 0 || grouped_one_row_case(ctx) != 0 ||
        reduce_specialization_case(ctx) != 0) {
        return 1;
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
