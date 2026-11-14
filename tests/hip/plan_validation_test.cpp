// Verifies that every legacy single-token plan operation is validated before it
// enters the plan. Undersized buffers must fail at yuuka_plan_add instead of
// reaching execution.
#include "yuuka_hip.h"

#include <cstdio>
#include <initializer_list>

namespace {

bool expect(yuuka_status got, yuuka_status want, const char * name) {
    if (got == want) {
        std::printf("%s: %s\n", name, got == YUUKA_OK ? "accepted" : "rejected");
        return true;
    }
    std::fprintf(stderr, "%s: status %d, want %d (%s)\n", name, int(got), int(want),
                 yuuka_last_error());
    return false;
}

yuuka_op make_op(uint32_t code, yuuka_buffer * buf) {
    yuuka_op op{};
    op.code = code;
    for (int i = 0; i < 12; ++i) op.buf[i] = buf;
    return op;
}

} // namespace

int main() {
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_buffer * tiny = nullptr;
    yuuka_buffer * roomy = nullptr;
    yuuka_plan * plan = nullptr;
    bool ok = yuuka_buffer_alloc(device, 4, &tiny) == YUUKA_OK &&
              yuuka_buffer_alloc(device, 256, &roomy) == YUUKA_OK &&
              yuuka_plan_create(device, &plan) == YUUKA_OK;
    if (!ok) {
        std::fprintf(stderr, "setup: %s\n", yuuka_last_error());
        return 1;
    }

    // Each undersized operation must be rejected at plan construction.
    {
        yuuka_op op = make_op(YUUKA_OP_RMSNORM, tiny);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "RMSNORM") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_L2NORM, tiny);
        op.n[0] = 1;
        op.n[1] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "L2NORM") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_GATED_RMSNORM, tiny);
        op.n[0] = 8;
        op.n[1] = 1;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "GATED_RMSNORM") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_RMSNORM_HEADS, tiny);
        op.n[0] = 8;
        op.n[1] = 1;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "RMSNORM_HEADS") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_ROPE, tiny);
        op.n[0] = 1;
        op.n[1] = 8;
        op.n[2] = 8;
        op.n[3] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "ROPE") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_ATTENTION, tiny);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 8;
        op.n[3] = 8;
        op.u[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "ATTENTION") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_CONV1D, tiny);
        op.n[0] = 4;
        op.n[1] = 2;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "CONV1D") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_GDN, tiny);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 4;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "GDN") && ok;
    }
    for (uint32_t code : {YUUKA_OP_SILU_MUL, YUUKA_OP_ADD, YUUKA_OP_COPY, YUUKA_OP_SCALE}) {
        yuuka_op op = make_op(code, tiny);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "elementwise") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_KV_WRITE, tiny);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "KV_WRITE") && ok;
    }
    {
        // Contiguous prefill attention has no dense query variant; rejecting
        // u[3] here prevents a validation range smaller than the kernel reads.
        yuuka_op op = make_op(YUUKA_OP_ATTENTION_PREFILL, roomy);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 4;
        op.n[3] = 4;
        op.n[4] = 1;
        op.u[0] = 1;
        op.u[3] = 1;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_UNSUPPORTED,
                    "ATTENTION_PREFILL dense") && ok;
    }
    {
        // The products rows*hidden and rows*used wrap to zero in 64 bits. The
        // validator must reject the dimensions before checking buffer ranges.
        yuuka_op op = make_op(YUUKA_OP_MOE_REDUCE_ADD, tiny);
        op.n[0] = uint64_t{1} << 32;
        op.n[1] = uint64_t{1} << 32;
        op.n[2] = uint64_t{1} << 32;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "MOE_REDUCE_ADD overflow") && ok;
    }

    // The same shapes with roomy buffers must still be accepted.
    {
        yuuka_op op = make_op(YUUKA_OP_RMSNORM, roomy);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "RMSNORM valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_L2NORM, roomy);
        op.n[0] = 1;
        op.n[1] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "L2NORM valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_GATED_RMSNORM, roomy);
        op.n[0] = 8;
        op.n[1] = 1;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "GATED_RMSNORM valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_RMSNORM_HEADS, roomy);
        op.n[0] = 8;
        op.n[1] = 1;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "RMSNORM_HEADS valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_ROPE, roomy);
        op.n[0] = 1;
        op.n[1] = 8;
        op.n[2] = 8;
        op.n[3] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "ROPE valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_ATTENTION, roomy);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 8;
        op.n[3] = 8;
        op.u[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "ATTENTION valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_CONV1D, roomy);
        op.n[0] = 4;
        op.n[1] = 2;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "CONV1D valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_GDN, roomy);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 4;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "GDN valid") && ok;
    }
    for (uint32_t code : {YUUKA_OP_SILU_MUL, YUUKA_OP_ADD, YUUKA_OP_COPY, YUUKA_OP_SCALE}) {
        yuuka_op op = make_op(code, roomy);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "elementwise valid") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_KV_WRITE, roomy);
        op.n[0] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "KV_WRITE valid") && ok;
    }

    yuuka_plan_destroy(plan);
    yuuka_buffer_free(roomy);
    yuuka_buffer_free(tiny);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
