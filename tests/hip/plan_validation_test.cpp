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
    {
        yuuka_op op = make_op(YUUKA_OP_RECURRENT_TOKEN, tiny);
        op.n[0] = 1;
        op.n[1] = 1;
        op.n[2] = 128;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "RECURRENT_TOKEN") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_CONV_L2NORM, tiny);
        op.n[0] = 256;
        op.n[1] = 4;
        op.n[2] = 256;
        op.n[3] = 128;
        op.n[4] = 128;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT, "CONV_L2NORM") && ok;
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
    {
        // The one-row selected router takes logits, ids, and weights only. The
        // roomy buffer holds 64 exponents, 8 ids, and 8 weights.
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.n[0] = 1;
        op.n[1] = 64;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "MOE_ROUTE_SELECTED") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.n[0] = 2;
        op.n[1] = 64;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "MOE_ROUTE_SELECTED rows") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.n[0] = 1;
        op.n[1] = 64;
        op.n[2] = 33;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_UNSUPPORTED,
                    "MOE_ROUTE_SELECTED slots") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.n[0] = 1;
        op.n[1] = 1025;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_UNSUPPORTED,
                    "MOE_ROUTE_SELECTED experts") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, tiny);
        op.n[0] = 1;
        op.n[1] = 2;
        op.n[2] = 1;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "MOE_ROUTE_SELECTED short") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.buf[2] = nullptr;
        op.n[0] = 1;
        op.n[1] = 64;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "MOE_ROUTE_SELECTED missing") && ok;
    }
    {
        yuuka_op op = make_op(YUUKA_OP_MOE_ROUTE_SELECTED, roomy);
        op.off[1] = 4;
        op.n[0] = 1;
        op.n[1] = 64;
        op.n[2] = 8;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "MOE_ROUTE_SELECTED offset") && ok;
    }
    {
        // The streaming LM head keeps one accumulator per row per channel in
        // registers, so a row block above the cap has no kernel.
        yuuka_op op = make_op(YUUKA_OP_LM_HEAD, roomy);
        op.n[0] = 256;
        op.n[1] = 1;
        op.n[2] = 17;
        op.u[0] = YUUKA_DTYPE_Q8_0;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "LM_HEAD rows") && ok;
    }
    {
        // A K step that is not a multiple of 256 cannot be one quantization
        // block, which is a capability failure rather than an argument error.
        yuuka_op op = make_op(YUUKA_OP_LM_HEAD, roomy);
        op.n[0] = 384;
        op.n[1] = 1;
        op.n[2] = 1;
        op.u[0] = YUUKA_DTYPE_Q8_0;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_UNSUPPORTED,
                    "LM_HEAD width") && ok;
    }
    {
        // roomy holds 256 bytes, while one 256-wide Q8_0 weight row is 272.
        yuuka_op op = make_op(YUUKA_OP_LM_HEAD, roomy);
        op.n[0] = 256;
        op.n[1] = 1;
        op.n[2] = 1;
        op.u[0] = YUUKA_DTYPE_Q8_0;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_ERR_INVALID_ARGUMENT,
                    "LM_HEAD weight size") && ok;
    }
    {
        yuuka_buffer * scratch = nullptr;
        if (yuuka_buffer_alloc(device, 2048, &scratch) != YUUKA_OK) {
            std::fprintf(stderr, "SwiGLU Q8_1 test scratch: %s\n", yuuka_last_error());
            ok = false;
        } else {
            yuuka_op op = make_op(YUUKA_OP_SILU_MUL_QUANT_Q8_1, scratch);
            op.n[0] = 128;
            op.n[1] = 1;
            op.off[0] = 16;
            op.off[1] = 544;
            op.off[2] = 1088;
            op.off[3] = 1224;
            for (int i = 0; i < 4; ++i) {
                yuuka_op bad = op;
                bad.buf[i] = tiny;
                ok = expect(yuuka_plan_add(plan, &bad), YUUKA_ERR_INVALID_ARGUMENT,
                            "SwiGLU Q8_1 short buffer") && ok;
                bad = op;
                ++bad.off[i];
                ok = expect(yuuka_plan_add(plan, &bad), YUUKA_ERR_INVALID_ARGUMENT,
                            "SwiGLU Q8_1 unaligned offset") && ok;
            }
            yuuka_op bad = op;
            bad.off[2] = op.off[1];
            ok = expect(yuuka_plan_add(plan, &bad), YUUKA_ERR_INVALID_ARGUMENT,
                        "SwiGLU Q8_1 overlap") && ok;
            bad = op;
            bad.n[0] = 160;
            ok = expect(yuuka_plan_add(plan, &bad), YUUKA_ERR_UNSUPPORTED,
                        "SwiGLU Q8_1 width") && ok;
            yuuka_plan * fused_plan = nullptr;
            ok = expect(yuuka_plan_create(device, &fused_plan), YUUKA_OK,
                        "SwiGLU Q8_1 plan") && ok;
            if (fused_plan != nullptr) {
                ok = expect(yuuka_plan_add(fused_plan, &op), YUUKA_OK,
                            "SwiGLU Q8_1 valid offset spans") && ok;
                yuuka_plan_destroy(fused_plan);
            }
            yuuka_buffer_free(scratch);
        }
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

    // The fused decode matvec forms need a packed weight buffer and a 256-aligned
    // input width, so they are checked separately below.
    // The fused delta-net pair needs more room than the elementwise buffers: the
    // recurrent state is 128x128 FP32 per head.
    yuuka_buffer * wide = nullptr;
    if (yuuka_buffer_alloc(device, 1u << 20, &wide) != YUUKA_OK) {
        std::fprintf(stderr, "wide buffer: %s\n", yuuka_last_error());
        ok = false;
    } else {
        yuuka_op op = make_op(YUUKA_OP_RECURRENT_TOKEN, wide);
        op.n[0] = 1;
        op.n[1] = 2;
        op.n[2] = 128;
        ok = expect(yuuka_plan_add(plan, &op), YUUKA_OK, "RECURRENT_TOKEN valid") && ok;
        yuuka_op conv = make_op(YUUKA_OP_CONV_L2NORM, wide);
        conv.n[0] = 256;
        conv.n[1] = 4;
        conv.n[2] = 256;
        conv.n[3] = 128;
        conv.n[4] = 128;
        ok = expect(yuuka_plan_add(plan, &conv), YUUKA_OK, "CONV_L2NORM valid") && ok;
        // Both kernels are compiled for the supported geometry, so a shape the
        // plan builder would refuse must be refused here as well.
        yuuka_op narrow = make_op(YUUKA_OP_RECURRENT_TOKEN, wide);
        narrow.n[0] = 1;
        narrow.n[1] = 2;
        narrow.n[2] = 64;
        ok = expect(yuuka_plan_add(plan, &narrow), YUUKA_ERR_UNSUPPORTED,
                    "RECURRENT_TOKEN narrow state") && ok;
        yuuka_op unaligned = make_op(YUUKA_OP_CONV_L2NORM, wide);
        unaligned.n[0] = 192;
        unaligned.n[1] = 4;
        unaligned.n[2] = 128;
        unaligned.n[3] = 128;
        unaligned.n[4] = 128;
        ok = expect(yuuka_plan_add(plan, &unaligned), YUUKA_ERR_UNSUPPORTED,
                    "CONV_L2NORM unaligned channels") && ok;

        // The fused decode matvec forms: the residual epilogue and the two-matrix
        // pair. Two megabytes hold the 256 x 256 Q4_K matrices the shape rules
        // ask for; the input and output rows are 256 floats.
        yuuka_buffer * packed = nullptr;
        yuuka_buffer * rows = nullptr;
        if (yuuka_buffer_alloc(device, 1u << 21, &packed) != YUUKA_OK ||
            yuuka_buffer_alloc(device, 4096, &rows) != YUUKA_OK) {
            std::fprintf(stderr, "fused matvec buffers: %s\n", yuuka_last_error());
            ok = false;
        } else {
            yuuka_op residual = make_op(YUUKA_OP_GEMV_RESIDUAL, packed);
            residual.buf[1] = rows;
            residual.buf[2] = rows;
            residual.buf[3] = rows;
            residual.n[0] = 256;
            residual.n[1] = 256;
            residual.n[2] = 1;
            residual.u[0] = YUUKA_DTYPE_Q4_K;
            ok = expect(yuuka_plan_add(plan, &residual), YUUKA_OK, "GEMV_RESIDUAL valid") && ok;
            yuuka_op pair = make_op(YUUKA_OP_GEMV_PAIR, packed);
            pair.buf[1] = packed;
            pair.buf[2] = rows;
            pair.buf[3] = rows;
            pair.n[0] = 256;
            pair.n[1] = 256;
            pair.n[2] = 256;
            pair.u[0] = YUUKA_DTYPE_Q4_K;
            pair.u[1] = YUUKA_DTYPE_Q4_K;
            pair.u[2] = 0;
            ok = expect(yuuka_plan_add(plan, &pair), YUUKA_OK, "GEMV_PAIR valid") && ok;
            // The SwiGLU epilogue consumes the second row and needs equal widths.
            yuuka_op swiglu = pair;
            swiglu.u[2] = 1;
            swiglu.n[2] = 128;
            ok = expect(yuuka_plan_add(plan, &swiglu), YUUKA_ERR_INVALID_ARGUMENT,
                        "GEMV_PAIR swiglu unequal widths") && ok;
            // A mixed-dtype pair would need a run-time dispatch the kernel does
            // not take, so it is rejected instead of being run the slow way.
            yuuka_op mixed = pair;
            mixed.u[1] = YUUKA_DTYPE_Q6_K;
            ok = expect(yuuka_plan_add(plan, &mixed), YUUKA_ERR_UNSUPPORTED,
                        "GEMV_PAIR mixed dtypes") && ok;
            // An unsupported input width is rejected before the kernel sees it.
            yuuka_op narrow_in = residual;
            narrow_in.n[0] = 192;
            ok = expect(yuuka_plan_add(plan, &narrow_in), YUUKA_ERR_UNSUPPORTED,
                        "GEMV_RESIDUAL unaligned input") && ok;
            yuuka_buffer_free(rows);
            yuuka_buffer_free(packed);
        }
        yuuka_buffer_free(wide);
    }

    yuuka_plan_destroy(plan);
    yuuka_buffer_free(roomy);
    yuuka_buffer_free(tiny);
    yuuka_device_close(device);
    return ok ? 0 : 1;
}
