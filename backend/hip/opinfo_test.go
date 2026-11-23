//go:build hip && linux

package hip

import (
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

// TestOpNameAndRolesCoverABI checks that every op code the C ABI declares has
// a name and a recorded first-slot role. A new code without an entry would make
// the attribution tables silent about it, so this fails instead.
func TestOpNameAndRolesCoverABI(t *testing.T) {
	if MaxOpCode < 44 {
		t.Fatalf("shadow covers op codes up to %d, want at least 44", MaxOpCode)
	}
	names := make(map[string]uint32, MaxOpCode)
	for code := uint32(1); code <= MaxOpCode; code++ {
		name := OpName(code)
		if name == "" || name == "unknown" {
			t.Errorf("op code %d has no name", code)
			continue
		}
		if other, ok := names[name]; ok {
			t.Errorf("op codes %d and %d share the name %q", other, code, name)
		}
		names[name] = code
		if roles := opRoles(code); roles[0] == RoleUnused {
			t.Errorf("op code %d (%s) has no recorded first-slot role", code, name)
		}
	}
	// Pin the names used in the prefill and decode attribution.
	for code, want := range map[uint32]string{
		OpGemv:             "gemv",
		OpGDN:              "gdn",
		OpRecurrentToken:   "recurrent_token",
		OpConvL2Norm:       "conv_l2norm",
		OpRMSNorm:          "rmsnorm",
		OpSiluMul:          "silu_mul",
		OpQuantQ8_1:        "quant_q8_1",
		OpLMHead:           "lm_head",
		OpGemvResidual:     "gemv_residual",
		OpGemvPair:         "gemv_pair",
		OpSiluMulQuantQ8_1: "silu_mul_quant_q8_1",
		OpPrefillQ8Pair:    "prefill_q8_0_pair",
	} {
		if got := OpName(code); got != want {
			t.Errorf("OpName(%d) = %q, want %q", code, got, want)
		}
	}
}

// TestPlanOpShadow checks that the Go-side shadow records what the plan was
// given: code, dimensions, dtype selector, buffer identity, and the role of
// each slot.
func TestPlanOpShadow(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if got := plan.Ops(); len(got) != 0 || plan.OpCount() != 0 {
		t.Fatalf("new plan holds %d operations", len(got))
	}

	// A Q4_K row at 256 values is 144 bytes, so four output rows are 576.
	w, err := dev.Alloc(4 * 144)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	x, err := dev.Alloc(256 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err := plan.AddGemv(w, x, out, 256, 4, tensor.Q4_K); err != nil {
		t.Fatal(err)
	}
	if err := plan.AddCopy(out, x, 4); err != nil {
		t.Fatal(err)
	}
	if err := plan.Add(out, x, 4); err != nil {
		t.Fatal(err)
	}
	plan.MarkTail()
	if err := plan.AddScale(out, 4, 0.5); err != nil {
		t.Fatal(err)
	}

	ops := plan.Ops()
	if len(ops) != 4 || plan.OpCount() != 4 {
		t.Fatalf("plan holds %d operations, want 4", len(ops))
	}
	gemv := ops[0]
	if gemv.Index != 0 || gemv.Code != OpGemv || gemv.Name() != "gemv" {
		t.Fatalf("gemv shadow: %+v", gemv)
	}
	if gemv.N[0] != 256 || gemv.N[1] != 4 {
		t.Fatalf("gemv dimensions: %v", gemv.N)
	}
	if gemv.U[0] != uint32(tensor.Q4_K) {
		t.Fatalf("gemv dtype selector %d, want %d", gemv.U[0], tensor.Q4_K)
	}
	if gemv.Bufs[0] != w.ID() || gemv.Bufs[1] != x.ID() || gemv.Bufs[2] != out.ID() {
		t.Fatalf("gemv buffers %v do not identify the added buffers", gemv.Bufs)
	}
	if gemv.Roles[0] != RoleWeight || gemv.Roles[1] != RoleActivation ||
		gemv.Roles[2] != RoleOutput {
		t.Fatalf("gemv roles: %v", gemv.Roles)
	}
	if gemv.Tail() {
		t.Fatal("operation before MarkTail is marked as tail")
	}
	if copyOp := ops[1]; copyOp.Name() != "copy" || copyOp.Bufs[0] != out.ID() ||
		copyOp.Roles[0] != RoleOutput {
		t.Fatalf("copy shadow: %+v", copyOp)
	}
	if addOp := ops[2]; addOp.Name() != "add" || addOp.Roles[0] != RoleAccumulator {
		t.Fatalf("add shadow: %+v", addOp)
	}
	scale := ops[3]
	if !scale.Tail() || scale.Index != 3 || scale.Name() != "scale" {
		t.Fatalf("tail shadow: %+v", scale)
	}
	// Ops returns a copy: mutating it must not change the plan's record.
	ops[0].Code = 0
	if plan.Ops()[0].Code != OpGemv {
		t.Fatal("Ops returned an aliased slice")
	}
}

// TestProfiledTailAlignment checks the property ExecuteProfiled documents: with
// runTail=false the returned times line up with a prefix of the shadow.
func TestProfiledTailAlignment(t *testing.T) {
	dev := openTestDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer event.Close()
	x, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	y, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if err := plan.AddCopy(y, x, 4); err != nil {
		t.Fatal(err)
	}
	plan.MarkTail()
	if err := plan.AddScale(y, 4, 2); err != nil {
		t.Fatal(err)
	}
	if len(plan.Ops()) != 2 {
		t.Fatalf("plan holds %d operations, want 2", len(plan.Ops()))
	}
	times, err := plan.ExecuteProfiled(stream, 0, 0, false, event)
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Wait(); err != nil {
		t.Fatal(err)
	}
	if len(times) != 1 {
		t.Fatalf("profiled %d operations without the tail, want 1", len(times))
	}
	if times[0].Code != OpCopy {
		t.Fatalf("profiled operation %d (%s), want copy", times[0].Code, OpName(times[0].Code))
	}
	full, err := plan.ExecuteProfiled(stream, 0, 0, true, event)
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Wait(); err != nil {
		t.Fatal(err)
	}
	if len(full) != len(plan.Ops()) {
		t.Fatalf("profiled %d operations with the tail, plan holds %d", len(full),
			len(plan.Ops()))
	}
	for i, info := range plan.Ops() {
		if full[i].Code != info.Code {
			t.Fatalf("operation %d: profiled code %d, shadow code %d", i, full[i].Code,
				info.Code)
		}
	}
}
