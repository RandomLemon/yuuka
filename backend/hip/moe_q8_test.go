//go:build hip && linux

package hip

import (
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

// q8MoEBuffers is the buffer set one Q8_1 expert route needs, sized for the
// given shapes. The buffers are real device allocations so p.check sees the
// same bounds the submission path checks.
type q8MoEBuffers struct {
	normQ8, normDS                  *Buffer
	gateW, upW, downW               *Buffer
	counts, assignRows, assignSlots *Buffer
	gate, up, down                  *Buffer
	actQ8, actDS                    *Buffer
}

func allocQ8MoEBuffers(t *testing.T, dev *Device, hidden, ffn, rows, experts, used int64) *q8MoEBuffers {
	t.Helper()
	slots := rows * used
	b := &q8MoEBuffers{}
	sizes := []struct {
		dst   **Buffer
		bytes uint64
	}{
		{&b.normQ8, uint64(rows * hidden)},
		{&b.normDS, uint64(rows * (hidden / 32) * 8)},
		// The weight spans use the widest IQ block the tests exercise: 110
		// bytes per 256 values for IQ3_S, which is also larger than the 66
		// bytes of IQ2_XXS.
		{&b.gateW, uint64(experts * ffn * (hidden / 256) * 110)},
		{&b.upW, uint64(experts * ffn * (hidden / 256) * 110)},
		{&b.downW, uint64(experts * hidden * (ffn / 256) * 110)},
		{&b.counts, uint64(experts * 4)},
		{&b.assignRows, uint64(experts * slots * 4)},
		{&b.assignSlots, uint64(experts * slots * 4)},
		{&b.gate, uint64(slots * ffn * 4)},
		{&b.up, uint64(slots * ffn * 4)},
		{&b.down, uint64(slots * hidden * 4)},
		{&b.actQ8, uint64(slots * ffn)},
		{&b.actDS, uint64(slots * (ffn / 32) * 8)},
	}
	for _, s := range sizes {
		buf, err := dev.Alloc(s.bytes)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = buf.Close() })
		*s.dst = buf
	}
	return b
}

// TestPlanMoEQ8ShapeRules pins the add-time rules of the int8 expert ops: the
// IQ-only dtype set, the 256-value width rule for both the K width and the
// output width, and the minimum activation spans. Every rejection must happen
// before submission so a caller can keep the FP32 route.
func TestPlanMoEQ8ShapeRules(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	const hidden, ffn, rows, experts, used = 2048, 512, 32, 2, 1
	b := allocQ8MoEBuffers(t, dev, hidden, ffn, rows, experts, used)

	gateUp := func(dtype tensor.DType, h, f int64, normQ8, normDS *Buffer) error {
		return plan.AddMoEGateUpQ8(normQ8, normDS, b.gateW, b.upW, b.counts,
			b.assignRows, b.assignSlots, b.gate, b.up, h, f, rows, experts, used, dtype)
	}
	if err := gateUp(tensor.Q4_K, hidden, ffn, b.normQ8, b.normDS); err == nil {
		t.Fatal("Q8 gate/up accepted a Q4_K expert dtype")
	}
	if err := gateUp(tensor.IQ2XXS, hidden+128, ffn, b.normQ8, b.normDS); err == nil {
		t.Fatal("Q8 gate/up accepted a K width that is not a multiple of 256")
	}
	if err := gateUp(tensor.IQ2XXS, hidden, ffn+64, b.normQ8, b.normDS); err == nil {
		t.Fatal("Q8 gate/up accepted an output width that is not a multiple of 256")
	}
	smallQ8, err := dev.Alloc(uint64(rows*hidden) - 1)
	if err != nil {
		t.Fatal(err)
	}
	defer smallQ8.Close()
	if err := gateUp(tensor.IQ2XXS, hidden, ffn, smallQ8, b.normDS); err == nil {
		t.Fatal("Q8 gate/up accepted an undersized int8 activation tile")
	}
	smallDS, err := dev.Alloc(uint64(rows*(hidden/32)*8) - 8)
	if err != nil {
		t.Fatal(err)
	}
	defer smallDS.Close()
	if err := gateUp(tensor.IQ2XXS, hidden, ffn, b.normQ8, smallDS); err == nil {
		t.Fatal("Q8 gate/up accepted an undersized scale tile")
	}

	down := func(dtype tensor.DType, f, h int64, actQ8, actDS *Buffer) error {
		return plan.AddMoEDownQ8(actQ8, actDS, b.downW, b.counts, b.assignRows,
			b.assignSlots, b.down, f, h, rows, experts, used, dtype)
	}
	if err := down(tensor.Q4_K, ffn, hidden, b.actQ8, b.actDS); err == nil {
		t.Fatal("Q8 down accepted a Q4_K expert dtype")
	}
	if err := down(tensor.IQ3S, ffn+128, hidden, b.actQ8, b.actDS); err == nil {
		t.Fatal("Q8 down accepted a K width that is not a multiple of 256")
	}
	smallDownQ8, err := dev.Alloc(uint64(rows*used*ffn) - 1)
	if err != nil {
		t.Fatal(err)
	}
	defer smallDownQ8.Close()
	if err := down(tensor.IQ3S, ffn, hidden, smallDownQ8, b.actDS); err == nil {
		t.Fatal("Q8 down accepted an undersized int8 activation tile")
	}
	smallDownDS, err := dev.Alloc(uint64(rows*used*(ffn/32)*8) - 8)
	if err != nil {
		t.Fatal(err)
	}
	defer smallDownDS.Close()
	if err := down(tensor.IQ3S, ffn, hidden, b.actQ8, smallDownDS); err == nil {
		t.Fatal("Q8 down accepted an undersized scale tile")
	}

	if plan.OpCount() != 0 {
		t.Fatalf("rejected operations were added: %d", plan.OpCount())
	}

	// The shapes the model uses must be accepted and reach the plan.
	if err := gateUp(tensor.IQ2XXS, hidden, ffn, b.normQ8, b.normDS); err != nil {
		t.Fatal(err)
	}
	if err := down(tensor.IQ3S, ffn, hidden, b.actQ8, b.actDS); err != nil {
		t.Fatal(err)
	}
	if got := plan.OpCount(); got != 2 {
		t.Fatalf("plan holds %d operations, want 2", got)
	}
	ops := plan.Ops()
	if ops[0].Code != OpMoEGateUpQ8 || ops[0].N[0] != hidden || ops[0].N[1] != ffn {
		t.Fatalf("unexpected gate/up op: %+v", ops[0])
	}
	if ops[1].Code != OpMoEDownQ8 || ops[1].N[0] != ffn || ops[1].N[1] != hidden {
		t.Fatalf("unexpected down op: %+v", ops[1])
	}
}
