//go:build hip && linux

package hip

import (
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

// mmqMoEBuffers is the buffer set the int8 WMMA expert route needs, sized for
// the given shapes. The buffers are real device allocations so p.check sees the
// same bounds the submission path checks.
type mmqMoEBuffers struct {
	normQ8, normDS                  *Buffer
	gateW, upW, downW               *Buffer
	counts, assignRows, assignSlots *Buffer
	gate, up, down                  *Buffer
	actQ8, actDS                    *Buffer
}

func allocMMQMoEBuffers(t *testing.T, dev *Device, hidden, ffn, rows, experts, used int64) *mmqMoEBuffers {
	t.Helper()
	slots := rows * used
	b := &mmqMoEBuffers{}
	sizes := []struct {
		dst   **Buffer
		bytes uint64
	}{
		{&b.normQ8, uint64(rows * hidden)},
		{&b.normDS, uint64(rows * (hidden / 32) * 8)},
		// The weight spans use the widest K-quant block the tests exercise: 210
		// bytes per 256 values for Q6_K.
		{&b.gateW, uint64(experts * ffn * (hidden / 256) * 210)},
		{&b.upW, uint64(experts * ffn * (hidden / 256) * 210)},
		{&b.downW, uint64(experts * hidden * (ffn / 256) * 210)},
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

// TestPlanMoEMMQShapeRules pins the add-time rules of the int8 WMMA expert ops:
// the K-quant dtype set, the 128-value width rule for the K width and the
// output width, the measured assignment tile widths, and the minimum activation
// spans. Every rejection must happen before submission so a caller can keep the
// dp4a or FP32 route.
func TestPlanMoEMMQShapeRules(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	const hidden, ffn, rows, experts, used = 2048, 512, 32, 2, 1
	b := allocMMQMoEBuffers(t, dev, hidden, ffn, rows, experts, used)

	gateUp := func(dtype tensor.DType, h, f, tileJ int64) error {
		return plan.AddMoEGateUpMMQ(b.normQ8, b.normDS, b.gateW, b.upW, b.counts,
			b.assignRows, b.assignSlots, b.gate, b.up, h, f, rows, experts, used,
			tileJ, dtype)
	}
	if err := gateUp(tensor.IQ2XXS, hidden, ffn, 16); err == nil {
		t.Fatal("MMQ gate/up accepted an IQ expert dtype")
	}
	if err := gateUp(tensor.F32, hidden, ffn, 16); err == nil {
		t.Fatal("MMQ gate/up accepted an F32 expert dtype")
	}
	if err := gateUp(tensor.Q4_K, hidden+64, ffn, 16); err == nil {
		t.Fatal("MMQ gate/up accepted a K width that is not a multiple of 128")
	}
	if err := gateUp(tensor.Q4_K, hidden, ffn+64, 16); err == nil {
		t.Fatal("MMQ gate/up accepted an output width that is not a multiple of 128")
	}
	if err := gateUp(tensor.Q4_K, hidden, ffn, 24); err == nil {
		t.Fatal("MMQ gate/up accepted an unmeasured assignment tile")
	}
	smallQ8, err := dev.Alloc(uint64(rows*hidden) - 1)
	if err != nil {
		t.Fatal(err)
	}
	defer smallQ8.Close()
	if err := plan.AddMoEGateUpMMQ(smallQ8, b.normDS, b.gateW, b.upW, b.counts,
		b.assignRows, b.assignSlots, b.gate, b.up, hidden, ffn, rows, experts, used,
		16, tensor.Q4_K); err == nil {
		t.Fatal("MMQ gate/up accepted an undersized int8 activation tile")
	}

	down := func(dtype tensor.DType, f, h, tileJ int64) error {
		return plan.AddMoEDownMMQ(b.actQ8, b.actDS, b.downW, b.counts,
			b.assignRows, b.assignSlots, b.down, f, h, rows, experts, used, tileJ,
			dtype)
	}
	if err := down(tensor.IQ3S, ffn, hidden, 16); err == nil {
		t.Fatal("MMQ down accepted an IQ expert dtype")
	}
	if err := down(tensor.Q6_K, ffn+64, hidden, 16); err == nil {
		t.Fatal("MMQ down accepted a K width that is not a multiple of 128")
	}
	if err := down(tensor.Q6_K, ffn, hidden, 20); err == nil {
		t.Fatal("MMQ down accepted an unmeasured assignment tile")
	}
	smallDownDS, err := dev.Alloc(uint64(rows*used*(ffn/32)*8) - 8)
	if err != nil {
		t.Fatal(err)
	}
	defer smallDownDS.Close()
	if err := plan.AddMoEDownMMQ(b.actQ8, smallDownDS, b.downW, b.counts,
		b.assignRows, b.assignSlots, b.down, ffn, hidden, rows, experts, used, 16,
		tensor.Q6_K); err == nil {
		t.Fatal("MMQ down accepted an undersized scale tile")
	}

	if plan.OpCount() != 0 {
		t.Fatalf("rejected operations were added: %d", plan.OpCount())
	}

	// The shapes the model uses must be accepted and reach the plan with the
	// assignment tile width in n[5].
	if err := gateUp(tensor.Q4_K, hidden, ffn, 32); err != nil {
		t.Fatal(err)
	}
	if err := down(tensor.Q6_K, ffn, hidden, 64); err != nil {
		t.Fatal(err)
	}
	if got := plan.OpCount(); got != 2 {
		t.Fatalf("plan holds %d operations, want 2", got)
	}
	ops := plan.Ops()
	if ops[0].Code != OpMoEGateUpMMQ || ops[0].N[0] != hidden || ops[0].N[1] != ffn ||
		ops[0].N[5] != 32 || ops[0].U[0] != uint32(tensor.Q4_K) {
		t.Fatalf("unexpected gate/up op: %+v", ops[0])
	}
	if ops[1].Code != OpMoEDownMMQ || ops[1].N[0] != ffn || ops[1].N[1] != hidden ||
		ops[1].N[5] != 64 || ops[1].U[0] != uint32(tensor.Q6_K) {
		t.Fatalf("unexpected down op: %+v", ops[1])
	}
}
