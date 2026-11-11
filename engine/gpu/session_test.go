//go:build hip && linux

package gpu

import (
	"testing"

	"github.com/yuuka-ai/yuuka/memory"
)

func TestGraphFallbackOnDynamicFailure(t *testing.T) {
	const capacity = 64
	reader, model, _, ids := tinyPrefillFixture(t, capacity)
	m, err := Open(reader, model, Options{
		Blocks:        16,
		BlockSize:     8,
		PrefillChunk:  16,
		EnableGraph:   true,
		GraphMinSteps: 1,
	})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer m.Close()

	newSession := func(graph bool) (*Session, *memory.Table, []float32) {
		t.Helper()
		table := memory.NewTable(m.Allocator())
		if err := table.Cover(capacity); err != nil {
			t.Fatal(err)
		}
		session, err := m.NewSession(table, 0, nil, SessionOptions{
			Capacity:      capacity,
			EnableGraph:   graph,
			GraphMinSteps: 1,
		})
		if err != nil {
			table.Release()
			t.Fatal(err)
		}
		logits, err := session.Prefill(ids, true)
		if err != nil {
			session.Close()
			table.Release()
			t.Fatal(err)
		}
		return session, table, append([]float32(nil), logits...)
	}

	graphed, table, graphPrefill := newSession(true)
	defer func() {
		_ = graphed.Close()
		table.Release()
	}()
	reference, refTable, refPrefill := newSession(false)
	defer func() {
		_ = reference.Close()
		refTable.Release()
	}()

	first := int32(top1(graphPrefill))
	if int32(top1(refPrefill)) != first {
		t.Fatalf("prefill tokens differ")
	}

	graphedLogits, err := graphed.Decode(first)
	if err != nil {
		t.Fatal(err)
	}
	if !graphed.GraphActive() {
		t.Fatal("decode graph was not captured")
	}
	refLogits, err := reference.Decode(first)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxDiff(graphedLogits, refLogits); diff > 2e-3 {
		t.Fatalf("graph replay logits differ by %g", diff)
	}
	second := int32(top1(graphedLogits))

	// Invalidate the dynamic parameter buffer. Replay must be abandoned and the
	// stream fallback must still match a plain session.
	if err := graphed.dyn.Close(); err != nil {
		t.Fatal(err)
	}
	fallbackLogits, err := graphed.Decode(second)
	if err != nil {
		t.Fatalf("fallback decode: %v", err)
	}
	if graphed.GraphActive() {
		t.Fatal("graph remained active after a failed replay")
	}
	refStepLogits, err := reference.Decode(second)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxDiff(fallbackLogits, refStepLogits); diff > 2e-3 {
		t.Fatalf("fallback logits differ by %g", diff)
	}
	if top1(fallbackLogits) != top1(refStepLogits) {
		t.Fatal("fallback produced a different token")
	}
}

func TestSessionBuffersAreNotRetainedByModel(t *testing.T) {
	const capacity = 48
	reader, model, _, ids := tinyPrefillFixture(t, capacity)
	m, err := Open(reader, model, Options{
		Blocks:        8,
		BlockSize:     8,
		PrefillChunk:  16,
		EnableGraph:   false,
		GraphMinSteps: 1,
	})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer m.Close()
	before := m.Allocations()
	for cycle := range 3 {
		table := memory.NewTable(m.Allocator())
		if err := table.Cover(capacity); err != nil {
			t.Fatal(err)
		}
		session, err := m.NewSession(table, 0, nil, SessionOptions{Capacity: capacity})
		if err != nil {
			table.Release()
			t.Fatal(err)
		}
		logits, err := session.Prefill(ids, true)
		if err != nil {
			session.Close()
			table.Release()
			t.Fatal(err)
		}
		if _, err := session.Decode(int32(top1(logits))); err != nil {
			session.Close()
			table.Release()
			t.Fatal(err)
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
		table.Release()
		if after := m.Allocations(); after != before {
			t.Fatalf("cycle %d grew model allocation list from %d to %d", cycle, before, after)
		}
	}
}
