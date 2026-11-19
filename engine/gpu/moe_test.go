//go:build hip && linux

package gpu

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// tinyMoEFixture writes the F32 tiny qwen35moe model, which exercises the
// router, the routed expert projections, and the shared expert through the same
// planner path as a real MoE GGUF.
func tinyMoEFixture(t *testing.T) (*gguf.Reader, *qwen35.Model, *tokenizer.Tokenizer) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-qwen35moe.gguf")
	if err := fixture.WriteTinyQwen35MoE(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	return reader, model, tok
}

// TestTinyGPUMoEMatchesCPU checks the decode graph of an MoE model end to end:
// routing, the selected-expert kernels, the shared expert, and the residual
// accumulation must reproduce the CPU reference logits, before and after a
// reset that rebuilds the plan from scratch.
func TestTinyGPUMoEMatchesCPU(t *testing.T) {
	const capacity = 16
	reader, model, tok := tinyMoEFixture(t)
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := New(reader, model, capacity)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()

	for pass := range 2 {
		if pass > 0 {
			rt.Reset()
		}
		got, err := rt.Forward(ids)
		if err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
		diff := maxDiff(cpuLogits, got)
		if top1(cpuLogits) != top1(got) {
			t.Fatalf("pass %d: top-1 token differs: cpu %d gpu %d (max diff %.6g)",
				pass, top1(cpuLogits), top1(got), diff)
		}
		if diff > 2e-3 {
			t.Fatalf("pass %d: max logit diff %.6g exceeds 2e-3", pass, diff)
		}
	}
}

// TestTinyGPUChunkedPrefillMoEMatchesCPU checks the grouped expert prefill path,
// including a partial final chunk, for chunk sizes that select the batched GEMV
// and WMMA routes for the router projection.
func TestTinyGPUChunkedPrefillMoEMatchesCPU(t *testing.T) {
	reader, model, tok := tinyMoEFixture(t)
	ids, err := tok.Encode(strings.Repeat("Hello world. ", 8))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 40 {
		ids = append(ids, make([]int32, 40-len(ids))...)
	}
	ids = ids[:40]
	capacity := len(ids) + 8
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := New(reader, model, capacity)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	for _, chunk := range []int{3, 8} {
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		rt.Reset()
		got, err := rt.Forward(ids)
		if err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		diff := maxDiff(cpuLogits, got)
		if top1(cpuLogits) != top1(got) {
			t.Fatalf("chunk %d: top-1 token differs: cpu %d gpu %d (max diff %.6g)",
				chunk, top1(cpuLogits), top1(got), diff)
		}
		if diff > 2e-3 {
			t.Fatalf("chunk %d: max logit diff %.6g exceeds 2e-3", chunk, diff)
		}
	}
}
