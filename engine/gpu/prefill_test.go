//go:build hip && linux

package gpu

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func tinyPrefillFixture(t *testing.T, capacity int) (*gguf.Reader, *qwen35.Model, *tokenizer.Tokenizer, []int32) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
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
	ids, err := tok.Encode(strings.Repeat("Hello world. ", 8))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 40 {
		ids = append(ids, make([]int32, 40-len(ids))...)
	}
	ids = ids[:40]
	if capacity < len(ids)+8 {
		capacity = len(ids) + 8
	}
	return reader, model, tok, ids
}

func maxDiff(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > worst {
			worst = diff
		}
	}
	return worst
}

func top1(values []float32) int {
	best := 0
	for i, v := range values {
		if v > values[best] {
			best = i
		}
	}
	return best
}

// TestTinyGPUChunkedPrefillMatchesCPU checks that chunk sizes that select the
// batched GEMV and WMMA paths both reproduce the CPU reference logits and
// greedy token, including partial final chunks.
func TestTinyGPUChunkedPrefillMatchesCPU(t *testing.T) {
	const capacity = 64
	reader, model, _, ids := tinyPrefillFixture(t, capacity)
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

	for _, chunk := range []int{1, 3, 7, 16, 17, 33, 64} {
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

// TestTinyGPUChunkedDecodeMatchesCPU verifies that the KV and recurrent state
// after chunked prefill produce the same next-token logits as an unchunked CPU
// prefill followed by one decode step.
func TestTinyGPUChunkedDecodeMatchesCPU(t *testing.T) {
	const capacity = 64
	reader, model, _, ids := tinyPrefillFixture(t, capacity)
	next := ids[len(ids)-1]
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Forward(ids, cpuState); err != nil {
		t.Fatal(err)
	}
	cpuStep, err := model.Step(next, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := New(reader, model, capacity)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	for _, chunk := range []int{5, 16, 33} {
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatal(err)
		}
		rt.Reset()
		if _, err := rt.Forward(ids); err != nil {
			t.Fatalf("chunk %d prefill: %v", chunk, err)
		}
		got, err := rt.Step(next)
		if err != nil {
			t.Fatalf("chunk %d step: %v", chunk, err)
		}
		diff := maxDiff(cpuStep, got)
		if top1(cpuStep) != top1(got) {
			t.Fatalf("chunk %d: decode top-1 differs (max diff %.6g)", chunk, diff)
		}
		if diff > 2e-3 {
			t.Fatalf("chunk %d: decode max diff %.6g exceeds 2e-3", chunk, diff)
		}
	}
}

// TestPrefillChunkValidation rejects non-positive sizes and clamps oversized
// requests to the context capacity.
func TestPrefillChunkValidation(t *testing.T) {
	const capacity = 8
	reader, model, _, _ := tinyPrefillFixture(t, capacity)
	rt, err := New(reader, model, capacity)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(0); err == nil {
		t.Fatal("expected an error for chunk size 0")
	}
	if err := rt.SetPrefillChunk(-3); err == nil {
		t.Fatal("expected an error for a negative chunk size")
	}
	if err := rt.SetPrefillChunk(1 << 20); err != nil {
		t.Fatalf("oversized chunk should clamp: %v", err)
	}
	if rt.prefillRows != capacity {
		t.Fatalf("clamped chunk rows = %d, want %d", rt.prefillRows, capacity)
	}
}
