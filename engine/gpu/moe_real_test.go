//go:build hip && linux

package gpu

import (
	"os"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// openMoEModel loads the opt-in qwen35moe GGUF named by YUUKA_TEST_MOE_MODEL.
// The real-model tests load 10.7 GB of weights and run the CPU reference, so
// they stay behind the environment variable like the CPU MoE golden test.
func openMoEModel(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *qwen35.Model) {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MOE_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MOE_MODEL is not set")
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
	if !model.Config.IsMoE() {
		t.Skipf("model architecture %q is not a MoE architecture",
			model.Config.Architecture)
	}
	return reader, tok, model
}

// TestGPUMoEGoldenGeneration checks the real qwen35moe model end to end: the
// first-token logits must match the CPU reference, and the four greedy tokens
// must match both the CPU reference and llama.cpp's golden continuation. An F32
// KV pool isolates the expert arithmetic from KV storage rounding.
func TestGPUMoEGoldenGeneration(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode("The capital of France is")
	if err != nil {
		t.Fatal(err)
	}
	const capacity = 64
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := NewWithKVType(reader, model, capacity, tensor.F32)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	gpuLogits, err := rt.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	diff := maxDiff(cpuLogits, gpuLogits)
	if top1(cpuLogits) != top1(gpuLogits) {
		t.Fatalf("first token differs: cpu %d gpu %d (max logit diff %.6g)",
			top1(cpuLogits), top1(gpuLogits), diff)
	}
	if diff > 0.1 {
		t.Fatalf("max logit diff %.6g exceeds 0.1", diff)
	}

	gpuTokens := make([]int32, 0, 4)
	for i := range 4 {
		best := int32(top1(gpuLogits))
		gpuTokens = append(gpuTokens, best)
		if i+1 < 4 {
			gpuLogits, err = rt.Step(best)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	cpuTokens := make([]int32, 0, 4)
	cpuNext := cpuLogits
	for i := range 4 {
		best := int32(top1(cpuNext))
		cpuTokens = append(cpuTokens, best)
		if i+1 < 4 {
			cpuNext, err = model.Step(best, cpuState)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	want := []int32{11751, 11, 264, 3177} // " Paris, a city"
	if !equalTokens(cpuTokens, want) {
		t.Fatalf("cpu tokens %v, want %v", cpuTokens, want)
	}
	if !equalTokens(gpuTokens, want) {
		t.Fatalf("gpu tokens %v, want %v (max first-logit diff %.6g)",
			gpuTokens, want, diff)
	}
	t.Logf("max first-logit diff %.6g, tokens %v", diff, gpuTokens)
}

// prefillLogitTolerance bounds the chunked-prefill logit difference against the
// CPU reference. Prefill agreement is gated on token behavior, not on a tight
// numerical bound: the engine's chunked prefill drifts from the CPU reference as
// the prefix grows even for the dense model (recorded 0.006 at 8 tokens and 0.57
// at 128 tokens on Qwen3.5-4B), and an MoE model amplifies that drift through
// near-tie routing: perturbing a router weight by 1e-4 relative moves this
// model's own logits by 0.64. The recorded value for the 256-token prompt below
// is 0.96 at a 64-row chunk and 1.04 at a 256-row chunk, with a matching top-1
// token and the same four greedy continuation tokens.
const prefillLogitTolerance = 2.0

// TestGPUMoEChunkedPrefill checks the grouped expert prefill path: the
// next-token logits must keep the CPU top-1 token, and the four greedy tokens
// decoded from a 256-token prompt must match the CPU continuation.
func TestGPUMoEChunkedPrefill(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 256 {
		t.Skipf("prompt produced %d tokens, need 256", len(ids))
	}
	ids = ids[:256]
	const capacity = 256 + 8
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := NewWithKVType(reader, model, capacity, tensor.F32)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	var gpuLogits []float32
	for _, chunk := range []int{64, 256} {
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		rt.Reset()
		gpuLogits, err = rt.Forward(ids)
		if err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		diff := maxDiff(cpuLogits, gpuLogits)
		if top1(cpuLogits) != top1(gpuLogits) {
			t.Fatalf("chunk %d: top-1 token differs: cpu %d gpu %d (max diff %.6g)",
				chunk, top1(cpuLogits), top1(gpuLogits), diff)
		}
		if diff > prefillLogitTolerance {
			t.Fatalf("chunk %d: max logit diff %.6g exceeds %.1f",
				chunk, diff, prefillLogitTolerance)
		}
		t.Logf("chunk %d: max logit diff %.6g", chunk, diff)
	}

	// The behavioral gate: four greedy tokens after the chunked prefill must
	// match the CPU continuation from the same prompt.
	const continuation = 4
	gpuTokens := make([]int32, 0, continuation)
	for i := range continuation {
		best := int32(top1(gpuLogits))
		gpuTokens = append(gpuTokens, best)
		if i+1 < continuation {
			gpuLogits, err = rt.Step(best)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	cpuTokens := make([]int32, 0, continuation)
	next := cpuLogits
	for i := range continuation {
		best := int32(top1(next))
		cpuTokens = append(cpuTokens, best)
		if i+1 < continuation {
			next, err = model.Step(best, cpuState)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if !equalTokens(gpuTokens, cpuTokens) {
		t.Fatalf("chunked prefill continuation %v, cpu %v", gpuTokens, cpuTokens)
	}
	t.Logf("chunked prefill continuation %v", gpuTokens)
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
