//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func goldenPrompt(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *qwen35.Model, []int32) {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
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
	ids, err := tok.Encode("The capital of France is")
	if err != nil {
		t.Fatal(err)
	}
	return reader, tok, model, ids
}

func TestGPUMatchesCPU(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()

	// CPU reference logits.
	cpuState, err := qwen35.NewState(model.Config, 64)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(prompt, cpuState)
	if err != nil {
		t.Fatal(err)
	}
	gpuLogits, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	maxDiff := 0.0
	for i := range cpuLogits {
		diff := math.Abs(float64(cpuLogits[i] - gpuLogits[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 5.0 {
		t.Fatalf("GPU/CPU logits diverge: max abs diff %.3f", maxDiff)
	}
	cpuTop := argmax(cpuLogits)
	gpuTop := argmax(gpuLogits)
	if cpuTop != gpuTop {
		t.Fatalf("first token differs: cpu=%d gpu=%d (max logit diff %.3f)", cpuTop, gpuTop, maxDiff)
	}
	t.Logf("first step max logit diff %.6g", maxDiff)

	// Generate greedily on both paths.
	cpuTokens := make([]int32, 0, 6)
	gpuTokens := make([]int32, 0, 6)
	cpuLogitsCopy := append([]float32(nil), cpuLogits...)
	for i := 0; i < 6; i++ {
		cpuToken := argmax(cpuLogitsCopy)
		cpuTokens = append(cpuTokens, cpuToken)
		gpuToken := argmax(gpuLogits)
		gpuTokens = append(gpuTokens, gpuToken)
		if i+1 < 6 {
			cpuLogitsCopy, err = model.Step(cpuToken, cpuState)
			if err != nil {
				t.Fatal(err)
			}
			gpuLogits, err = rt.Step(gpuToken)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	want := []int32{11751, 13, 198, 32, 13, 2912}
	if !reflect.DeepEqual(cpuTokens, want) {
		t.Fatalf("CPU tokens %v, want %v", cpuTokens, want)
	}
	if !reflect.DeepEqual(gpuTokens, want) {
		t.Fatalf("GPU tokens %v, want %v", gpuTokens, want)
	}
}

func TestGPUReset(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	rt, err := New(reader, model, 32)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	first, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	firstToken := argmax(first)
	rt.Reset()
	second, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if argmax(second) != firstToken {
		t.Fatalf("reset changed output: %d vs %d", firstToken, argmax(second))
	}
}

func argmax(values []float32) int32 {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return int32(best)
}
