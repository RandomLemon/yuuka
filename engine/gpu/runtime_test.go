//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"reflect"
	"sort"
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

func TestGPUGoldenPrompts(t *testing.T) {
	reader, tok, model, _ := goldenPrompt(t)
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	cases := []struct {
		prompt string
		want   []int32
	}{
		{"The capital of France is", []int32{11751, 13, 198, 32, 13, 2912}},
		{"1+1=", []int32{17, 3709, 17, 10, 17, 28}},
		{"Once upon a time", []int32{11, 303, 264, 1814, 2400, 314}},
	}
	for _, tc := range cases {
		rt.Reset()
		ids, err := tok.Encode(tc.prompt)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := rt.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]int32, 0, len(tc.want))
		for i := 0; i < len(tc.want); i++ {
			token := argmax(logits)
			got = append(got, token)
			if i+1 < len(tc.want) {
				logits, err = rt.Step(token)
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%q: tokens %v, want %v", tc.prompt, got, tc.want)
		}
	}
}

func TestGPULayerTraceMatchesCPU(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	cpuState, err := qwen35.NewState(model.Config, 64)
	if err != nil {
		t.Fatal(err)
	}
	cpuTraces := make(map[int][]float32)
	if _, err := model.ForwardTrace(prompt, cpuState, func(layer int, x []float32) {
		cpuTraces[layer] = append([]float32(nil), x...)
	}); err != nil {
		t.Fatal(err)
	}
	_, gpuTraces, err := rt.ForwardTrace(prompt)
	if err != nil {
		t.Fatal(err)
	}
	worstLayer := -1
	worstDiff := 0.0
	for layer := range model.Config.BlockCount {
		want := cpuTraces[layer]
		got := gpuTraces[layer]
		maxDiff := 0.0
		for i := range want {
			diff := math.Abs(float64(want[i] - got[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff > worstDiff {
			worstDiff = maxDiff
			worstLayer = layer
		}
		if maxDiff > 0.1 {
			t.Fatalf("layer %d hidden state differs by %.6g", layer, maxDiff)
		}
	}
	t.Logf("worst layer %d max abs hidden diff %.6g", worstLayer, worstDiff)
}

func TestGPUProfileDecode(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	logits, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	token := argmax(logits)
	times, err := rt.ProfileStep(token)
	if err != nil {
		t.Fatal(err)
	}
	if len(times) == 0 {
		t.Fatal("profile returned no operations")
	}
	names := map[uint32]string{
		1: "gemv", 2: "rmsnorm", 3: "l2norm", 4: "gated_rmsnorm",
		5: "rope", 6: "attention", 7: "conv1d", 8: "gdn", 9: "silu_mul",
		10: "add", 11: "copy", 12: "scale", 13: "embedding",
		14: "kv_write", 15: "rmsnorm_heads", 16: "gemm",
	}
	totals := make(map[uint32]uint64)
	var total uint64
	for _, item := range times {
		totals[item.Code] += item.Nanos
		total += item.Nanos
	}
	type row struct {
		code uint32
		ns   uint64
	}
	rows := make([]row, 0, len(totals))
	for code, ns := range totals {
		rows = append(rows, row{code, ns})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ns > rows[j].ns })
	t.Logf("profiled %d ops, total %.3f ms", len(times), float64(total)/1e6)
	for i, item := range rows {
		if i == 10 {
			break
		}
		name := names[item.code]
		if name == "" {
			name = "unknown"
		}
		t.Logf("%-14s %8.3f ms  %5.1f%%", name, float64(item.ns)/1e6,
			100*float64(item.ns)/float64(total))
	}
}
