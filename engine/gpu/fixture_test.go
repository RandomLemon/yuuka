//go:build hip && linux

package gpu

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestTinyGPUMatchesCPU(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}

	cpuState, err := qwen35.NewState(model.Config, 16)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	rt, err := New(reader, model, 16)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	gpuLogits, err := rt.Forward(ids)
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
	if maxDiff > 1e-3 {
		t.Fatalf("tiny fixture GPU/CPU max abs diff %.6g", maxDiff)
	}
	if argmax(cpuLogits) != argmax(gpuLogits) {
		t.Fatalf("tiny fixture top-1 differs: cpu=%d gpu=%d",
			argmax(cpuLogits), argmax(gpuLogits))
	}

	rt.Reset()
	again, err := rt.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	if argmax(again) != argmax(gpuLogits) {
		t.Fatal("tiny fixture GPU reset changed the top-1 token")
	}
}

func TestTinyGPULayerTrace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	cpuState, err := qwen35.NewState(model.Config, 16)
	if err != nil {
		t.Fatal(err)
	}
	cpuTraces := make(map[int][]float32)
	if _, err := model.ForwardTrace(ids, cpuState, func(layer int, x []float32) {
		cpuTraces[layer] = append([]float32(nil), x...)
	}); err != nil {
		t.Fatal(err)
	}
	rt, err := New(reader, model, 16)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	_, gpuTraces, err := rt.ForwardTrace(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpuTraces) != model.Config.BlockCount {
		t.Fatalf("GPU trace count %d, want %d", len(gpuTraces), model.Config.BlockCount)
	}
	for layer := range model.Config.BlockCount {
		want := cpuTraces[layer]
		got := gpuTraces[layer]
		if len(want) != len(got) {
			t.Fatalf("layer %d length %d vs %d", layer, len(got), len(want))
		}
		maxDiff := 0.0
		for i := range want {
			diff := math.Abs(float64(want[i] - got[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff > 1e-3 {
			t.Fatalf("layer %d hidden state differs by %.6g", layer, maxDiff)
		}
		t.Logf("layer %d max abs diff %.6g", layer, maxDiff)
	}
}
