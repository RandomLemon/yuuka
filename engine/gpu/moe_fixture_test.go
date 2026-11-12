//go:build hip && linux

package gpu

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/dense"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func loadTinyMoEGPU(t *testing.T) (*gguf.Reader, *dense.Model, *tokenizer.Tokenizer, []int32) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-qwen3moe.gguf")
	if err := fixture.WriteTinyMoE(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	model, err := dense.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	return reader, model, tok, ids
}

func newTinyMoESession(t *testing.T, reader *gguf.Reader, model *dense.Model, format tensor.DType) (*MoEModel, *MoESession) {
	t.Helper()
	mo, err := OpenMoE(reader, model, Options{
		Blocks:       2,
		BlockSize:    16,
		PrefillChunk: 8,
		KVType:       format,
	})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	t.Cleanup(func() { _ = mo.Close() })
	table := memory.NewTable(mo.Allocator())
	if err := table.Cover(32); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(table.Release)
	session, err := mo.NewMoESession(table, 0, 32)
	if err != nil {
		t.Fatal(err)
	}
	return mo, session
}

func TestTinyMoEGPUMatchesCPU(t *testing.T) {
	reader, model, _, ids := loadTinyMoEGPU(t)
	cpuState, err := dense.NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}
	for _, format := range []tensor.DType{tensor.F32, tensor.F16, tensor.Q8_0} {
		t.Run(format.String(), func(t *testing.T) {
			_, session := newTinyMoESession(t, reader, model, format)
			defer session.Close()
			logits, err := session.Prefill(ids, true)
			if err != nil {
				t.Fatal(err)
			}
			maxDiff := 0.0
			for i := range cpuLogits {
				diff := math.Abs(float64(cpuLogits[i] - logits[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			if maxDiff > 2e-3 {
				t.Fatalf("MoE GPU/CPU logits differ by %.6g", maxDiff)
			}
			if argmax(cpuLogits) != argmax(logits) {
				t.Fatalf("MoE top-1 differs: cpu=%d gpu=%d",
					argmax(cpuLogits), argmax(logits))
			}
			t.Logf("%s prefill max logit diff %.6g", format, maxDiff)
		})
	}
}

func TestTinyMoELayerTrace(t *testing.T) {
	reader, model, _, ids := loadTinyMoEGPU(t)
	cpuState, err := dense.NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	cpuTraces := make(map[int][]float32)
	if _, err := model.ForwardTrace(ids[:1], cpuState, func(layer int, x []float32) {
		cpuTraces[layer] = append([]float32(nil), x...)
	}); err != nil {
		t.Fatal(err)
	}
	mo, err := OpenMoE(reader, model, Options{Blocks: 2, BlockSize: 16, PrefillChunk: 8})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer mo.Close()
	table := memory.NewTable(mo.Allocator())
	if err := table.Cover(32); err != nil {
		t.Fatal(err)
	}
	defer table.Release()
	session, err := mo.NewMoESessionTrace(table, 0, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	_, gpuTraces, err := session.ForwardTrace(ids[:1])
	if err != nil {
		t.Fatal(err)
	}
	for layer := range model.Config.BlockCount {
		maxDiff := 0.0
		for i := range cpuTraces[layer] {
			maxDiff = math.Max(maxDiff, math.Abs(float64(cpuTraces[layer][i]-gpuTraces[layer][i])))
		}
		if maxDiff > 2e-3 {
			t.Fatalf("layer %d hidden state differs by %.6g", layer, maxDiff)
		}
		t.Logf("layer %d hidden diff %.6g", layer, maxDiff)
	}
}

func TestTinyMoEQuantizedGPUMatchesCPU(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen3moe-q4k.gguf")
	if err := fixture.WriteTinyMoEQuantized(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	model, err := dense.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	if model.Layers[0].GateExps.DType != tensor.Q4_K {
		t.Fatalf("expert dtype %s, want Q4_K", model.Layers[0].GateExps.DType)
	}
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	seq := make([]int32, 0, 9)
	for len(seq) < 9 {
		seq = append(seq, ids...)
	}
	seq = seq[:9]

	cpuState, err := dense.NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(seq, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	mo, err := OpenMoE(reader, model, Options{
		Blocks:       4,
		BlockSize:    16,
		PrefillChunk: 3,
	})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	t.Cleanup(func() { _ = mo.Close() })
	newSession := func() *MoESession {
		table := memory.NewTable(mo.Allocator())
		if err := table.Cover(32); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(table.Release)
		session, err := mo.NewMoESession(table, 0, 32)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = session.Close() })
		return session
	}
	compare := func(name string, got []float32) {
		maxDiff := 0.0
		for i := range cpuLogits {
			maxDiff = math.Max(maxDiff, math.Abs(float64(cpuLogits[i]-got[i])))
		}
		if maxDiff > 5e-3 {
			t.Fatalf("%s logits differ from CPU by %.6g", name, maxDiff)
		}
		if argmax(cpuLogits) != argmax(got) {
			t.Fatalf("%s top-1 differs: cpu=%d gpu=%d",
				name, argmax(cpuLogits), argmax(got))
		}
		t.Logf("%s max logit diff %.6g", name, maxDiff)
	}

	session := newSession()
	gpuLogits, err := session.Prefill(seq, true)
	if err != nil {
		t.Fatal(err)
	}
	compare("Q4_K chunked prefill", gpuLogits)

	session = newSession()
	if _, err := session.Prefill(seq[:8], false); err != nil {
		t.Fatal(err)
	}
	gpuDecode, err := session.Decode(seq[8])
	if err != nil {
		t.Fatal(err)
	}
	compare("Q4_K prefill plus decode", gpuDecode)
}

func TestTinyMoEAttentionPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen3moe-zero.gguf")
	if err := fixture.WriteTinyMoEZeroExperts(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	model, err := dense.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	cpuState, err := dense.NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids[:1], cpuState)
	if err != nil {
		t.Fatal(err)
	}
	mo, err := OpenMoE(reader, model, Options{Blocks: 2, BlockSize: 16, PrefillChunk: 8})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer mo.Close()
	table := memory.NewTable(mo.Allocator())
	if err := table.Cover(32); err != nil {
		t.Fatal(err)
	}
	defer table.Release()
	session, err := mo.NewMoESession(table, 0, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	gpuLogits, err := session.Decode(ids[0])
	if err != nil {
		t.Fatal(err)
	}
	maxDiff := 0.0
	for i := range cpuLogits {
		maxDiff = math.Max(maxDiff, math.Abs(float64(cpuLogits[i]-gpuLogits[i])))
	}
	if maxDiff > 2e-3 {
		t.Fatalf("attention path differs by %.6g", maxDiff)
	}
	t.Logf("attention-only max logit diff %.6g", maxDiff)
}
