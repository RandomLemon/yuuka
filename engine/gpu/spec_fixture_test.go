//go:build hip && linux

package gpu

import (
	"context"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestTinyGPUPrefillLogits(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spec-qwen35.gguf")
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
	rt, err := New(reader, model, 32)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rt.Forward(ids[:1]); err != nil {
		t.Fatal(err)
	}
	candidates := ids[1:]
	all, err := rt.PrefillLogits(candidates)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != len(candidates) {
		t.Fatalf("got %d logits rows, want %d", len(all), len(candidates))
	}
	cpuState, err := qwen35.NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := model.PrefillLogits(append([]int32{ids[0]}, candidates...), cpuState)
	if err != nil {
		t.Fatal(err)
	}
	for row := range all {
		maxDiff := 0.0
		for i := range cpu[row+1] {
			maxDiff = math.Max(maxDiff, math.Abs(float64(cpu[row+1][i]-all[row][i])))
		}
		if maxDiff > 5e-3 {
			t.Fatalf("row %d logits differ from CPU by %.6g", row, maxDiff)
		}
	}
}

func TestTinyGPUSpeculativeMatchesGreedy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spec-qwen35.gguf")
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
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	runner, err := engine.NewWithRuntime(rt, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.Encode("aaaaaaaaaaaaaaaa")
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := runner.Generate(context.Background(), prompt, engine.GenerateOptions{
		MaxTokens: 16,
		IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	speculative, err := runner.Generate(context.Background(), prompt, engine.GenerateOptions{
		MaxTokens: 16,
		IgnoreEOS: true,
		Speculative: &engine.SpeculativeOptions{
			MaxDraft: 4,
			MinMatch: 2,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(speculative.Tokens, baseline.Tokens) {
		t.Fatalf("speculative tokens %v, want %v", speculative.Tokens, baseline.Tokens)
	}
	t.Logf("draft=%d accepted=%d", speculative.DraftTokens, speculative.AcceptedDraft)
}
