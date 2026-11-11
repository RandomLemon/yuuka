package engine

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestLookupDraft(t *testing.T) {
	history := []int32{1, 2, 3, 4, 1, 2, 3}
	draft := lookupDraft(history, 4, 2)
	if want := []int32{4, 1, 2, 3}; !reflect.DeepEqual(draft, want) {
		t.Fatalf("draft %v, want %v", draft, want)
	}
	if got := lookupDraft([]int32{1, 2, 3}, 4, 2); got != nil {
		t.Fatalf("draft without an earlier match = %v", got)
	}
	if got := lookupDraft([]int32{5, 5, 5, 5}, 3, 2); !reflect.DeepEqual(got, []int32{5}) {
		t.Fatalf("overlapping draft = %v", got)
	}
}

func tinySpecEngine(t *testing.T) *Engine {
	t.Helper()
	path := filepath.Join(t.TempDir(), "spec-qwen35.gguf")
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
	runner, err := New(model, tok, 64, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	return runner
}

func TestSpeculativeMatchesGreedy(t *testing.T) {
	runner := tinySpecEngine(t)
	prompt, err := runner.Tokenizer.Encode("aaaaaaaaaaaaaaaa")
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens: 16,
		IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	speculative, err := runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens: 16,
		IgnoreEOS: true,
		Speculative: &SpeculativeOptions{
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
	if speculative.OutputTokens != baseline.OutputTokens {
		t.Fatalf("output tokens %d, want %d", speculative.OutputTokens, baseline.OutputTokens)
	}
	if speculative.DraftTokens == 0 || speculative.AcceptedDraft == 0 {
		t.Fatalf("speculative verification did not accept draft tokens: draft=%d accepted=%d",
			speculative.DraftTokens, speculative.AcceptedDraft)
	}
	t.Logf("draft=%d accepted=%d", speculative.DraftTokens, speculative.AcceptedDraft)
}

func TestSpeculativeRequiresGreedy(t *testing.T) {
	runner := tinySpecEngine(t)
	prompt, err := runner.Tokenizer.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := NewSampler(SamplerConfig{Temperature: 0.8, TopK: 40, TopP: 0.9}, runner.Runtime.VocabSize())
	if err != nil {
		t.Fatal(err)
	}
	runner.Sampler = sampler
	_, err = runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens:   4,
		Speculative: &SpeculativeOptions{MaxDraft: 4, MinMatch: 2},
	})
	if err == nil {
		t.Fatal("speculative decoding accepted a non-greedy sampler")
	}
}
