package engine

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// eosRuntime always selects the tokenizer EOS token. It isolates the automatic
// EOS stop path from model weights and logits.
type eosRuntime struct {
	vocab int
	eos   int32
}

func (r *eosRuntime) Reset() {}

func (r *eosRuntime) logits() []float32 {
	out := make([]float32, r.vocab)
	out[r.eos] = 1
	return out
}

func (r *eosRuntime) Forward([]int32) ([]float32, error) { return r.logits(), nil }
func (r *eosRuntime) Step(int32) ([]float32, error)      { return r.logits(), nil }
func (r *eosRuntime) Capacity() int                      { return 8 }
func (r *eosRuntime) VocabSize() int                     { return r.vocab }
func (r *eosRuntime) Close() error                       { return nil }

func TestGenerateStopsOnEOS(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-spm-eos.gguf")
	if err := fixture.WriteTinySPMVocab(path); err != nil {
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
	runner, err := NewWithRuntime(&eosRuntime{
		vocab: tok.VocabSize(),
		eos:   tok.EOSToken(),
	}, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	result, err := runner.Generate(context.Background(), []int32{263}, GenerateOptions{
		MaxTokens: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != "stop" {
		t.Fatalf("finish reason %q, want stop", result.FinishReason)
	}
	if result.OutputTokens != 0 || len(result.Tokens) != 0 {
		t.Fatalf("generated %d tokens (%v), want none", result.OutputTokens, result.Tokens)
	}
}
