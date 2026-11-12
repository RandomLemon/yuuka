package engine

import (
	"context"
	"encoding/binary"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// scriptedSpecRuntime forces a draft block whose first token matches the greedy
// prediction and whose second token does not. It lets the speculative loop test
// partial acceptance, rollback, and replay without depending on model weights.
type scriptedSpecRuntime struct {
	tokens []int32
}

func (r *scriptedSpecRuntime) Reset() { r.tokens = nil }

func (r *scriptedSpecRuntime) nextLogits() []float32 {
	logits := make([]float32, 16)
	token := int32(7)
	switch len(r.tokens) {
	case 5:
		token = 3
	case 6:
		token = 4
	case 7:
		token = 5
	case 8:
		token = 6
	}
	logits[token] = 1
	return logits
}

func (r *scriptedSpecRuntime) appendToken(token int32) ([]float32, error) {
	if len(r.tokens) >= r.Capacity() {
		return nil, fmt.Errorf("scripted runtime capacity exhausted")
	}
	r.tokens = append(r.tokens, token)
	return r.nextLogits(), nil
}

func (r *scriptedSpecRuntime) Forward(tokens []int32) ([]float32, error) {
	var logits []float32
	for _, token := range tokens {
		var err error
		logits, err = r.appendToken(token)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (r *scriptedSpecRuntime) Step(token int32) ([]float32, error) {
	return r.appendToken(token)
}

func (r *scriptedSpecRuntime) PrefillLogits(tokens []int32) ([][]float32, error) {
	out := make([][]float32, 0, len(tokens))
	for _, token := range tokens {
		logits, err := r.appendToken(token)
		if err != nil {
			return nil, err
		}
		out = append(out, logits)
	}
	return out, nil
}

func (r *scriptedSpecRuntime) SnapshotState() ([]byte, int, error) {
	data := make([]byte, len(r.tokens)*4)
	for i, token := range r.tokens {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(token))
	}
	return data, len(r.tokens), nil
}

func (r *scriptedSpecRuntime) RestoreState(data []byte, position int) error {
	if len(data)%4 != 0 || len(data)/4 != position {
		return fmt.Errorf("scripted snapshot has %d bytes for position %d", len(data), position)
	}
	tokens := make([]int32, len(data)/4)
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
	}
	r.tokens = tokens
	return nil
}

func (r *scriptedSpecRuntime) Capacity() int  { return 32 }
func (r *scriptedSpecRuntime) VocabSize() int { return 16 }
func (r *scriptedSpecRuntime) Close() error   { return nil }

func TestSpeculativePartialAcceptanceMatchesGreedy(t *testing.T) {
	base := tinySpecEngine(t)
	defer base.Close()
	runner, err := NewWithRuntime(&scriptedSpecRuntime{}, base.Tokenizer, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	prompt := []int32{1, 2, 3, 1, 2}
	baseline, err := runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens: 4,
		IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	speculative, err := runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens: 4,
		IgnoreEOS: true,
		Speculative: &SpeculativeOptions{
			MaxDraft: 2,
			MinMatch: 2,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(speculative.Tokens, baseline.Tokens) {
		t.Fatalf("speculative tokens %v, want %v", speculative.Tokens, baseline.Tokens)
	}
	if speculative.DraftTokens != 2 || speculative.AcceptedDraft != 1 {
		t.Fatalf("draft/accepted = %d/%d, want 2/1",
			speculative.DraftTokens, speculative.AcceptedDraft)
	}
}

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
