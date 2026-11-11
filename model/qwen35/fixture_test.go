package qwen35

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func openTinyFixture(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *Model) {
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
	model, err := Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	return reader, tok, model
}

func TestTinyFixtureForward(t *testing.T) {
	_, tok, model := openTinyFixture(t)
	if model.Config.BlockCount != 2 {
		t.Fatalf("block count %d", model.Config.BlockCount)
	}
	if got, want := len(model.Config.FullAttentionLayers()), 1; got != want {
		t.Fatalf("full attention layers %d, want %d", got, want)
	}
	if !model.OutputTied() {
		t.Fatal("tiny fixture should use a tied LM head")
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 5 {
		t.Fatalf("encoded token count %d, want 5", len(ids))
	}
	state, err := NewState(model.Config, 16)
	if err != nil {
		t.Fatal(err)
	}
	first, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range first {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logit %d is not finite: %v", i, value)
		}
	}
	firstCopy := append([]float32(nil), first...)
	if _, err := model.Forward(ids, state); err != nil {
		t.Fatal(err)
	}
	state.Reset()
	second, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(firstCopy, second) {
		t.Fatal("fixture forward is not deterministic after reset")
	}
}

func TestTinyFixtureChatRendering(t *testing.T) {
	_, tok, _ := openTinyFixture(t)
	prompt, err := tok.RenderPrompt("You are concise.", "Hi", tokenizer.ChatOptions{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 8 {
		t.Fatalf("chat prompt encoded to only %d tokens", len(ids))
	}
	seen := map[int32]bool{}
	for _, id := range ids {
		seen[id] = true
	}
	if ids[0] != 256 || !seen[257] {
		t.Fatalf("chat prompt boundaries %v", ids)
	}
}
