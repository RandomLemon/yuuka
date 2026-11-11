package engine

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestTinyFixtureGeneration(t *testing.T) {
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
	runner, err := New(model, tok, 16, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	prompt, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Generate(context.Background(), prompt, GenerateOptions{
		MaxTokens: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.OutputTokens != 3 {
		t.Fatalf("generated %d tokens, want 3", result.OutputTokens)
	}
	if len(result.Tokens) != 3 {
		t.Fatalf("token slice length %d", len(result.Tokens))
	}
	if result.FinishReason != "length" {
		t.Fatalf("finish reason %q", result.FinishReason)
	}
}
