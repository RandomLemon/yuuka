package engine

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/dense"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestDenseRuntimeGeneration(t *testing.T) {
	for _, arch := range []string{dense.Llama, dense.Qwen3} {
		t.Run(arch, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tiny-"+arch+".gguf")
			if err := fixture.WriteTinyDense(path, arch); err != nil {
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
			model, err := dense.Load(reader)
			if err != nil {
				t.Fatal(err)
			}
			runtime, err := dense.NewRuntime(model, 16)
			if err != nil {
				t.Fatal(err)
			}
			runner, err := NewWithRuntime(runtime, tok, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer runner.Close()
			prompt, err := tok.Encode("Hello")
			if err != nil {
				t.Fatal(err)
			}
			result, err := runner.Generate(context.Background(), prompt, GenerateOptions{MaxTokens: 3, IgnoreEOS: true})
			if err != nil {
				t.Fatal(err)
			}
			if result.OutputTokens != 3 {
				t.Fatalf("generated %d tokens, want 3", result.OutputTokens)
			}
			if len(result.Text) == 0 {
				t.Fatal("generated text is empty")
			}
		})
	}
}
