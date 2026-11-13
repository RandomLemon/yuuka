package dense

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func openTinyDense(t *testing.T, arch string) (*gguf.Reader, *tokenizer.Tokenizer, *Model) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-"+arch+".gguf")
	if err := fixture.WriteTinyDense(path, arch); err != nil {
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

func TestDenseForward(t *testing.T) {
	for _, arch := range []string{Llama, Qwen3} {
		t.Run(arch, func(t *testing.T) {
			_, tok, model := openTinyDense(t, arch)
			if model.Config.Architecture != arch {
				t.Fatalf("architecture %q", model.Config.Architecture)
			}
			if arch == Llama && model.OutputTied() {
				t.Fatal("llama fixture should have an untied output head")
			}
			if arch == Qwen3 && !model.OutputTied() {
				t.Fatal("qwen3 fixture should have a tied output head")
			}
			ids, err := tok.Encode("Hello world")
			if err != nil {
				t.Fatal(err)
			}
			state, err := NewState(model.Config, 16)
			if err != nil {
				t.Fatal(err)
			}
			traces := make(map[int][]float32)
			logits, err := model.ForwardTrace(ids, state, func(layer int, x []float32) {
				traces[layer] = append([]float32(nil), x...)
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(traces) != model.Config.BlockCount {
				t.Fatalf("trace count %d, want %d", len(traces), model.Config.BlockCount)
			}
			for i := range model.Config.BlockCount {
				trace, ok := traces[i]
				if !ok {
					t.Fatalf("missing trace for layer %d", i)
				}
				if len(trace) != model.Config.HiddenSize {
					t.Fatalf("trace %d length %d", i, len(trace))
				}
				for _, value := range trace {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						t.Fatalf("trace %d has a non-finite value", i)
					}
				}
			}
			for i, value := range logits {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("logit %d is not finite: %v", i, value)
				}
			}
			first := append([]float32(nil), logits...)
			state.Reset()
			again, err := model.Forward(ids, state)
			if err != nil {
				t.Fatal(err)
			}
			for i := range first {
				if first[i] != again[i] {
					t.Fatalf("logit %d differs after reset: %v vs %v", i, first[i], again[i])
				}
			}
		})
	}
}

// TestTinyLlamaMatchesLlamaCPP records the four greedy token IDs produced by
// llama.cpp's llama-completion for the tiny Llama fixture with the same prompt
// and tokenizer. It guards the Llama tensor layout and CPU math.
func TestTinyLlamaMatchesLlamaCPP(t *testing.T) {
	_, tok, model := openTinyDense(t, Llama)
	ids, err := tok.EncodePrompt("Hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != 256 || ids[1] != 263 {
		t.Fatalf("prompt tokens %v, want [256 263]", ids)
	}
	state, err := NewState(model.Config, 16)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 0, 4)
	for i := range 4 {
		best := int32(0)
		for id, value := range logits {
			if value > logits[best] {
				best = int32(id)
			}
		}
		got = append(got, best)
		if i < 3 {
			logits, err = model.Step(best, state)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	want := []int32{157, 259, 66, 182}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("greedy tokens %v, want %v", got, want)
	}
}
