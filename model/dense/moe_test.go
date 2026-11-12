package dense

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func loadTinyMoE(t *testing.T) (*gguf.Reader, *Model, *tokenizer.Tokenizer) {
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
	model, err := Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	return reader, model, tok
}

func TestMoEConfig(t *testing.T) {
	_, model, _ := loadTinyMoE(t)
	cfg := model.Config
	if cfg.Architecture != Qwen3MoE || !cfg.IsMoE() {
		t.Fatalf("architecture %q IsMoE=%t", cfg.Architecture, cfg.IsMoE())
	}
	if cfg.ExpertCount != 4 || cfg.ExpertUsedCount != 2 || cfg.ExpertFFNSize != 16 {
		t.Fatalf("expert config count=%d used=%d width=%d",
			cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNSize)
	}
	layer := &model.Layers[0]
	if layer.GateInp.Rows != 4 || layer.GateExps.Rows != 64 ||
		layer.UpExps.Rows != 64 || layer.DownExps.Rows != 128 {
		t.Fatalf("expert tensor rows: inp=%d gate=%d up=%d down=%d",
			layer.GateInp.Rows, layer.GateExps.Rows, layer.UpExps.Rows, layer.DownExps.Rows)
	}
	if layer.Gate.Data != nil {
		t.Fatal("MoE layer loaded a dense gate matrix")
	}
}

func TestSelectExperts(t *testing.T) {
	logits := []float32{1, 4, 2, 4}
	ids := make([]int32, 2)
	weights := make([]float32, 2)
	selectExperts(logits, ids, weights)
	if !reflect.DeepEqual(ids, []int32{1, 3}) {
		t.Fatalf("selected %v, want [1 3]", ids)
	}
	if math.Abs(float64(weights[0]+weights[1]-1)) > 1e-6 {
		t.Fatalf("weights %v do not sum to one", weights)
	}
	if math.Abs(float64(weights[0]-weights[1])) > 1e-6 {
		t.Fatalf("tied logits produced unequal weights %v", weights)
	}
}

func TestExpertSlice(t *testing.T) {
	_, model, _ := loadTinyMoE(t)
	layer := &model.Layers[0]
	for expert := range model.Config.ExpertCount {
		gate, err := expertSlice(layer.GateExps, expert, model.Config.ExpertFFNSize)
		if err != nil {
			t.Fatal(err)
		}
		if gate.Rows != int64(model.Config.ExpertFFNSize) {
			t.Fatalf("expert %d rows %d", expert, gate.Rows)
		}
		next, err := expertSlice(layer.GateExps, expert+1, model.Config.ExpertFFNSize)
		if expert+1 < model.Config.ExpertCount {
			if err != nil {
				t.Fatal(err)
			}
			if len(gate.Data) == 0 || len(next.Data) == 0 || &gate.Data[0] == &next.Data[0] {
				t.Fatalf("expert %d and %d share storage", expert, expert+1)
			}
		}
	}
}

func TestMoEQuantizedFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen3moe-q4k.gguf")
	if err := fixture.WriteTinyMoEQuantized(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	model, err := Load(reader)
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
	state, err := NewState(model.Config, 32)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != model.Config.VocabSize {
		t.Fatalf("logit count %d, want %d", len(logits), model.Config.VocabSize)
	}
	for i, logit := range logits {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
			t.Fatalf("logit %d is not finite: %v", i, logit)
		}
	}
}

func TestMoEGreedyGeneration(t *testing.T) {
	_, model, tok := loadTinyMoE(t)
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	run := func() []int32 {
		state, err := NewState(model.Config, 32)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := model.Forward(ids, state)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, 4)
		for range 4 {
			best := int32(0)
			for id, value := range logits {
				if value > logits[best] {
					best = int32(id)
				}
			}
			out = append(out, best)
			logits, err = model.Step(best, state)
			if err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	first := run()
	second := run()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("MoE generation is not deterministic: %v vs %v", first, second)
	}
	if len(first) != 4 {
		t.Fatalf("generated %d tokens, want 4", len(first))
	}
}
