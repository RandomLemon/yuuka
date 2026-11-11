package qwen35

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

type testHarness struct {
	reader    *gguf.Reader
	tokenizer *tokenizer.Tokenizer
	model     *Model
}

func newTestHarness(t *testing.T) *testHarness {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
	}
	r, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	tok, err := tokenizer.NewFromGGUF(r)
	if err != nil {
		t.Fatal(err)
	}
	model, err := Load(r)
	if err != nil {
		t.Fatal(err)
	}
	return &testHarness{reader: r, tokenizer: tok, model: model}
}

func (h *testHarness) generate(t *testing.T, prompt string, count int) []int32 {
	t.Helper()
	state, err := NewState(h.model.Config, 64)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := h.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := h.model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, 0, count)
	for i := 0; i < count; i++ {
		best := int32(0)
		for id, value := range logits {
			if value > logits[best] {
				best = int32(id)
			}
		}
		out = append(out, best)
		if i+1 < count {
			logits, err = h.model.Step(best, state)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	return out
}

func TestGoldenGeneration(t *testing.T) {
	h := newTestHarness(t)
	cases := []struct {
		prompt string
		want   []int32
	}{
		{"The capital of France is", []int32{11751, 13, 198, 32, 13, 2912}},
		{"1+1=", []int32{17, 3709, 17, 10, 17, 28}},
		{"Once upon a time", []int32{11, 303, 264, 1814, 2400, 314}},
	}
	for _, tc := range cases {
		got := h.generate(t, tc.prompt, len(tc.want))
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%q: tokens %v, want %v", tc.prompt, got, tc.want)
		}
	}
}

func TestGreedyDeterminism(t *testing.T) {
	h := newTestHarness(t)
	first := h.generate(t, "The capital of France is", 5)
	second := h.generate(t, "The capital of France is", 5)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("greedy output differs between runs: %v vs %v", first, second)
	}
}

func TestLogitsAgainstReference(t *testing.T) {
	h := newTestHarness(t)
	state, err := NewState(h.model.Config, 8)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := h.tokenizer.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	logits, err := h.model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	// Reference logits from llama.cpp CPU for the same GGUF and token sequence.
	// The tolerance covers activation quantization in the reference's integer
	// dot products; the CPU reference here keeps activations in FP32.
	reference := map[int32]float32{
		11:   13.071139,
		0:    12.340711,
		13:   11.254948,
		1017: 10.421552,
	}
	for id, want := range reference {
		if diff := float32(math.Abs(float64(logits[id] - want))); diff > 1.5 {
			t.Fatalf("logit %d = %.6f, reference %.6f, diff %.6f exceeds tolerance 1.5",
				id, logits[id], want, diff)
		}
	}
	if logits[11] <= logits[0] {
		t.Fatalf("top-1 changed: logit 11=%.6f logit 0=%.6f", logits[11], logits[0])
	}
}

func TestStateCapacity(t *testing.T) {
	h := newTestHarness(t)
	if _, err := NewState(h.model.Config, h.model.Config.ContextLength+1); err == nil {
		t.Fatal("expected context capacity error")
	}
	state, err := NewState(h.model.Config, 1)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := h.tokenizer.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.model.Forward(ids, state); err == nil {
		t.Fatal("expected context overflow error")
	}
}
