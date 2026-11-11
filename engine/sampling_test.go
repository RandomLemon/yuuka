package engine

import (
	"fmt"
	"testing"
)

func TestGreedySampler(t *testing.T) {
	s, err := NewSampler(SamplerConfig{}, 3)
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.Sample([]float32{0.5, 2.0, 1.0})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("greedy sample = %d, want 1", id)
	}
}

func TestTopKSampler(t *testing.T) {
	s, err := NewSampler(SamplerConfig{Temperature: 1, TopK: 1}, 3)
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		id, err := s.Sample([]float32{0.5, 2.0, 1.0})
		if err != nil {
			t.Fatal(err)
		}
		if id != 1 {
			t.Fatalf("top-k=1 sample = %d, want 1", id)
		}
	}
}

func TestSamplerDeterministic(t *testing.T) {
	first, err := NewSampler(SamplerConfig{Temperature: 1.5, TopK: 4, TopP: 0.9, Seed: 42}, 5)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewSampler(SamplerConfig{Temperature: 1.5, TopK: 4, TopP: 0.9, Seed: 42}, 5)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{0.1, 0.2, 3.0, 2.5, 0.3}
	for i := range 10 {
		a, err := first.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
		b, err := second.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
		if a != b {
			t.Fatalf("draw %d differs: %d vs %d", i, a, b)
		}
	}
}

func TestSamplerRejectsBadConfig(t *testing.T) {
	cases := []SamplerConfig{
		{Temperature: -1},
		{TopK: -1},
		{TopP: 1.5},
	}
	for _, cfg := range cases {
		if _, err := NewSampler(cfg, 10); err == nil {
			t.Fatalf("config %+v was accepted", cfg)
		}
	}
}

func TestTextStreamSplitsUTF8(t *testing.T) {
	text := "你好🙂"
	raw := []byte(text)
	stream := &textStream{}
	var got string
	for _, b := range raw {
		got += stream.Feed(string([]byte{b}))
	}
	got += stream.Flush()
	if got != text {
		t.Fatalf("streamed %q, want %q", got, text)
	}
	if stream.Text() != text {
		t.Fatalf("accumulated %q, want %q", stream.Text(), text)
	}
}

func ExampleSampler() {
	s, _ := NewSampler(SamplerConfig{Seed: 1}, 1)
	id, _ := s.Sample([]float32{1})
	fmt.Println(id)
	// Output: 0
}
