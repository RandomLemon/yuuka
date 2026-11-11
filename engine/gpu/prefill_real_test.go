//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUChunkedPrefillLongContext checks that different chunk boundaries over
// a 1024-token real-model prompt produce the same greedy token and nearly the
// same logits. It is opt-in through YUUKA_TEST_MODEL because it loads and
// prefills the full model several times.
func TestGPUChunkedPrefillLongContext(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
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
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 1024 {
		t.Skipf("prompt produced %d tokens, need 1024", len(ids))
	}
	ids = ids[:1024]
	const capacity = 1024 + 8
	rt, err := New(reader, model, capacity)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()

	var reference []float32
	for _, chunk := range []int{128, 512, 1024} {
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		rt.Reset()
		got, err := rt.Forward(ids)
		if err != nil {
			t.Fatalf("chunk %d prefill: %v", chunk, err)
		}
		if reference == nil {
			reference = append([]float32(nil), got...)
			t.Logf("chunk %d reference top1=%d", chunk, top1(got))
			continue
		}
		diff := 0.0
		for i := range got {
			diff = math.Max(diff, math.Abs(float64(got[i]-reference[i])))
		}
		if top1(got) != top1(reference) {
			t.Fatalf("chunk %d top-1 %d differs from reference %d (max diff %.6g)",
				chunk, top1(got), top1(reference), diff)
		}
		if diff > 0.05 {
			t.Fatalf("chunk %d max logit diff %.6g exceeds 0.05", chunk, diff)
		}
		t.Logf("chunk %d top1=%d max diff %.6g", chunk, top1(got), diff)
	}
}
