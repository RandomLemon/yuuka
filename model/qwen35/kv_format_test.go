package qwen35

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestKVFormatRoundTrip(t *testing.T) {
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.37)) * 3
	}
	values[0] = 0
	cases := []struct {
		format KVFormat
		tol    float32
	}{
		{KVF32, 0},
		{KVF16, 1e-3},
		{KVQ8_0, 2e-2},
	}
	for _, tc := range cases {
		t.Run(tc.format.String(), func(t *testing.T) {
			rowBytes, err := tc.format.RowBytes(len(values))
			if err != nil {
				t.Fatal(err)
			}
			packed := make([]byte, rowBytes)
			if err := tc.format.EncodeRow(packed, values); err != nil {
				t.Fatal(err)
			}
			decoded := make([]float32, len(values))
			if err := tc.format.DecodeRow(decoded, packed); err != nil {
				t.Fatal(err)
			}
			for i := range values {
				if diff := math.Abs(float64(decoded[i] - values[i])); diff > float64(tc.tol) {
					t.Fatalf("index %d: got %v want %v (diff %g)", i, decoded[i], values[i], diff)
				}
			}
		})
	}
}

func TestKVFormatPadsPartialQ8Block(t *testing.T) {
	rowBytes, err := KVQ8_0.RowBytes(40)
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * 34; rowBytes != want {
		t.Fatalf("Q8_0 row bytes %d, want %d", rowBytes, want)
	}
}

func TestPagedAttentionKVFormats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
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
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	run := func(format KVFormat) ([]int32, []float32, int) {
		state, err := NewStateWithFormat(model.Config, 32, format)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := model.Forward(ids, state)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]int32, 0, 4)
		for range 4 {
			best := int32(0)
			for id, value := range logits {
				if value > logits[best] {
					best = int32(id)
				}
			}
			got = append(got, best)
			logits, err = model.Step(best, state)
			if err != nil {
				t.Fatal(err)
			}
		}
		return got, append([]float32(nil), logits...), state.pool.RowBytes()
	}
	baseTokens, baseLogits, baseRowBytes := run(KVF32)
	for _, tc := range []struct {
		format KVFormat
		tol    float64
	}{
		{KVF16, 2e-2},
		{KVQ8_0, 2e-1},
	} {
		tokens, logits, rowBytes := run(tc.format)
		if tc.format == KVF16 && rowBytes >= baseRowBytes {
			t.Fatalf("%s row bytes %d did not shrink from F32 %d", tc.format, rowBytes, baseRowBytes)
		}
		for i := range baseTokens {
			if tokens[i] != baseTokens[i] {
				t.Fatalf("%s token %d: got %v want %v", tc.format, i, tokens, baseTokens)
			}
		}
		maxDiff := 0.0
		for i := range baseLogits {
			diff := math.Abs(float64(logits[i] - baseLogits[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff > tc.tol {
			t.Fatalf("%s logits differ from F32 by %.6g, tolerance %.6g",
				tc.format, maxDiff, tc.tol)
		}
		t.Logf("%s row bytes %d (F32 %d), max logit diff %.6g", tc.format,
			rowBytes, baseRowBytes, maxDiff)
	}
}
