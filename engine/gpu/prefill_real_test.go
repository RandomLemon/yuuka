//go:build hip && linux

package gpu

import (
	"bytes"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUChunkedPrefillLongContext checks that different chunk boundaries over
// a 1024-token real-model prompt produce the same greedy token and logits
// within 0.05. The reference run was bitwise identical, but this test does not
// require bitwise equality. It is opt-in through YUUKA_TEST_MODEL because it
// loads and prefills the full model several times. On a MoE model,
// YUUKA_TEST_MOE_ROUTE selects the routed-expert route, so the same chunk
// invariance gate runs over the Q8_1 dp4a and the int8 WMMA expert route.
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
	opts := moERouteOptions(t)
	opts.KVType = DefaultKVType
	opts.KVTypeSet = true
	rt, err := newRuntime(reader, model, capacity, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	t.Logf("routed-expert route: %s", moERouteName(opts))

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

func TestGPUFinalPrefillMatchesFullRows(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	last := model.Config.BlockCount - 1
	if model.Config.IsRecurrent(last) {
		t.Skip("model ends in a recurrent layer")
	}
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200))
	if err != nil {
		t.Fatal(err)
	}
	const chunk = 512
	if len(ids) < 2*chunk {
		t.Fatalf("prompt has %d tokens, want at least %d", len(ids), 2*chunk)
	}
	rt, err := NewWithKVType(reader, model, 2*chunk+8, tensor.F16)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(chunk); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	m := rt.model
	kv := []*hip.Buffer{m.poolK[last], m.poolV[last]}

	for _, tc := range []struct {
		name     string
		n, split int
	}{
		{"single", 1, 0},
		{"short_chunk", 128, 0},
		{"full_chunk", chunk, 0},
		{"short_tail", chunk + 1, 0},
		{"displaced_tail", chunk + 3, 0},
		{"two_chunks", 2 * chunk, 0},
		{"separate_prefix", 2 * chunk, chunk},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := func(fullRows bool) ([][]float32, [][]byte) {
				t.Helper()
				if err := s.Reset(); err != nil {
					t.Fatal(err)
				}
				// Remove both previously built plans and stale KV before
				// running the other graph on the same model.
				s.syncPrefillPlans()
				for rows, plan := range s.prefillPlans {
					if err := plan.Close(); err != nil {
						t.Fatal(err)
					}
					delete(s.prefillPlans, rows)
				}
				for _, buf := range kv {
					if err := m.device.Memset(m.stream, buf, 0, buf.Bytes()); err != nil {
						t.Fatal(err)
					}
				}
				if err := m.stream.Synchronize(); err != nil {
					t.Fatal(err)
				}
				if fullRows {
					for _, n := range []int{min(chunk, tc.n), tc.n % chunk} {
						if n == 0 || s.prefillPlans[n] != nil {
							continue
						}
						plan, err := m.buildPrefillPlan(s, n, prefillLastLogitFullFFN)
						if err != nil {
							t.Fatal(err)
						}
						s.prefillPlans[n] = plan
					}
				}
				if tc.split > 0 {
					logits, err := s.Prefill(ids[:tc.split], false)
					if err != nil {
						t.Fatal(err)
					}
					if logits != nil || s.Position() != tc.split {
						t.Fatalf("nonfinal prefix returned logits or position %d", s.Position())
					}
				}
				logits, err := s.Prefill(ids[tc.split:tc.n], true)
				if err != nil {
					t.Fatal(err)
				}
				pages := make([][]byte, len(kv))
				for i, buf := range kv {
					pages[i] = make([]byte, buf.Bytes())
					if err := m.device.Download(pages[i], buf); err != nil {
						t.Fatal(err)
					}
				}
				result := [][]float32{append([]float32(nil), logits...)}
				for range 3 {
					logits, err = s.Decode(int32(top1(logits)))
					if err != nil {
						t.Fatal(err)
					}
					result = append(result, append([]float32(nil), logits...))
				}
				return result, pages
			}
			want, wantKV := run(true)
			got, gotKV := run(false)
			for i := range wantKV {
				if !bytes.Equal(wantKV[i], gotKV[i]) {
					t.Fatalf("KV buffer %d differs from full rows", i)
				}
			}
			for i := range want {
				diff := 0.0
				for j := range want[i] {
					diff = math.Max(diff, math.Abs(float64(got[i][j]-want[i][j])))
				}
				if math.IsNaN(diff) || diff > 0.05 || top1(got[i]) != top1(want[i]) {
					t.Fatalf("step %d: max logit diff %.6g, top-1 %d vs %d", i, diff, top1(got[i]), top1(want[i]))
				}
				t.Logf("step %d: max logit diff %.6g, top-1 %d", i, diff, top1(got[i]))
			}
		})
	}
}
