//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUAttentionPathsAgree compares identical full-model prompts across the
// three prefill attention kernels: the tiled kernel (forced), the FP16 MMA
// kernel (default), and the legacy rocBLAS path. The KV pool is F16 so every
// arm reads the same stored values, and only the kernel changes.
func TestGPUAttentionPathsAgree(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL unset")
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
	// A 64-token prompt is the shortest chunk the FP16 MMA kernel accepts, and
	// it stays on the FP16 prefill route. The long prompt switches the prefill
	// GEMMs to the int8 Q8_1 route, which lands the two arms on different
	// activation grid points.
	n := 64
	repetitions := 7
	gate := 0.05
	if os.Getenv("YUUKA_TEST_ATTENTION_LONG") == "1" {
		n = 2048
		repetitions = 450
		gate = 0.3
	}
	// The 8:1 F16 MMA route can flip a near-tie MoE expert selection even when
	// the attention output is within the native bound; the approved contract
	// keeps the top-1 token and greedy continuation and uses the MoE route
	// envelope recorded in moe_real_test.go.
	if model.Config.IsMoE() {
		gate = moeAttentionRouteLogitTolerance
	}
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", repetitions))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < n {
		t.Skipf("only %d tokens, need %d", len(ids), n)
	}
	ids = ids[:n]
	capacity := n + 64
	type arm struct {
		name string
		opts Options
		// mma and blas report the dispatch flag the model resolved to.
		mma  bool
		blas bool
	}
	arms := []arm{
		{name: "tiled", opts: Options{KVType: tensor.F16, KVTypeSet: true, ForceTiledAttention: true}},
		{name: "mma", opts: Options{KVType: tensor.F16, KVTypeSet: true}, mma: true},
		{name: "blas", opts: Options{KVType: tensor.F16, KVTypeSet: true, LegacyBlasAttention: true}, blas: true},
	}
	logits := make(map[string][]float32, len(arms))
	for _, a := range arms {
		rt, err := newRuntime(reader, model, capacity, a.opts)
		if err != nil {
			t.Fatal(err)
		}
		if err := rt.SetPrefillChunk(512); err != nil {
			rt.Close()
			t.Fatal(err)
		}
		if got, want := rt.model.attentionBlasEnabled, a.blas; got != want {
			rt.Close()
			t.Fatalf("%s: BLAS attention enabled=%v, want %v", a.name, got, want)
		}
		got, err := rt.Forward(ids)
		rt.Close()
		if err != nil {
			t.Fatalf("%s: %v", a.name, err)
		}
		logits[a.name] = got
	}
	baseline := logits["tiled"]
	for _, a := range arms[1:] {
		got := logits[a.name]
		var maxDiff float64
		for i, v := range got {
			delta := math.Abs(float64(v - baseline[i]))
			if delta > maxDiff {
				maxDiff = delta
			}
		}
		t.Logf("model %s attention vs tiled: top1=%d/%d max_diff=%.6g", a.name,
			top1(got), top1(baseline), maxDiff)
		// The arms share everything but the attention kernel. The short-prompt
		// gate matches the tolerance the KV-format test uses; the long-prompt
		// gate matches the Q8_1 route's measured sensitivity, which moves logits
		// by the same amount as changing the KV storage format on one kernel.
		// The measurements are recorded in docs/benchmarks.md.
		if top1(got) != top1(baseline) || maxDiff > gate {
			t.Fatalf("%s attention exceeds the %g model-level logit gate (%.6g)",
				a.name, gate, maxDiff)
		}
	}
}

func TestGPULongSplitAttentionMatchesDirect(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL unset")
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
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 1200))
	if err != nil {
		t.Fatal(err)
	}
	const length = 8192
	if len(ids) < length {
		t.Fatalf("prompt has %d tokens, want %d", len(ids), length)
	}
	rt, err := NewWithKVType(reader, model, length+16, tensor.F16)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(2048); err != nil {
		t.Fatal(err)
	}
	m := rt.model
	scratch := m.prefill.attentionSplit
	if scratch == nil {
		t.Skip("model or device does not select the gfx1151 split route")
	}
	run := func(split bool) [][]float32 {
		t.Helper()
		if err := rt.session.Reset(); err != nil {
			t.Fatal(err)
		}
		m.prefill.attentionSplit = nil
		if split {
			m.prefill.attentionSplit = scratch
		}
		m.prefillGen++
		logits, err := rt.session.Prefill(ids[:length], true)
		if err != nil {
			t.Fatal(err)
		}
		result := [][]float32{append([]float32(nil), logits...)}
		for range 3 {
			logits, err = rt.session.Decode(int32(top1(logits)))
			if err != nil {
				t.Fatal(err)
			}
			result = append(result, append([]float32(nil), logits...))
		}
		return result
	}
	want := run(false)
	got := run(true)
	// Use the existing cross-route long-attention tolerance for F16/Q8_1.
	const tolerance = 0.3
	for step := range want {
		var maxDiff float64
		for i, v := range got[step] {
			maxDiff = math.Max(maxDiff, math.Abs(float64(v-want[step][i])))
		}
		if math.IsNaN(maxDiff) || maxDiff > tolerance || top1(got[step]) != top1(want[step]) {
			t.Fatalf("step %d: max logit diff %.6g, top-1 %d vs %d",
				step, maxDiff, top1(got[step]), top1(want[step]))
		}
		t.Logf("step %d: max logit diff %.6g, top-1 %d", step, maxDiff, top1(got[step]))
	}
}
