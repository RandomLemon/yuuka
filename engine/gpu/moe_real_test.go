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

// openMoEModel loads the opt-in qwen35moe GGUF named by YUUKA_TEST_MOE_MODEL.
// The real-model tests load 10.7 GB of weights and run the CPU reference, so
// they stay behind the environment variable like the CPU MoE golden test.
func openMoEModel(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *qwen35.Model) {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MOE_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MOE_MODEL is not set")
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
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	if !model.Config.IsMoE() {
		t.Skipf("model architecture %q is not a MoE architecture",
			model.Config.Architecture)
	}
	return reader, tok, model
}

// TestGPUMoEGoldenGeneration checks the real qwen35moe model end to end: the
// first-token logits must match the CPU reference, and the four greedy tokens
// must match both the CPU reference and llama.cpp's golden continuation. An F32
// KV pool isolates the expert arithmetic from KV storage rounding.
func TestGPUMoEGoldenGeneration(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode("The capital of France is")
	if err != nil {
		t.Fatal(err)
	}
	const capacity = 64
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	opts := moERouteOptions(t)
	opts.KVType = tensor.F32
	opts.KVTypeSet = true
	rt, err := newRuntime(reader, model, capacity, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	t.Logf("routed-expert route: %s", moERouteName(opts))
	gpuLogits, err := rt.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	diff := maxDiff(cpuLogits, gpuLogits)
	if top1(cpuLogits) != top1(gpuLogits) {
		t.Fatalf("first token differs: cpu %d gpu %d (max logit diff %.6g)",
			top1(cpuLogits), top1(gpuLogits), diff)
	}
	if diff > 0.1 {
		t.Fatalf("max logit diff %.6g exceeds 0.1", diff)
	}

	gpuTokens := make([]int32, 0, 4)
	for i := range 4 {
		best := int32(top1(gpuLogits))
		gpuTokens = append(gpuTokens, best)
		if i+1 < 4 {
			gpuLogits, err = rt.Step(best)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	cpuTokens := make([]int32, 0, 4)
	cpuNext := cpuLogits
	for i := range 4 {
		best := int32(top1(cpuNext))
		cpuTokens = append(cpuTokens, best)
		if i+1 < 4 {
			cpuNext, err = model.Step(best, cpuState)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	want := []int32{11751, 11, 264, 3177} // " Paris, a city"
	if !equalTokens(cpuTokens, want) {
		t.Fatalf("cpu tokens %v, want %v", cpuTokens, want)
	}
	if !equalTokens(gpuTokens, want) {
		t.Fatalf("gpu tokens %v, want %v (max first-logit diff %.6g)",
			gpuTokens, want, diff)
	}
	t.Logf("max first-logit diff %.6g, tokens %v", diff, gpuTokens)
}

// prefillLogitTolerance bounds the chunked-prefill logit difference against the
// CPU reference. Prefill agreement is gated on token behavior, not on a tight
// numerical bound: the engine's chunked prefill drifts from the CPU reference as
// the prefix grows even for the dense model (recorded 0.006 at 8 tokens and 0.57
// at 128 tokens on Qwen3.5-4B), and an MoE model amplifies that drift through
// near-tie routing: perturbing a router weight by 1e-4 relative moves this
// model's own logits by 0.64. The recorded value for the 256-token prompt below
// is 0.96 at a 64-row chunk and 1.04 at a 256-row chunk, with a matching top-1
// token and the same four greedy continuation tokens.
const prefillLogitTolerance = 2.0

// moeAttentionRouteLogitTolerance bounds the full-model logits of the 8:1 F16
// grouped-head WMMA attention route against the forced tiled control. Staging
// Q, the probability tile, and the value product on the FP16 MMA pipe moves a
// near-tie expert selection on this router-sensitive model, so the logit
// difference is the same order as prefillLogitTolerance even though the
// attention output itself stays within the native 5e-3 bound and the top-1
// token and greedy continuation are unchanged. Recorded maxima on the fixed
// 8192-token stream are 1.43 at 512, 1.26 at 2048, and 0.87 at 8192. The
// approved route ships only with identical top-1 and greedy tokens; this bound
// is the numerical envelope, not a substitute for that behavioral check.
const moeAttentionRouteLogitTolerance = 2.0

// TestGPUMoEChunkedPrefill checks the grouped expert prefill path: the
// next-token logits must keep the CPU top-1 token, and the four greedy tokens
// decoded from a 256-token prompt must match the CPU continuation.
func TestGPUMoEChunkedPrefill(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 256 {
		t.Skipf("prompt produced %d tokens, need 256", len(ids))
	}
	ids = ids[:256]
	const capacity = 256 + 8
	cpuState, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		t.Fatal(err)
	}
	cpuLogits, err := model.Forward(ids, cpuState)
	if err != nil {
		t.Fatal(err)
	}

	opts := moERouteOptions(t)
	opts.KVType = tensor.F32
	opts.KVTypeSet = true
	rt, err := newRuntime(reader, model, capacity, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	t.Logf("routed-expert route: %s", moERouteName(opts))
	var gpuLogits []float32
	for _, chunk := range []int{64, 256} {
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		rt.Reset()
		gpuLogits, err = rt.Forward(ids)
		if err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		diff := maxDiff(cpuLogits, gpuLogits)
		if top1(cpuLogits) != top1(gpuLogits) {
			t.Fatalf("chunk %d: top-1 token differs: cpu %d gpu %d (max diff %.6g)",
				chunk, top1(cpuLogits), top1(gpuLogits), diff)
		}
		if diff > prefillLogitTolerance {
			t.Fatalf("chunk %d: max logit diff %.6g exceeds %.1f",
				chunk, diff, prefillLogitTolerance)
		}
		t.Logf("chunk %d: max logit diff %.6g", chunk, diff)
	}

	// The behavioral gate: four greedy tokens after the chunked prefill must
	// match the CPU continuation from the same prompt.
	const continuation = 4
	gpuTokens := make([]int32, 0, continuation)
	for i := range continuation {
		best := int32(top1(gpuLogits))
		gpuTokens = append(gpuTokens, best)
		if i+1 < continuation {
			gpuLogits, err = rt.Step(best)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	cpuTokens := make([]int32, 0, continuation)
	next := cpuLogits
	for i := range continuation {
		best := int32(top1(next))
		cpuTokens = append(cpuTokens, best)
		if i+1 < continuation {
			next, err = model.Step(best, cpuState)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if !equalTokens(gpuTokens, cpuTokens) {
		t.Fatalf("chunked prefill continuation %v, cpu %v", gpuTokens, cpuTokens)
	}
	t.Logf("chunked prefill continuation %v", gpuTokens)
}

// TestGPUMoEPrefillGreedyContinuity compares the shipping 8:1 grouped-head
// WMMA attention route with the forced tiled control on the same fixed
// 8192-token stream. The two runtimes are sequential so only one model's
// weights are resident at a time. Each target prefix keeps the top-1 token and
// three subsequent greedy decode top-1 tokens, and every logit row stays
// within the long-attention cross-route bound. A target-path failure is a test
// failure, not a skip.
func TestGPUMoEPrefillGreedyContinuity(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	const length = 8192
	const chunk = 512
	const continuation = 3
	const tolerance = moeAttentionRouteLogitTolerance
	ids, err := tok.Encode(strings.Repeat(moePrefillBenchmarkText, length))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < length {
		t.Fatalf("benchmark prompt produced %d tokens, need %d", len(ids), length)
	}
	ids = ids[:length]
	type routeResult struct {
		rows   [][]float32
		tokens []int32
	}
	run := func(forceTiled bool) map[int]routeResult {
		t.Helper()
		opts := Options{KVType: tensor.F16, KVTypeSet: true, ForceTiledAttention: forceTiled}
		rt, err := newRuntime(reader, model, length+64, opts)
		if err != nil {
			t.Fatalf("HIP runtime unavailable: %v", err)
		}
		defer rt.Close()
		if err := rt.SetPrefillChunk(chunk); err != nil {
			t.Fatal(err)
		}
		out := make(map[int]routeResult, 3)
		for _, n := range []int{512, 2048, 8192} {
			if err := rt.session.Reset(); err != nil {
				t.Fatal(err)
			}
			logits, err := rt.session.Prefill(ids[:n], true)
			if err != nil {
				t.Fatalf("prefix %d: %v", n, err)
			}
			result := routeResult{tokens: []int32{int32(top1(logits))}}
			result.rows = append(result.rows, append([]float32(nil), logits...))
			token := result.tokens[0]
			for range continuation {
				logits, err = rt.session.Decode(token)
				if err != nil {
					t.Fatalf("prefix %d decode: %v", n, err)
				}
				token = int32(top1(logits))
				result.tokens = append(result.tokens, token)
				result.rows = append(result.rows, append([]float32(nil), logits...))
			}
			out[n] = result
		}
		return out
	}
	shipping := run(false)
	control := run(true)
	failed := false
	for _, n := range []int{512, 2048, 8192} {
		got, want := shipping[n], control[n]
		if len(got.tokens) != continuation+1 || len(want.tokens) != continuation+1 {
			t.Fatalf("prefix %d produced %d/%d greedy tokens", n, len(got.tokens),
				len(want.tokens))
		}
		if !equalTokens(got.tokens, want.tokens) {
			t.Errorf("prefix %d greedy tokens %v, tiled control %v", n, got.tokens,
				want.tokens)
			failed = true
		}
		worst := 0.0
		for row := range got.rows {
			if got.tokens[row] != int32(top1(got.rows[row])) ||
				want.tokens[row] != int32(top1(want.rows[row])) {
				t.Errorf("prefix %d row %d top-1 does not match the greedy token", n, row)
				failed = true
			}
			for i, value := range got.rows[row] {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("prefix %d row %d logit %d is not finite", n, row, i)
				}
				diff := math.Abs(float64(value - want.rows[row][i]))
				if diff > worst {
					worst = diff
				}
			}
		}
		if worst > tolerance {
			t.Errorf("prefix %d max logit difference %.6g exceeds the %.1f bound",
				n, worst, tolerance)
			failed = true
		}
		t.Logf("prefix %d: max logit diff %.6g, greedy tokens %v", n, worst,
			got.tokens)
	}
	if failed {
		t.FailNow()
	}
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
