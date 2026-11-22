//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

// moERouteOptions returns the Options that select the routed-expert prefill
// route under test. YUUKA_TEST_MOE_ROUTE=mmq builds the grouped int8 WMMA expert
// route, "q8" (or unset) keeps the released Q8_1 dp4a route, and any other value
// fails instead of silently profiling another route. The switch exists so the
// same model, chunk, and prompt can be attributed under both routes in one test
// run; it never changes a default.
func moERouteOptions(t *testing.T) Options {
	t.Helper()
	opts := Options{}
	switch route := os.Getenv("YUUKA_TEST_MOE_ROUTE"); route {
	case "", "q8":
	case "mmq":
		opts.MoEMMQ = true
	default:
		t.Fatalf("YUUKA_TEST_MOE_ROUTE=%q is not 'mmq' or 'q8'", route)
	}
	return opts
}

// moERouteName names the routed-expert route the options select, for test logs.
func moERouteName(opts Options) string {
	if opts.MoEMMQ {
		return "mmq"
	}
	return "q8"
}

// TestGPUMoEMmqMatchesQ8 runs the same prompt through both routed-expert
// prefill routes and requires the same first-token logits and the same greedy
// continuation. Both routes consume the same Q8_1 activation values and decode
// the same expert digits, so only the summation order and the assignment tile
// differ; a decode, scale-slot, or scatter error moves the logits far more than
// the gate below.
//
// The golden continuation is the one TestGPUMoEGoldenGeneration and the CPU
// reference pin (" Paris, a city"). The runtimes are opened one after the other
// because each one uploads the full 10.7 GB expert set.
func TestGPUMoEMmqMatchesQ8(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode("The capital of France is")
	if err != nil {
		t.Fatal(err)
	}
	const capacity = 64
	const continuation = 4
	want := []int32{11751, 11, 264, 3177} // " Paris, a city"

	results := make(map[string][]float32, 2)
	tokens := make(map[string][]int32, 2)
	for _, route := range []string{"q8", "mmq"} {
		t.Setenv("YUUKA_TEST_MOE_ROUTE", route)
		opts := moERouteOptions(t)
		opts.KVType = tensor.F32
		opts.KVTypeSet = true
		rt, err := newRuntime(reader, model, capacity, opts)
		if err != nil {
			t.Skipf("HIP runtime unavailable: %v", err)
		}
		logits, err := rt.Forward(ids)
		if err != nil {
			rt.Close()
			t.Fatal(err)
		}
		results[route] = append([]float32(nil), logits...)
		got := make([]int32, 0, continuation)
		for i := range continuation {
			best := int32(top1(logits))
			got = append(got, best)
			if i+1 == continuation {
				break
			}
			logits, err = rt.Step(best)
			if err != nil {
				rt.Close()
				t.Fatal(err)
			}
		}
		tokens[route] = got
		if err := rt.Close(); err != nil {
			t.Fatal(err)
		}
		if !equalTokens(got, want) {
			t.Fatalf("%s route tokens %v, want %v", route, got, want)
		}
	}
	q8, mmq := results["q8"], results["mmq"]
	if top1(q8) != top1(mmq) {
		t.Fatalf("top-1 token differs: q8 %d mmq %d", top1(q8), top1(mmq))
	}
	diff := 0.0
	for i := range q8 {
		diff = math.Max(diff, math.Abs(float64(q8[i]-mmq[i])))
	}
	if diff > 0.05 {
		t.Fatalf("max logit diff between the expert routes %.6g exceeds 0.05", diff)
	}
	t.Logf("mmq vs q8: top-1 %d, max logit diff %.6g, tokens %v", top1(mmq), diff,
		tokens["mmq"])
}
