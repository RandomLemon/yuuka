//go:build hip && linux

package gpu

import (
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUProfilePrefill attributes one 512-row prefill chunk at several
// absolute positions with HIP events around every plan operation. It is
// opt-in through YUUKA_TEST_MODEL because it loads the full model.
// YUUKA_TEST_KV selects f32 (default) or f16 KV storage. The measurements are
// diagnostic, not a benchmark under docs/testing.md: events add overhead and
// clocks, power, and thermal state are not controlled.
func TestGPUProfilePrefill(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
	}
	kvType := tensor.F32
	switch value := os.Getenv("YUUKA_TEST_KV"); value {
	case "", "f32", "F32":
	case "f16", "F16":
		kvType = tensor.F16
	default:
		t.Fatalf("YUUKA_TEST_KV %q is not f32 or f16", value)
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
	if len(ids) < 512 {
		t.Skipf("prompt produced %d tokens, need 512", len(ids))
	}
	ids = ids[:512]
	const capacity = 8192 + 16
	rt, err := NewWithKVType(reader, model, capacity, kvType)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(512); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	plan, err := s.prefillPlan(512)
	if err != nil {
		t.Fatal(err)
	}
	if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, ids); err != nil {
		t.Fatal(err)
	}
	if err := plan.Execute(rt.model.stream, 0, 0, true, rt.model.event); err != nil {
		t.Fatal(err)
	}
	if err := rt.model.event.Wait(); err != nil {
		t.Fatal(err)
	}
	for _, position := range []uint32{0, 2048, 7680} {
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, ids); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if err := plan.Execute(rt.model.stream, 0, position, true, rt.model.event); err != nil {
			t.Fatal(err)
		}
		if err := rt.model.event.Wait(); err != nil {
			t.Fatal(err)
		}
		wall := time.Since(start)
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, ids); err != nil {
			t.Fatal(err)
		}
		times, err := plan.ExecuteProfiled(rt.model.stream, 0, position, true, rt.model.event)
		if err != nil {
			t.Fatal(err)
		}
		logPrefillProfile(t, kvType, position, wall, times)
	}
}

func logPrefillProfile(t *testing.T, kvType tensor.DType, position uint32,
	wall time.Duration, times []hip.OpTime) {
	t.Helper()
	names := map[uint32]string{
		1: "gemv", 2: "rmsnorm", 3: "l2norm", 4: "gated_rmsnorm",
		5: "rope", 6: "attention", 7: "conv1d", 8: "gdn", 9: "silu_mul",
		10: "add", 11: "copy", 12: "scale", 13: "embedding",
		14: "kv_write", 15: "rmsnorm_heads", 16: "gemm",
		17: "gemv_batch", 18: "rmsnorm_batch", 19: "rmsnorm_heads_batch",
		20: "rope_batch", 21: "kv_write_batch", 22: "attention_prefill",
		23: "recurrent_chunk", 24: "embedding_batch", 25: "cast_f16",
		26: "kv_write_paged", 27: "attention_paged", 28: "kv_write_batch_paged",
		29: "attention_prefill_paged",
	}
	totals := make(map[uint32]uint64)
	counts := make(map[uint32]int)
	var total uint64
	for _, item := range times {
		totals[item.Code] += item.Nanos
		counts[item.Code]++
		total += item.Nanos
	}
	type row struct {
		code  uint32
		ns    uint64
		count int
	}
	rows := make([]row, 0, len(totals))
	for code, ns := range totals {
		rows = append(rows, row{code, ns, counts[code]})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ns > rows[j].ns })
	t.Logf("prefill rows=512 kv=%v position=%d wall=%.3f ms profiled=%.3f ms ops=%d",
		kvType, position, float64(wall.Microseconds())/1000,
		float64(total)/1e6, len(times))
	for _, item := range rows {
		name := names[item.code]
		if name == "" {
			name = "unknown"
		}
		t.Logf("  %-22s n=%4d %9.3f ms %5.1f%%", name, item.count,
			float64(item.ns)/1e6, 100*float64(item.ns)/float64(total))
	}
}
