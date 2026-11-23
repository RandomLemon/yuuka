//go:build hip && linux

package gpu

import (
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUProfilePrefill attributes a prefill chunk (512 rows by default) at
// several absolute positions with HIP events around every plan operation. It is
// opt-in through YUUKA_TEST_MODEL because it loads the full model.
// YUUKA_TEST_KV selects f32 (the profile baseline) or f16 KV storage. These
// diagnostic timings include HIP event overhead and use uncontrolled clocks,
// power, and thermal conditions.
// YUUKA_TEST_PROFILE_ROWS selects a comma-separated list of row counts.
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
	rowCounts := []int{512}
	if value := os.Getenv("YUUKA_TEST_PROFILE_ROWS"); value != "" {
		rowCounts = nil
		for _, field := range strings.Split(value, ",") {
			rows, err := strconv.Atoi(strings.TrimSpace(field))
			if err != nil || rows < 1 || rows > maxPrefillChunk {
				t.Fatalf("invalid YUUKA_TEST_PROFILE_ROWS entry %q", field)
			}
			rowCounts = append(rowCounts, rows)
		}
	}
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", maxPrefillChunk))
	if err != nil {
		t.Fatal(err)
	}
	const capacity = 8192 + 16
	rt, err := NewWithKVType(reader, model, capacity, kvType)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	for _, rows := range rowCounts {
		if rows > len(ids) {
			t.Fatalf("prompt produced %d tokens, need %d", len(ids), rows)
		}
		t.Run(strconv.Itoa(rows), func(t *testing.T) {
			profilePrefillRows(t, rt, kvType, ids[:rows])
		})
	}
}

func profilePrefillRows(t *testing.T, rt *Runtime, kvType tensor.DType, ids []int32) {
	t.Helper()
	rows := len(ids)
	if err := rt.SetPrefillChunk(rows); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	s.syncPrefillPlans()
	plan, err := s.prefillPlan(rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, ids); err != nil {
		t.Fatal(err)
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := plan.Execute(rt.model.stream, 0, 0, true, rt.model.event); err != nil {
		t.Fatal(err)
	}
	if err := rt.model.event.Wait(); err != nil {
		t.Fatal(err)
	}
	for _, position := range []uint32{0, 2048, 7680} {
		if int(position)+rows > s.capacity {
			continue
		}
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
		logPrefillProfile(t, rows, kvType, position, wall, times)
	}
}

func logPrefillProfile(t *testing.T, tokenRows int, kvType tensor.DType, position uint32,
	wall time.Duration, times []hip.OpTime) {
	t.Helper()
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
	t.Logf("prefill rows=%d kv=%v position=%d wall=%.3f ms profiled=%.3f ms ops=%d",
		tokenRows, kvType, position, float64(wall.Microseconds())/1000,
		float64(total)/1e6, len(times))
	for _, item := range rows {
		t.Logf("  %-22s n=%4d %9.3f ms %5.1f%%", hip.OpName(item.code), item.count,
			float64(item.ns)/1e6, 100*float64(item.ns)/float64(total))
	}
}
