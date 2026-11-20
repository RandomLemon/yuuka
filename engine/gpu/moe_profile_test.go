//go:build hip && linux

package gpu

import (
	"encoding/binary"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// TestGPUProfileMoE attributes one qwen35moe prefill chunk and one decode step
// per planned operation. It is opt-in through YUUKA_TEST_MOE_MODEL because it
// loads the full model, and it is a diagnostic, not a benchmark: the per
// operation events add overhead and clocks are not pinned.
func TestGPUProfileMoE(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 128 {
		t.Skipf("prompt produced %d tokens, need 128", len(ids))
	}
	ids = ids[:128]
	const capacity = 512
	rt, err := NewWithKVType(reader, model, capacity, tensor.F16)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(128); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	plan, err := s.prefillPlan(128)
	if err != nil {
		t.Fatal(err)
	}

	// Prefill: one chunk, first the wall clock on the plain path and then the
	// per-operation attribution.
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, ids); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := plan.Execute(rt.model.stream, 0, 0, true, rt.model.event); err != nil {
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
	times, err := plan.ExecuteProfiled(rt.model.stream, 0, 0, true, rt.model.event)
	if err != nil {
		t.Fatal(err)
	}
	logMoEProfile(t, "prefill", 128, wall, times)

	// Decode: one step after the chunk, profiled the same way.
	if _, err := rt.ProfileStep(ids[len(ids)-1]); err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	times, err = rt.ProfileStep(ids[len(ids)-1])
	if err != nil {
		t.Fatal(err)
	}
	logMoEProfile(t, "decode", 1, time.Since(start), times)
	logMoEExpertCounts(t, "decode", 1, rt, s.model.act.moeCounts)
}

// logMoEExpertCounts reports how many experts the last route selected and how
// many assignments each one received. Every selected expert re-reads its whole
// packed weight set, so the distribution is what decides whether the expert
// stage streams the full expert weights or a fraction of them.
func logMoEExpertCounts(t *testing.T, stage string, rows int, rt *Runtime,
	counts *hip.Buffer) {
	t.Helper()
	experts := rt.model.config.ExpertCount
	raw := make([]byte, 4*experts)
	if err := rt.model.device.Download(raw, counts); err != nil {
		t.Fatal(err)
	}
	active, total, most := 0, 0, 0
	for i := 0; i < experts; i++ {
		count := int(int32(binary.LittleEndian.Uint32(raw[4*i:])))
		if count > 0 {
			active++
		}
		total += count
		if count > most {
			most = count
		}
	}
	t.Logf("%s route: rows=%d experts=%d active=%d assignments=%d used=%d max_per_expert=%d",
		stage, rows, experts, active, total, rt.model.config.ExpertUsedCount, most)
}

func logMoEProfile(t *testing.T, stage string, rows int, wall time.Duration,
	times []hip.OpTime) {
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
	rowsOut := make([]row, 0, len(totals))
	for code, ns := range totals {
		rowsOut = append(rowsOut, row{code, ns, counts[code]})
	}
	sort.Slice(rowsOut, func(i, j int) bool { return rowsOut[i].ns > rowsOut[j].ns })
	t.Logf("%s rows=%d wall=%.3f ms profiled=%.3f ms ops=%d", stage, rows,
		float64(wall.Microseconds())/1000, float64(total)/1e6, len(times))
	for _, item := range rowsOut {
		t.Logf("  %-24s n=%4d %9.3f ms %5.1f%%", hip.OpName(item.code), item.count,
			float64(item.ns)/1e6, 100*float64(item.ns)/float64(total))
	}
}
