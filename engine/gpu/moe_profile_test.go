//go:build hip && linux

package gpu

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
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
// operation events add overhead and clocks are not pinned. YUUKA_TEST_MOE_ROUTE
// selects the routed-expert route so the same chunk can be attributed under the
// Q8_1 dp4a and the int8 WMMA route in the same session.
func TestGPUProfileMoE(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 90))
	if err != nil {
		t.Fatal(err)
	}
	const capacity = 1024
	chunks := []int{128, 512}
	if len(ids) < chunks[len(chunks)-1] {
		t.Skipf("prompt produced %d tokens, need %d", len(ids), chunks[len(chunks)-1])
	}
	// YUUKA_TEST_MOE_ROUTE selects the routed-expert route, so one session can
	// attribute the same chunk under the Q8_1 and the int8 WMMA route.
	opts := moERouteOptions(t)
	opts.KVType = tensor.F16
	opts.KVTypeSet = true
	rt, err := newRuntime(reader, model, capacity, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	t.Logf("routed-expert route: %s", moERouteName(opts))
	s := rt.session
	var times []hip.OpTime
	for _, rows := range chunks {
		if err := rt.SetPrefillChunk(rows); err != nil {
			t.Fatal(err)
		}
		plan, err := s.prefillPlan(rows)
		if err != nil {
			t.Fatal(err)
		}
		chunkIDs := ids[:rows]

		// Prefill: one chunk, first the wall clock on the plain path and then
		// the per-operation attribution.
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, chunkIDs); err != nil {
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
		if err := rt.model.device.UploadInt32(rt.model.prefill.tokens, chunkIDs); err != nil {
			t.Fatal(err)
		}
		times, err = plan.ExecuteProfiled(rt.model.stream, 0, 0, true, rt.model.event)
		if err != nil {
			t.Fatal(err)
		}
		logMoEProfile(t, "prefill", rows, wall, times)
	}

	// Decode: one step after the widest chunk, profiled the same way.
	if _, err := rt.ProfileStep(ids[len(ids)-1]); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	times, err = rt.ProfileStep(ids[len(ids)-1])
	if err != nil {
		t.Fatal(err)
	}
	logMoEProfile(t, "decode", 1, time.Since(start), times)
	logMoEExpertCounts(t, "decode", 1, rt, s.model.act.moeCounts)
}

// moePrefillBenchmarkText is the fixed benchmark stream the prefill comparison
// driver tokenizes, so this attribution covers the same prefixes the release
// score uses. It is repeated rather than loaded from the driver's token file so
// the profile stays self-contained; the prefixes are identical because BPE
// encoding is deterministic for a leading token prefix.
const moePrefillBenchmarkText = "The quick brown fox jumps over the lazy dog. "

// TestGPUProfileMoEPrefillLong attributes the fixed 8192-token benchmark stream
// chunk by chunk at the release chunk size. Unlike TestGPUProfileMoE, every
// chunk is executed at its populated absolute position, so the attention and
// recurrent stages carry the same context they carry in the release run. It is
// opt-in through YUUKA_TEST_MOE_MODEL and is a diagnostic: the events add
// submission overhead and the clocks are not pinned.
func TestGPUProfileMoEPrefillLong(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	const length = 8192
	const chunk = 512
	ids, err := tok.Encode(strings.Repeat(moePrefillBenchmarkText, length))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < length {
		t.Fatalf("benchmark prompt produced %d tokens, need %d", len(ids), length)
	}
	ids = ids[:length]
	capacity := length + 64
	opts := moERouteOptions(t)
	opts.KVType = tensor.F16
	opts.KVTypeSet = true
	// The step-1 state check compares the recurrence change against the frozen
	// build. Forcing the tiled attention keeps the attention arithmetic on the
	// pre-change kernel while the rest of the candidate plan is active; the
	// release runs leave this unset.
	if os.Getenv("YUUKA_TEST_MOE_FORCE_TILED") != "" {
		opts.ForceTiledAttention = true
	}
	rt, err := newRuntime(reader, model, capacity, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	t.Logf("routed-expert route: %s", moERouteName(opts))
	if err := rt.SetPrefillChunk(chunk); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	plan, err := s.prefillPlan(chunk)
	if err != nil {
		t.Fatal(err)
	}
	cfg := rt.model.config
	preparedWidth := uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + 2*cfg.SSMTimeStepRank)
	rows := uint64(chunk)
	t.Logf("prepared span=%d floats/row ffn_gate=%.0f q=%.0f arch=%q state=%d rows=%d",
		preparedWidth, float64(rt.model.prefill.ffnGate.Bytes())/4/float64(rows),
		float64(rt.model.prefill.q.Bytes())/4/float64(rows),
		rt.model.device.Info().GCNArch, cfg.SSMStateSize, chunk)
	names := decodeBufferNames(rt.model, s)
	shadow := plan.Ops()
	for _, n := range []int{512, 2048, 8192} {
		t.Run(fmt.Sprintf("N%d", n), func(t *testing.T) {
			if err := s.Reset(); err != nil {
				t.Fatal(err)
			}
			var wall time.Duration
			samples := make([]decodeSample, 0, 4*len(shadow))
			for start := 0; start < n; start += chunk {
				if err := rt.model.device.UploadInt32(rt.model.prefill.tokens,
					ids[start:start+chunk]); err != nil {
					t.Fatal(err)
				}
				last := start+chunk == n
				begin := time.Now()
				times, err := plan.ExecuteProfiled(rt.model.stream, 0, uint32(start),
					last, rt.model.event)
				if err != nil {
					t.Fatal(err)
				}
				wall += time.Since(begin)
				if err := rt.model.event.Wait(); err != nil {
					t.Fatal(err)
				}
				if len(times) > len(shadow) {
					t.Fatalf("position %d executed %d operations, plan holds %d",
						start, len(times), len(shadow))
				}
				if last && len(times) != len(shadow) {
					t.Fatalf("final chunk executed %d operations, plan holds %d",
						len(times), len(shadow))
				}
				for i, item := range times {
					if shadow[i].Code != item.Code || (!last && shadow[i].Tail()) {
						t.Fatalf("position %d: executed operation %d is %s (tail=%v),"+
							" plan prefix has %s", start, i, hip.OpName(item.Code),
							shadow[i].Tail(), shadow[i].Name())
					}
					samples = append(samples, decodeSample{
						index: i,
						info:  shadow[i],
						ms:    float64(item.Nanos) / 1e6,
					})
					if item.Code == hip.OpGemm || item.Code == hip.OpGemmSwiglu {
						info := shadow[i]
						sample := &samples[len(samples)-1]
						sample.tensor, sample.role = resolveOperationNames(info, names)
						sample.dtype = tensor.DType(info.U[0])
						sample.hasDType = true
						sample.nIn = info.N[0]
						sample.nOut = info.N[1]
						sample.rows = info.N[2]
						sample.matmul = true
					}
				}
				t.Logf("chunk position=%d final=%v operations=%d wall=%.3f ms",
					start, last, len(times), float64(time.Since(begin).Microseconds())/1000)
			}
			logMoEPrefillLong(t, n, wall, samples, shadow)
		})
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	t.Run("continuation", func(t *testing.T) {
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
		if logits, err := s.Prefill(ids[:chunk], false); err != nil {
			t.Fatal(err)
		} else if logits != nil {
			t.Fatalf("nonfinal prefill returned %d logits", len(logits))
		}
		logits, err := s.Prefill(ids[chunk:2*chunk], true)
		if err != nil {
			t.Fatal(err)
		}
		if len(logits) != model.Config.VocabSize {
			t.Fatalf("final prefill returned %d logits, want %d", len(logits),
				model.Config.VocabSize)
		}
		for i, value := range logits {
			if value != value || value > 3.4e38 || value < -3.4e38 {
				t.Fatalf("logit %d is not finite", i)
			}
		}
		snapshot, err := s.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		logitDigest := sha256.Sum256(float32Bytes(logits))
		stateDigest := sha256.Sum256(snapshot)
		t.Logf("continuation position=%d logits_sha256=%x state_sha256=%x logits=%d state=%d",
			s.Position(), logitDigest, stateDigest, len(logits), len(snapshot))
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
	})
}

func float32Bytes(values []float32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(value))
	}
	return out
}

// logMoEPrefillLong reports the aggregated operation cost over every populated
// chunk of one prompt length. Dense GEMM rows keep the tensor leaf, weight
// dtype, selected path, and shape, because one OpGemm code covers projections
// whose subtiles are not distinguishable from the plan.
func logMoEPrefillLong(t *testing.T, n int, wall time.Duration,
	samples []decodeSample, shadow []hip.OpInfo) {
	t.Helper()
	type gemmKey struct {
		tensor string
		dtype  tensor.DType
		path   uint32
		nIn    int64
		nOut   int64
		rows   int64
	}
	totals := make(map[uint32]uint64)
	counts := make(map[uint32]int)
	var total uint64
	for _, sample := range samples {
		totals[sample.info.Code] += uint64(sample.ms * 1e6)
		counts[sample.info.Code]++
		total += uint64(sample.ms * 1e6)
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
	t.Logf("prefill N=%d wall=%.3f ms profiled=%.3f ms ops=%d chunks=%d",
		n, float64(wall.Microseconds())/1000, float64(total)/1e6, len(samples),
		(len(shadow)+n-1)/n)
	for _, item := range rowsOut {
		t.Logf("  %-24s n=%4d %9.3f ms %5.1f%%", hip.OpName(item.code), item.count,
			float64(item.ns)/1e6, 100*float64(item.ns)/float64(total))
	}
	gemms := make(map[gemmKey]struct {
		ns    uint64
		count int
	})
	for _, sample := range samples {
		if sample.info.Code != hip.OpGemm && sample.info.Code != hip.OpGemmSwiglu {
			continue
		}
		key := gemmKey{sample.tensor, sample.dtype, sample.info.U[1],
			sample.nIn, sample.nOut, sample.rows}
		entry := gemms[key]
		entry.ns += uint64(sample.ms * 1e6)
		entry.count++
		gemms[key] = entry
	}
	type gemmRow struct {
		key   gemmKey
		ns    uint64
		count int
	}
	gemmRows := make([]gemmRow, 0, len(gemms))
	for key, entry := range gemms {
		gemmRows = append(gemmRows, gemmRow{key, entry.ns, entry.count})
	}
	sort.Slice(gemmRows, func(i, j int) bool { return gemmRows[i].ns > gemmRows[j].ns })
	for _, item := range gemmRows {
		t.Logf("  gemm %-28s dtype=%-4v path=%d %5d->%-5d r%-5d n=%4d %9.3f ms",
			item.key.tensor, item.key.dtype, item.key.path, item.key.nIn,
			item.key.nOut, item.key.rows, item.count, float64(item.ns)/1e6)
	}
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
