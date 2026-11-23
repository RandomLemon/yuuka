//go:build hip && linux

package gpu

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// The recorded decode attribution the gates below compare against, measured on
// one gfx1151 device with Qwen3.5-4B-Q4_K_M. The operation count is structural
// and exact; the timings track the current tree and are re-recorded when a
// decode kernel lands, because each change moves them by design: 97.6 ms before
// the streaming matvec family, 89.4 ms after the streaming LM head, 31.6 ms
// after the matvec family moved to the streaming kernel, 24.3 ms after the fused
// gated delta-net, 23.4 ms after the residual and gate/up fusions, 17.3 ms after
// decode attention was split over the key range and the matvec was remapped to
// one wave per row, and 17.1 ms after the norms were rewritten around a wave
// reduction and the beta/alpha projections were merged. The measured history
// stays in docs/benchmarks.md.
const (
	decodeBaselineOps     = 355
	decodeBaselineTotalMS = 17.070
	decodeBaselineGemvMS  = 15.400
)

// Gate bands. Device clocks are not pinned, so the absolute total is only gated
// loosely; the operation count is structural and exact, and the GEMV share is
// the invariant the decode kernel work is built on.
const (
	decodeTotalLowRatio  = 0.5
	decodeTotalHighRatio = 2.0
	decodeShareTolerance = 0.05
)

// TestGPUProfileDecode attributes one decode step per planned operation and per
// tensor. It reproduces the attribution tables of docs/benchmarks.md and gates
// the result against the recorded attribution. Event instrumentation adds
// a submission gap per operation, so the absolute numbers are a diagnosis, not
// a release measurement; see docs/testing.md section 6.3.
func TestGPUProfileDecode(t *testing.T) {
	reader, tok, model, prompt := goldenPrompt(t)
	// YUUKA_TEST_DECODE_CONTEXT extends the prompt so the same attribution can be
	// produced at a longer decode context, which is how the split-key attention
	// per-layer cost is gated. Unset profiles the short golden prompt.
	context := 64
	if value := os.Getenv("YUUKA_TEST_DECODE_CONTEXT"); value != "" {
		if _, err := fmt.Sscanf(value, "%d", &context); err != nil || context < 8 {
			t.Fatalf("YUUKA_TEST_DECODE_CONTEXT=%q is not a token count", value)
		}
	}
	if context > len(prompt) {
		ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", context/8+2))
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) < context {
			t.Fatalf("tokenizer produced %d tokens, need %d", len(ids), context)
		}
		prompt = ids[:context]
	}
	// YUUKA_TEST_DECODE_INT8 selects the int8 decode matvec so the
	// same attribution can be produced for each variant: "fused" quantizes inside
	// the matvec block, "prequantized" submits one quantization operation per
	// activation group. Unset keeps the default FP32 streaming route.
	opts := Options{}
	// YUUKA_TEST_DECODE_KV selects the KV pool format, so the same attribution and
	// the same attention gate run for F32, F16, and Q8_0.
	switch kv := os.Getenv("YUUKA_TEST_DECODE_KV"); kv {
	case "":
	case "f32":
		opts.KVType = tensor.F32
		opts.KVTypeSet = true
	case "f16":
		opts.KVType = tensor.F16
		opts.KVTypeSet = true
	case "q8_0":
		opts.KVType = tensor.Q8_0
		opts.KVTypeSet = true
	default:
		t.Fatalf("YUUKA_TEST_DECODE_KV=%q is not f32, f16, or q8_0", kv)
	}
	route := "fp32 streaming"
	// YUUKA_TEST_DECODE_UNFUSED=1 builds the plan without the residual epilogue
	// and gate/up pair, which is how the same-session A/B measured the fusions.
	if os.Getenv("YUUKA_TEST_DECODE_UNFUSED") != "" {
		opts.NoResidualFusion = true
		opts.NoPairFusion = true
		route += ", fusions off"
	}
	switch route8 := os.Getenv("YUUKA_TEST_DECODE_INT8"); route8 {
	case "":
	case "fused":
		opts.Int8Matvec = true
		route = "int8 fused"
	case "prequantized":
		opts.Int8Matvec = true
		opts.PrequantizedInt8 = true
		route = "int8 prequantized"
	default:
		t.Fatalf("YUUKA_TEST_DECODE_INT8=%q is not 'fused' or 'prequantized'", route8)
	}
	rt, err := newRuntime(reader, model, context+64, opts)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	t.Logf("decode matvec route: %s, decode context %d", route, context)
	defer rt.Close()
	logits, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	// One unprofiled decode step warms the decode-only weight rows and the KV
	// pages, so the attribution does not carry the first-touch fault of
	// blk.0.ssm_beta, which is otherwise the slowest single operation.
	warm, err := rt.Step(argmax(logits))
	if err != nil {
		t.Fatal(err)
	}
	token := argmax(warm)
	times, err := rt.ProfileStep(token)
	if err != nil {
		t.Fatal(err)
	}
	if len(times) == 0 {
		t.Fatal("profile returned no operations")
	}
	shadow := rt.session.plan.Ops()
	if len(shadow) != len(times) {
		t.Fatalf("the decode plan holds %d operations and the profile returned %d;"+
			" the shadow and the submission disagree", len(shadow), len(times))
	}
	names := decodeBufferNames(rt.model, rt.session)
	samples := decodeSamples(shadow, times, names)
	report := newDecodeReport(samples)
	printDecodeReport(t, rt, report)
	// The operation count and the matvec share are recorded for the default
	// route; the int8 and unfused variants add quantization operations and drop
	// fusions by design, so they are reported without that gate.
	if route == "fp32 streaming" {
		checkDecodeGates(t, report)
	}
	checkDecodeAttentionGate(t, report, context)
}

// decodeAttentionPerLayerGate is the split-key attention cost a 512-token decode
// context must stay under. The serial position loop cost 1.04 ms per layer at
// that context.
const decodeAttentionPerLayerGate = 0.05

// checkDecodeAttentionGate enforces the split-key acceptance cost at the
// 512-token context it was written for and reports the cost at longer contexts.
// Those rows are measurements, not gates: the cost is the KV pool bytes, and at
// 2048 tokens one F32 layer already moves 16 MB, which is 68 us per layer at the
// 235 GB/s read ceiling regardless of the kernel.
func checkDecodeAttentionGate(t *testing.T, r *decodeReport, context int) {
	t.Helper()
	for _, row := range r.codes {
		if row.code != hip.OpAttentionPaged || row.ops == 0 {
			continue
		}
		perLayer := row.ms / float64(row.ops)
		if context == 512 && perLayer > decodeAttentionPerLayerGate {
			t.Fatalf("decode attention costs %.4f ms per layer at a 512-token context,"+
				" the gate is %.3f ms", perLayer, decodeAttentionPerLayerGate)
		}
		t.Logf("decode attention %.4f ms per layer at a %d-token context"+
			" (%s %.3f ms)", perLayer, context,
			map[bool]string{true: "gate", false: "recorded"}[context == 512],
			decodeAttentionPerLayerGate)
	}
}

// decodeSample is one profiled operation with the model buffer that produced it
// resolved to a name.
type decodeSample struct {
	index int
	info  hip.OpInfo
	// role is the resolved name of the buffer that names the operation: the
	// packed weights for a matmul, otherwise the first slot that is not unused.
	role string
	// tensor is the leaf name of role, with the layer prefix removed, so the
	// 32 per-layer projections of one kind aggregate into one row.
	tensor string
	dtype  tensor.DType
	// hasDType is false for the operations whose u[0] is not a dtype.
	hasDType bool
	nIn      int64
	nOut     int64
	rows     int64
	ms       float64
	// moved is the bytes the operation is expected to move; the GEMV family
	// reads its packed weights once, and the delta-net forms read and write the
	// recurrent state once.
	moved  uint64
	matmul bool
	// stateful marks the operations whose traffic is recurrent or convolutional
	// state, which the summary reports as one line.
	stateful bool
}

func (s decodeSample) gbps() float64 {
	if s.ms <= 0 || s.moved == 0 {
		return 0
	}
	return float64(s.moved) / (s.ms * 1e6)
}

// shapeText describes the dimensions the attribution cares about: the matrix
// for a matmul, the state geometry for the gated delta-net, and the element
// count for everything else.
func (s decodeSample) shapeText() string {
	switch {
	case s.info.Code == hip.OpGemvPair:
		return fmt.Sprintf("%d->%d x2", s.nIn, s.nOut)
	case s.matmul:
		text := fmt.Sprintf("%d->%d", s.nIn, s.nOut)
		if s.rows > 1 {
			text = fmt.Sprintf("%s r%d", text, s.rows)
		}
		return text
	case s.info.Code == hip.OpGDN || s.info.Code == hip.OpRecurrentToken:
		return fmt.Sprintf("%d g x %d h x %d", s.info.N[0], s.info.N[1], s.info.N[2])
	case s.info.Code == hip.OpConvL2Norm:
		return fmt.Sprintf("%d ch k%d", s.info.N[0], s.info.N[1])
	case s.info.Code == hip.OpAttentionPaged || s.info.Code == hip.OpAttentionPrefillPaged ||
		s.info.Code == hip.OpAttention || s.info.Code == hip.OpAttentionPrefill:
		return fmt.Sprintf("%d kv x %d h x %d d", s.info.N[0], s.info.N[1], s.info.N[2])
	default:
		return fmt.Sprintf("%d", s.nIn)
	}
}

// decodeSamples pairs the plan shadow with the profiled times and resolves the
// buffers each operation touches.
func decodeSamples(shadow []hip.OpInfo, times []hip.OpTime, names map[uintptr]string) []decodeSample {
	samples := make([]decodeSample, 0, len(times))
	for i, item := range times {
		info := shadow[i]
		s := decodeSample{
			index: i,
			info:  info,
			ms:    float64(item.Nanos) / 1e6,
		}
		s.tensor, s.role = resolveOperationNames(info, names)
		s.moved, s.matmul, s.rows = operationTraffic(info)
		switch info.Code {
		case hip.OpGDN, hip.OpRecurrentToken, hip.OpConv1D, hip.OpConvL2Norm:
			s.stateful = true
		}
		switch info.Code {
		case hip.OpGemv, hip.OpGemvBatch, hip.OpGemm, hip.OpLMHead, hip.OpGemvQ8_1,
			hip.OpGemmSwiglu, hip.OpQuantQ8_1, hip.OpGemvResidual, hip.OpGemvPair,
			hip.OpMoEGateUpSelected, hip.OpMoEDownSelected, hip.OpMoEGateUpQ8,
			hip.OpMoEDownQ8, hip.OpMoEGateUpMMQ, hip.OpMoEDownMMQ:
			s.dtype = tensor.DType(info.U[0])
			s.hasDType = true
		}
		s.nIn = info.N[0]
		s.nOut = info.N[1]
		if s.matmul && s.rows == 0 {
			s.rows = 1
		}
		samples = append(samples, s)
	}
	return samples
}

// resolveOperationNames returns the leaf tensor name and the full buffer name of
// the operation. Slots are scanned in role priority order so a matmul is named
// by its weights, a stateful operation by its state, and an elementwise
// operation by its output.
func resolveOperationNames(info hip.OpInfo, names map[uintptr]string) (tensorName, role string) {
	priority := []hip.OpRole{
		hip.RoleWeight, hip.RoleState, hip.RoleOutput, hip.RoleAccumulator,
		hip.RoleActivation, hip.RolePool, hip.RoleTable, hip.RoleControl,
		hip.RoleScratch,
	}
	for _, want := range priority {
		for slot := range info.Bufs {
			if info.Roles[slot] != want {
				continue
			}
			if name := names[info.Bufs[slot]]; name != "" {
				return leafTensorName(name), name
			}
		}
	}
	return "?", "?"
}

// leafTensorName removes the "blk.N." prefix so the per-layer copies of one
// projection aggregate together.
func leafTensorName(name string) string {
	if bracket := strings.IndexByte(name, '['); bracket >= 0 {
		return name[:bracket]
	}
	if !strings.HasPrefix(name, "blk.") {
		return name
	}
	if dot := strings.Index(name[len("blk."):], "."); dot >= 0 {
		return name[len("blk.")+dot+1:]
	}
	return name
}

// operationTraffic returns the bytes an operation is expected to move, whether
// it is a packed-matrix read, and the row count it covers.
func operationTraffic(info hip.OpInfo) (moved uint64, matmul bool, rows int64) {
	batch := info.N[2]
	if batch <= 0 {
		batch = 1
	}
	switch info.Code {
	case hip.OpGemv, hip.OpLMHead, hip.OpGemvQ8_1, hip.OpGemvResidual:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, 1
	case hip.OpGemvPair:
		// Two packed matrices read by one launch; they may differ in dtype and
		// width.
		rowA, errA := tensor.DType(info.U[0]).RowBytes(info.N[0])
		rowB, errB := tensor.DType(info.U[1]).RowBytes(info.N[0])
		if errA != nil || errB != nil {
			return 0, false, 0
		}
		return uint64(rowA)*uint64(info.N[1]) + uint64(rowB)*uint64(info.N[2]), true, 1
	case hip.OpGemvBatch:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGemm:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGemmSwiglu:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return 2 * uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGDN:
		// groups, heads, state size: one 4-byte element per state cell, moved
		// by the decay, the two S^T products, and the rank-1 update.
		state := uint64(info.N[1]) * uint64(info.N[2]) * uint64(info.N[2]) * 4
		return 5 * state, false, 1
	case hip.OpRecurrentToken:
		// The fused per-token operator reads the head state into registers once
		// and writes it back once.
		state := uint64(info.N[1]) * uint64(info.N[2]) * uint64(info.N[2]) * 4
		return 2 * state, false, 1
	case hip.OpConvL2Norm:
		// The convolution ring state is read and written, the weights and the
		// qkv row are read, the result is written, and the q and k groups are
		// normalized in place.
		channels := uint64(info.N[0])
		kernel := uint64(info.N[1])
		qk := uint64(info.N[2])
		return 2*channels*(kernel-1)*4 + channels*kernel*4 + channels*4 + channels*4 +
			2*qk*4, false, 1
	case hip.OpEmbedding, hip.OpEmbeddingBatch:
		// One packed weight row per token, read by the gather.
		return uint64(info.N[1]) * uint64(batch), false, batch
	default:
		// The remaining operations move activation-sized buffers, which the
		// plan does not describe well enough to publish a bandwidth for.
		return 0, false, batch
	}
}

// decodeTensorRow aggregates every operation that reads the same tensor.
type decodeTensorRow struct {
	tensor   string
	dtype    tensor.DType
	shape    string
	ops      int
	ms       float64
	moved    uint64
	matmul   bool
	stateful bool
}

// decodeDTypeRow aggregates the matmul family by weight dtype.
type decodeDTypeRow struct {
	dtype tensor.DType
	ops   int
	ms    float64
	moved uint64
	rows  int64
}

// decodeCodeRow aggregates by operation code, which is the per-operation table
// of docs/benchmarks.md.
type decodeCodeRow struct {
	code uint32
	ops  int
	ms   float64
}

type decodeReport struct {
	samples []decodeSample
	codes   []decodeCodeRow
	tensors []decodeTensorRow
	dtypes  []decodeDTypeRow
	total   float64
	matmul  float64
}

func newDecodeReport(samples []decodeSample) *decodeReport {
	r := &decodeReport{samples: samples}
	codes := map[uint32]*decodeCodeRow{}
	tensors := map[string]*decodeTensorRow{}
	dtypes := map[tensor.DType]*decodeDTypeRow{}
	for _, s := range samples {
		r.total += s.ms
		if s.info.Code == hip.OpGemv || s.info.Code == hip.OpLMHead ||
			s.info.Code == hip.OpGemvBatch || s.info.Code == hip.OpGemm ||
			s.info.Code == hip.OpGemvQ8_1 || s.info.Code == hip.OpGemvResidual ||
			s.info.Code == hip.OpGemvPair {
			r.matmul += s.ms
		}
		if row, ok := codes[s.info.Code]; ok {
			row.ops++
			row.ms += s.ms
		} else {
			codes[s.info.Code] = &decodeCodeRow{code: s.info.Code, ops: 1, ms: s.ms}
		}
		key := fmt.Sprintf("%s|%s|%d|%d", s.tensor, s.dtype, s.nIn, s.nOut)
		if row, ok := tensors[key]; ok {
			row.ops++
			row.ms += s.ms
			row.moved += s.moved
		} else {
			tensors[key] = &decodeTensorRow{
				tensor: s.tensor, dtype: s.dtype, shape: s.shapeText(),
				ops: 1, ms: s.ms, moved: s.moved, matmul: s.matmul,
				stateful: s.stateful,
			}
		}
		if s.matmul && s.hasDType {
			if row, ok := dtypes[s.dtype]; ok {
				row.ops++
				row.ms += s.ms
				row.moved += s.moved
			} else {
				dtypes[s.dtype] = &decodeDTypeRow{
					dtype: s.dtype, ops: 1, ms: s.ms, moved: s.moved,
				}
			}
		}
	}
	for _, row := range codes {
		r.codes = append(r.codes, *row)
	}
	sort.Slice(r.codes, func(i, j int) bool { return r.codes[i].ms > r.codes[j].ms })
	for _, row := range tensors {
		r.tensors = append(r.tensors, *row)
	}
	sort.Slice(r.tensors, func(i, j int) bool { return r.tensors[i].ms > r.tensors[j].ms })
	for _, row := range dtypes {
		r.dtypes = append(r.dtypes, *row)
	}
	sort.Slice(r.dtypes, func(i, j int) bool { return r.dtypes[i].ms > r.dtypes[j].ms })
	return r
}

// printDecodeReport writes the four attribution tables. The per-operation table
// is one line per plan index, sorted by time, so a slow operation can be named
// without a counter database.
func printDecodeReport(t *testing.T, rt *Runtime, r *decodeReport) {
	t.Helper()
	info := rt.model.device.Info()
	// HIP reports half of rocminfo's compute-unit count on this APU, so the
	// figure is labelled with its source instead of being compared.
	t.Logf("device %s (%s), HIP reports %d compute units, HIP %s", info.Name,
		info.GCNArch, info.ComputeUnits, info.HIPVersion)
	t.Logf("profiled %d operations, total %.3f ms; recorded baseline %.3f ms in"+
		" %d operations", len(r.samples), r.total, decodeBaselineTotalMS,
		decodeBaselineOps)

	t.Logf("by operation code:")
	t.Logf("%-20s %5s %10s %7s", "operation", "ops", "ms", "share")
	for _, row := range r.codes {
		t.Logf("%-20s %5d %10.3f %6.1f%%", hip.OpName(row.code), row.ops, row.ms,
			100*row.ms/r.total)
	}

	t.Logf("by tensor (matmul weights read once per token, delta-net state read and" +
		" written once):")
	t.Logf("%-26s %-6s %-14s %5s %9s %8s %8s %7s", "tensor", "dtype", "shape",
		"ops", "ms", "MB", "GB/s", "share")
	var matmulMS, stateMS float64
	var matmulMoved, stateMoved uint64
	for _, row := range r.tensors {
		shape := row.shape
		dtype := "-"
		if row.matmul {
			matmulMS += row.ms
			matmulMoved += row.moved
			dtype = row.dtype.String()
		}
		if row.stateful {
			stateMS += row.ms
			stateMoved += row.moved
		}
		gbps := 0.0
		if row.ms > 0 && row.moved != 0 {
			gbps = float64(row.moved) / (row.ms * 1e6)
		}
		t.Logf("%-26s %-6s %-14s %5d %9.3f %8.1f %8.1f %6.1f%%", row.tensor,
			dtype, shape, row.ops, row.ms, float64(row.moved)/1e6, gbps,
			100*row.ms/r.total)
	}
	t.Logf("%-26s %-6s %-14s %5d %9.3f %8.1f %8.1f %6.1f%%", "profiled total", "", "",
		len(r.samples), r.total, float64(matmulMoved+stateMoved)/1e6,
		float64(matmulMoved+stateMoved)/(r.total*1e6), 100.0)
	t.Logf("matmul family: %.3f ms of %.3f ms (%.1f%%), %.1f MB of packed weights,"+
		" %.1f GB/s effective", matmulMS, r.total, 100*matmulMS/r.total,
		float64(matmulMoved)/1e6, float64(matmulMoved)/(matmulMS*1e6))
	if stateMS > 0 {
		t.Logf("delta-net state and convolution: %.3f ms of %.3f ms (%.1f%%), %.1f MB"+
			" moved, %.1f GB/s effective", stateMS, r.total, 100*stateMS/r.total,
			float64(stateMoved)/1e6, float64(stateMoved)/(stateMS*1e6))
	}

	t.Logf("by weight dtype (matmul family only):")
	t.Logf("%-8s %5s %10s %8s %10s", "dtype", "ops", "ms", "MB", "GB/s")
	for _, row := range r.dtypes {
		gbps := 0.0
		if row.ms > 0 {
			gbps = float64(row.moved) / (row.ms * 1e6)
		}
		t.Logf("%-8s %5d %10.3f %8.1f %10.1f", row.dtype, row.ops, row.ms,
			float64(row.moved)/1e6, gbps)
	}

	sorted := make([]decodeSample, len(r.samples))
	copy(sorted, r.samples)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ms > sorted[j].ms })
	t.Logf("per operation index, sorted by time:")
	t.Logf("%-5s %-16s %-30s %-16s %9s %8s", "op", "kind", "buffer", "shape", "ms", "GB/s")
	for _, sample := range sorted {
		gbpsText := "-"
		if gbps := sample.gbps(); gbps > 0 {
			gbpsText = fmt.Sprintf("%.1f", gbps)
		}
		t.Logf("%-5d %-16s %-30s %-16s %9.4f %8s", sample.index, sample.info.Name(),
			sample.role, sample.shapeText(), sample.ms, gbpsText)
	}
}

// checkDecodeGates compares the measured attribution against the recorded one.
func checkDecodeGates(t *testing.T, r *decodeReport) {
	t.Helper()
	if got := len(r.samples); got != decodeBaselineOps {
		t.Fatalf("the decode step runs %d operations, the recorded attribution is %d;"+
			" update the attribution in docs/benchmarks.md before changing the plan shape", got,
			decodeBaselineOps)
	}
	if r.total < decodeTotalLowRatio*decodeBaselineTotalMS ||
		r.total > decodeTotalHighRatio*decodeBaselineTotalMS {
		t.Fatalf("profiled decode step is %.3f ms, which is outside the recorded"+
			" %.1fx..%.1fx band of %.3f ms; re-record the attribution in"+
			" docs/benchmarks.md for this device", r.total,
			decodeTotalLowRatio, decodeTotalHighRatio, decodeBaselineTotalMS)
	}
	share := r.matmul / r.total
	baselineShare := decodeBaselineGemvMS / decodeBaselineTotalMS
	if share < baselineShare-decodeShareTolerance || share > baselineShare+decodeShareTolerance {
		t.Fatalf("the GEMV family is %.1f%% of the decode step, the recorded attribution"+
			" is %.1f%%; a change moved the bottleneck, so update docs/benchmarks.md"+
			" and the gate bands", 100*share, 100*baselineShare)
	}
	t.Logf("gate: %d operations (recorded %d), total %.3f ms (recorded %.3f ms),"+
		" GEMV share %.1f%% (recorded %.1f%%)", len(r.samples), decodeBaselineOps,
		r.total, decodeBaselineTotalMS, 100*share, 100*baselineShare)
}

// decodeBufferNames labels every buffer the decode plan can reference so a
// planned operation can be named. Weights keep their GGUF tensor names; the
// activation, pool, and session buffers get a stable role name.
func decodeBufferNames(m *Model, s *Session) map[uintptr]string {
	names := make(map[uintptr]string, len(m.weights)+64)
	add := func(b *hip.Buffer, name string) {
		if b == nil {
			return
		}
		if id := b.ID(); id != 0 {
			names[id] = name
		}
	}
	for name, w := range m.weights {
		add(w.buf, name)
	}
	act := m.act
	for _, item := range []struct {
		buf  *hip.Buffer
		name string
	}{
		{act.x, "act.x"},
		{act.norm, "act.norm"},
		{act.q, "act.q"},
		{act.k, "act.k"},
		{act.v, "act.v"},
		{act.attn, "act.attn"},
		{act.tmp, "act.tmp"},
		{act.qkv, "act.qkv"},
		{act.z, "act.z"},
		{act.beta, "act.beta"},
		{act.alpha, "act.alpha"},
		{act.conv, "act.conv"},
		{act.gdn, "act.gdn"},
		{act.gdnNorm, "act.gdn_norm"},
		{act.ffnGate, "act.ffn_gate"},
		{act.ffnUp, "act.ffn_up"},
		{act.logits, "act.logits"},
		{act.scores, "act.scores"},
	} {
		add(item.buf, item.name)
	}
	p := m.prefill
	for _, item := range []struct {
		buf  *hip.Buffer
		name string
	}{
		{p.x, "prefill.x"},
		{p.norm, "prefill.norm"},
		{p.q, "prefill.q"},
		{p.k, "prefill.k"},
		{p.v, "prefill.v"},
		{p.attn, "prefill.attn"},
		{p.tmp, "prefill.tmp"},
		{p.qkv, "prefill.qkv"},
		{p.z, "prefill.z"},
		{p.beta, "prefill.beta"},
		{p.alpha, "prefill.alpha"},
		{p.conv, "prefill.conv"},
		{p.gdnNorm, "prefill.gdn_norm"},
		{p.ffnGate, "prefill.ffn_gate"},
		{p.ffnUp, "prefill.ffn_up"},
		{p.xF16, "prefill.x_f16"},
		{p.yF16, "prefill.y_f16"},
		{p.tokens, "prefill.tokens"},
		{p.moeLogits, "prefill.moe_logits"},
		{p.moeIDs, "prefill.moe_ids"},
		{p.moeWeights, "prefill.moe_weights"},
		{p.moeCounts, "prefill.moe_counts"},
		{p.moeAssignRows, "prefill.moe_assign_rows"},
		{p.moeAssignSlots, "prefill.moe_assign_slots"},
		{p.moeAssignWeights, "prefill.moe_assign_weights"},
		{p.moeGate, "prefill.moe_gate"},
		{p.moeUp, "prefill.moe_up"},
		{p.moeOut, "prefill.moe_out"},
		{p.moeQ8, "prefill.moe_q8"},
		{p.moeDS, "prefill.moe_ds"},
		{p.xQ8, "prefill.x_q8"},
		{p.xDS, "prefill.x_ds"},
		{p.attentionBlas, "prefill.attention_blas"},
		{p.attentionSplit, "prefill.attention_split"},
	} {
		add(item.buf, item.name)
	}
	for i, buf := range m.poolK {
		add(buf, fmt.Sprintf("kv.pool_k[%d]", i))
	}
	for i, buf := range m.poolV {
		add(buf, fmt.Sprintf("kv.pool_v[%d]", i))
	}
	add(s.tableBuf, "session.block_table")
	add(s.dyn, "session.dynamic")
	for i, buf := range s.conv {
		add(buf, fmt.Sprintf("session.conv[blk.%d]", i))
	}
	for i, buf := range s.recState {
		add(buf, fmt.Sprintf("session.rec_state[blk.%d]", i))
	}
	for i, buf := range s.traceBufs {
		add(buf, fmt.Sprintf("session.trace[blk.%d]", i))
	}
	return names
}
