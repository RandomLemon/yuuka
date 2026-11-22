//go:build hip && linux

package gpu

import (
	"fmt"
	"strings"
	"sync"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

// DefaultKVType is the full-attention KV pool storage used when
// Options.KVTypeSet is false.
const DefaultKVType = tensor.F16

type weight struct {
	buf   *hip.Buffer
	dtype tensor.DType
}

type activations struct {
	x       *hip.Buffer
	norm    *hip.Buffer
	q       *hip.Buffer
	k       *hip.Buffer
	v       *hip.Buffer
	attn    *hip.Buffer
	tmp     *hip.Buffer
	qkv     *hip.Buffer
	z       *hip.Buffer
	beta    *hip.Buffer
	alpha   *hip.Buffer
	conv    *hip.Buffer
	gdn     *hip.Buffer
	gdnNorm *hip.Buffer
	ffnGate *hip.Buffer
	ffnUp   *hip.Buffer
	logits  *hip.Buffer
	scores  *hip.Buffer
	// MoE decode scratch. The routed experts write one slot per selected
	// expert, so every buffer is sized by the used-expert count.
	moeLogits        *hip.Buffer // router logits [experts]
	moeIDs           *hip.Buffer // int32 routing ids [used]
	moeWeights       *hip.Buffer // normalized routing weights [used]
	moeCounts        *hip.Buffer // int32 per-expert assignment counts [experts]
	moeAssignRows    *hip.Buffer // int32 binned assignment rows [experts*used]
	moeAssignSlots   *hip.Buffer // int32 binned assignment slots [experts*used]
	moeAssignWeights *hip.Buffer // binned assignment weights [experts*used]
	moeAct           *hip.Buffer // silu(gate)*up per slot [used*ExpertFFNSize]
	moeOut           *hip.Buffer // expert down outputs per slot [used*hidden]
}

// int8Activations holds the Q8_1 form of one decode activation row. The plan
// quantizes a row once per group of matvecs that read it, and the group's
// operations reuse the same digits, so one pair of buffers serves the whole
// decode graph.
type int8Activations struct {
	qs *hip.Buffer // int8 digits, one per activation value
	ds *hip.Buffer // float2 block scales the quantizer writes
}

type prefillActivations struct {
	attentionBlas  *hip.Buffer
	attentionSplit *hip.Buffer
	x              *hip.Buffer
	norm           *hip.Buffer
	q              *hip.Buffer
	k              *hip.Buffer
	v              *hip.Buffer
	attn           *hip.Buffer
	tmp            *hip.Buffer
	qkv            *hip.Buffer
	z              *hip.Buffer
	beta           *hip.Buffer
	alpha          *hip.Buffer
	conv           *hip.Buffer
	gdnNorm        *hip.Buffer
	ffnGate        *hip.Buffer
	ffnUp          *hip.Buffer
	xF16           *hip.Buffer
	yF16           *hip.Buffer
	xQ8            *hip.Buffer
	xDS            *hip.Buffer
	tokens         *hip.Buffer
	// MoE prefill scratch. The grouped kernels bin every token's selected
	// experts, so these are sized by rows*used. moeCounts and the three
	// assignment buffers are the routing bins that AddMoERoute fills.
	moeLogits        *hip.Buffer // router logits [rows,experts]
	moeIDs           *hip.Buffer // int32 routing ids [rows,used]
	moeWeights       *hip.Buffer // normalized routing weights [rows,used]
	moeCounts        *hip.Buffer // int32 per-expert counts [experts]
	moeAssignRows    *hip.Buffer // int32 binned rows [experts,rows*used]
	moeAssignSlots   *hip.Buffer // int32 binned slots [experts,rows*used]
	moeAssignWeights *hip.Buffer // binned weights [experts,rows*used]
	moeGate          *hip.Buffer // expert gate projections [rows*used,ffn]
	moeUp            *hip.Buffer // expert up projections [rows*used,ffn]
	moeOut           *hip.Buffer // weighted expert output [rows*used,hidden]
	moeQ8            *hip.Buffer // int8 silu(gate)*up per slot [rows*used,ffn]
	moeDS            *hip.Buffer // float2 Q8_1 scales for moeQ8
}

// Options configures a shared GPU model and its KV pool.
type Options struct {
	Blocks       int
	BlockSize    int
	PrefillChunk int
	// EnableGraph lets the session capture the decode plan into a HIP Graph
	// after GraphMinSteps steps and replay it; serving.md owns the contract and
	// the CLI and server leave it off. The replay path is measured on the
	// reduced decode plan.
	EnableGraph   bool
	GraphMinSteps int
	// KVType selects the full-attention KV pool storage when KVTypeSet is true.
	// An unset selector uses DefaultKVType; tensor.F32 is the zero value, so it
	// must set KVTypeSet to request the reference format explicitly.
	KVType tensor.DType
	// KVTypeSet distinguishes an explicit KVType from an unset selector.
	KVTypeSet bool
	// LegacyBlasAttention selects the rocBLAS prefill attention path for F16 KV
	// pools. It is off by default: the FP16 MMA kernel covers the shapes this
	// path can stage and is measured faster at every chunk length it accepts
	// (kernels.md section 7.5). The path stays available for A/B measurement.
	LegacyBlasAttention bool
	// ForceTiledAttention keeps the tiled paged prefill attention kernel where
	// the FP16 MMA kernel would be selected. It exists so a differential test
	// can compare the two kernels on identical plan inputs.
	ForceTiledAttention bool
	// ForceDecodeGemv keeps the per-row GEMV for every decode matvec, including
	// the output projection, where SelectDecodeGemvPath selects the streaming
	// kernel. It exists so a differential test can compare the two families on
	// identical plan inputs; production sessions leave it false.
	ForceDecodeGemv bool
	// Int8Matvec selects the int8 decode matvec instead of the FP32 streaming
	// kernel for the K-quants and Q8_0. It is off by default because the route
	// is measured slower end to end on gfx1151, and it exists so the two
	// arithmetics can be compared on identical plans (no default path changes).
	Int8Matvec bool
	// ForceMoEQ8 keeps the grouped Q8_1 dp4a route for the routed-expert prefill
	// where SelectMoEPrefillPath would take the int8 WMMA route. It exists so the
	// two routes can be compared on identical plan inputs; production sessions
	// leave it false, and the int8 WMMA route is the measured default for the
	// shapes it serves (docs/kernels.md, docs/benchmarks/qwen3.6_35b_a3b.md).
	ForceMoEQ8 bool
	// PrequantizedInt8 submits one Q8_1 quantization operation per activation
	// group instead of letting each int8 matvec block quantize the row itself.
	// It exists so the two activation sources can be measured against each other;
	// the fused form is the selected one (docs/benchmarks.md).
	PrequantizedInt8 bool
	// NoResidualFusion keeps the decode matvec that ends a sub-layer and the
	// separate residual ADD as two operations instead of letting the matvec add
	// the residual in its epilogue, which is the default.
	NoResidualFusion bool
	// NoPairFusion keeps two projections that share an activation row as two
	// launches instead of the fused pair that forms both dots over one traversal,
	// which is the default. With the SwiGLU epilogue it also restores the
	// separate SILU_MUL.
	NoPairFusion bool
	// LegacyDeltaNet builds the recurrent layer from the separate CONV1D,
	// L2NORM, GDN, and GATED_RMSNORM operations instead of the fused
	// CONV_L2NORM and RECURRENT_TOKEN pair, which is the selected path. It
	// exists so a differential test can compare the two plans on identical
	// inputs and so the unfused arithmetic stays reachable when a fused kernel
	// is suspect; production sessions leave it false.
	LegacyDeltaNet bool
}

const (
	maxPrefillChunk = 4096
	// maxAttentionBlasScratch bounds the reusable per-model scratch allocated
	// for matrixized prefill attention. Pools whose requirement exceeds the
	// budget keep the tiled kernel instead of failing model open.
	maxAttentionBlasScratch = 4096 << 20
)

// Model owns device weights, shared decode/prefill activations, the KV block
// pools, and the block allocator. Sessions bind request block tables to the
// pool. Model execution is serialized by stepMu, so sessions must be stepped
// one at a time; the serving scheduler guarantees that ordering.
type Model struct {
	device *hip.Device
	stream *hip.Stream
	event  *hip.Event
	config *qwen35.Config
	alloc  *memory.Allocator

	weights map[string]weight
	act     activations
	act8    int8Activations
	prefill prefillActivations

	poolK     []*hip.Buffer
	poolV     []*hip.Buffer
	blockSize int
	blocks    int
	capacity  int
	kvType    tensor.DType
	kvRow     uint64

	attentionBlasCapable bool
	attentionBlasEnabled bool
	forceTiledAttention  bool
	forceDecodeGemv      bool
	int8Matvec           bool
	forceMoEQ8           bool
	prequantizedInt8     bool
	legacyDeltaNet       bool
	noResidualFusion     bool
	noPairFusion         bool
	// int8Act and int8ActWidth record the activation row whose Q8_1 digits the
	// int8 matvecs of the current group read. They are a build-time check: the
	// plan is sequential, so a matvec on any other row means the planner forgot
	// to quantize it.
	int8Act      *hip.Buffer
	int8ActWidth int64
	prefillRows  int
	prefillGen   int64
	prefillAll   []*hip.Buffer

	specLogits *hip.Buffer

	stepMu sync.Mutex
	all    []*hip.Buffer
	closed bool
}

// Open uploads weights and allocates the device pool shared by all sessions.
func Open(reader *gguf.Reader, model *qwen35.Model, opts Options) (*Model, error) {
	if model == nil || reader == nil {
		return nil, fmt.Errorf("gpu: model and reader are required")
	}
	if opts.Blocks <= 0 || opts.BlockSize <= 0 {
		return nil, fmt.Errorf("gpu: KV pool needs positive blocks and block size")
	}
	if err := validateModelSupport(model); err != nil {
		return nil, err
	}
	kvType := opts.KVType
	if !opts.KVTypeSet {
		kvType = DefaultKVType
	}
	kvRow, err := hip.KVRowBytes(kvType, int64(model.Config.KVHeadCount*model.Config.HeadDim))
	if err != nil {
		return nil, err
	}
	alloc, err := memory.NewAllocator(opts.Blocks, opts.BlockSize)
	if err != nil {
		return nil, err
	}
	device, err := hip.Open(0)
	if err != nil {
		return nil, err
	}
	stream, err := device.NewStream()
	if err != nil {
		_ = device.Close()
		return nil, err
	}
	event, err := device.NewEvent()
	if err != nil {
		_ = stream.Close()
		_ = device.Close()
		return nil, err
	}
	useBlasAttention := kvType == tensor.F16 && opts.LegacyBlasAttention
	m := &Model{
		attentionBlasCapable: useBlasAttention,
		attentionBlasEnabled: useBlasAttention,
		forceTiledAttention:  opts.ForceTiledAttention,
		forceDecodeGemv:      opts.ForceDecodeGemv,
		int8Matvec:           opts.Int8Matvec,
		forceMoEQ8:           opts.ForceMoEQ8,
		prequantizedInt8:     opts.PrequantizedInt8,
		legacyDeltaNet:       opts.LegacyDeltaNet,
		noResidualFusion:     opts.NoResidualFusion,
		noPairFusion:         opts.NoPairFusion,
		device:               device,
		stream:               stream,
		event:                event,
		config:               model.Config,
		alloc:                alloc,
		weights:              make(map[string]weight, len(reader.Tensors())),
		blockSize:            opts.BlockSize,
		blocks:               opts.Blocks,
		capacity:             opts.Blocks * opts.BlockSize,
		kvType:               kvType,
		kvRow:                kvRow,
		poolK:                make([]*hip.Buffer, model.Config.BlockCount),
		poolV:                make([]*hip.Buffer, model.Config.BlockCount),
	}
	if err := m.setup(reader, model); err != nil {
		_ = m.Close()
		return nil, err
	}
	chunk := opts.PrefillChunk
	if chunk <= 0 {
		chunk = defaultPrefillChunk
	}
	if chunk > maxPrefillChunk {
		chunk = maxPrefillChunk
	}
	if chunk > m.capacity {
		chunk = m.capacity
	}
	if err := m.configurePrefill(chunk); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

const (
	defaultPrefillChunk = 512
	maxSpecDraft        = 16
)

func (m *Model) setup(reader *gguf.Reader, model *qwen35.Model) error {
	for _, info := range reader.Tensors() {
		data, err := info.Data()
		if err != nil {
			return err
		}
		buf, err := m.allocDevice(uint64(len(data)))
		if err != nil {
			return fmt.Errorf("allocate %s: %w", info.Name, err)
		}
		if err := m.device.Upload(buf, data); err != nil {
			return fmt.Errorf("upload %s: %w", info.Name, err)
		}
		m.weights[info.Name] = weight{buf: buf, dtype: info.DType}
	}
	if err := m.allocActivations(); err != nil {
		return err
	}
	if err := m.allocPools(); err != nil {
		return err
	}
	spec, err := m.allocDevice(uint64(maxSpecDraft) * uint64(m.config.VocabSize) * 4)
	if err != nil {
		return fmt.Errorf("allocate speculative logits: %w", err)
	}
	m.specLogits = spec
	return m.stream.Synchronize()
}

func (m *Model) allocDevice(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	buf, err := m.device.Alloc(bytes)
	if err != nil {
		return nil, err
	}
	m.all = append(m.all, buf)
	return buf, nil
}

// allocTransient allocates a session-owned buffer that is not tracked in the
// model lifetime list. The session is responsible for closing it.
func (m *Model) allocTransient(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	return m.device.Alloc(bytes)
}

// Allocations returns the number of model-lifetime device allocations. Session
// buffers are not included; this is used by tests to detect growth.
func (m *Model) Allocations() int { return len(m.all) }

// decodeAttentionWideContext and decodeAttentionTile select the number of key
// positions one split-key decode attention block owns, and must match
// flash_tile in kernels/hip/attention/attention.hip: 32 positions below the
// threshold, 64 above it.
const (
	decodeAttentionWideContext = 1024
	decodeAttentionTile        = 32
	decodeAttentionWideTile    = 64
)

// decodeAttentionScratch returns the float count of the decode attention
// workspace. The tiled kernel writes one partial output vector per head and tile
// plus a running max and normalizer per head and tile, laid out as
// heads * ntiles * (headDim + 2) floats with ntiles = ceil(capacity / tile).
// The serial fallback needs heads * scoreStride floats, so the workspace is the
// larger of the two. scoreStride is the per-head score stride the plan passes
// (the model capacity), which bounds capacity.
func decodeAttentionScratch(heads, headDim, scoreStride, capacity int) uint64 {
	tile := decodeAttentionTile
	if capacity > decodeAttentionWideContext {
		tile = decodeAttentionWideTile
	}
	tiles := (capacity + tile - 1) / tile
	tiled := heads * tiles * (headDim + 2)
	serial := heads * scoreStride
	if tiled > serial {
		return uint64(tiled)
	}
	return uint64(serial)
}

// ffnWidth returns the widest feed-forward activation width a layer writes.
// A dense layer writes its single FFN width; an MoE layer writes the shared
// expert and the routed expert widths, so the shared activation buffers must
// hold the wider of the two.
func (m *Model) ffnWidth() int {
	cfg := m.config
	if cfg.IsMoE() {
		return max(cfg.ExpertFFNSize, cfg.SharedFFNSize)
	}
	return cfg.FFNSize
}

func (m *Model) allocActivations() error {
	cfg := m.config
	var err error
	ffnWidth := m.ffnWidth()
	type activationSize struct {
		dst **hip.Buffer
		n   uint64
	}
	sizes := []activationSize{
		{&m.act.x, uint64(cfg.HiddenSize)},
		{&m.act.norm, uint64(cfg.HiddenSize)},
		{&m.act.q, uint64(cfg.HeadCount * cfg.HeadDim * 2)},
		{&m.act.k, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&m.act.v, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&m.act.attn, uint64(cfg.HeadCount * cfg.HeadDim)},
		{&m.act.tmp, uint64(cfg.HiddenSize)},
		{&m.act.qkv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&m.act.z, uint64(cfg.SSMInnerSize)},
		{&m.act.beta, uint64(cfg.SSMTimeStepRank)},
		{&m.act.alpha, uint64(cfg.SSMTimeStepRank)},
		{&m.act.conv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&m.act.gdn, uint64(cfg.SSMInnerSize)},
		{&m.act.gdnNorm, uint64(cfg.SSMInnerSize)},
		{&m.act.ffnGate, uint64(ffnWidth)},
		{&m.act.ffnUp, uint64(ffnWidth)},
		{&m.act.logits, uint64(cfg.VocabSize)},
		{&m.act.scores, decodeAttentionScratch(cfg.HeadCount, int(cfg.HeadDim), m.capacity, m.capacity)},
	}
	if cfg.IsMoE() {
		experts := uint64(cfg.ExpertCount)
		used := uint64(cfg.ExpertUsedCount)
		sizes = append(sizes,
			activationSize{&m.act.moeLogits, experts},
			activationSize{&m.act.moeIDs, used},
			activationSize{&m.act.moeWeights, used},
			activationSize{&m.act.moeCounts, experts},
			activationSize{&m.act.moeAssignRows, experts * used},
			activationSize{&m.act.moeAssignSlots, experts * used},
			activationSize{&m.act.moeAssignWeights, experts * used},
			activationSize{&m.act.moeAct, used * uint64(cfg.ExpertFFNSize)},
			activationSize{&m.act.moeOut, used * uint64(cfg.HiddenSize)},
		)
	}
	for _, entry := range sizes {
		buf, err := m.allocDevice(entry.n * 4)
		if err != nil {
			return err
		}
		*entry.dst = buf
	}
	// One Q8_1 row wide enough for the widest decode matvec input: hidden for
	// the attention and feed-forward projections, and each layer's own width for
	// its output projection and value path.
	widest := uint64(cfg.HiddenSize)
	for _, width := range []uint64{
		uint64(cfg.HeadCount * cfg.HeadDim),
		uint64(cfg.SSMInnerSize),
		uint64(ffnWidth),
	} {
		if width > widest {
			widest = width
		}
	}
	if m.act8.qs, err = m.allocDevice(widest); err != nil {
		return err
	}
	if m.act8.ds, err = m.allocDevice((widest / 32) * 8); err != nil {
		return err
	}
	return nil
}

// allocPools allocates one K and V block pool per full-attention layer. The
// pools use allocator block IDs directly.
func (m *Model) allocPools() error {
	cfg := m.config
	bytes := uint64(m.blocks) * uint64(m.blockSize) * m.kvRow
	for i := range cfg.BlockCount {
		if cfg.IsRecurrent(i) {
			continue
		}
		k, err := m.allocDevice(bytes)
		if err != nil {
			return err
		}
		v, err := m.allocDevice(bytes)
		if err != nil {
			return err
		}
		if err := m.device.Memset(m.stream, k, 0, bytes); err != nil {
			return err
		}
		if err := m.device.Memset(m.stream, v, 0, bytes); err != nil {
			return err
		}
		m.poolK[i] = k
		m.poolV[i] = v
	}
	return nil
}

// SetPrefillChunk selects the prompt chunk size. Existing sessions detect the
// generation change and rebuild their cached chunk plans.
func (m *Model) SetPrefillChunk(rows int) error {
	if m == nil || m.closed {
		return fmt.Errorf("gpu: model is closed")
	}
	if rows <= 0 {
		return fmt.Errorf("gpu: prefill chunk %d must be positive", rows)
	}
	if rows > maxPrefillChunk {
		rows = maxPrefillChunk
	}
	if rows > m.capacity {
		rows = m.capacity
	}
	m.stepMu.Lock()
	defer m.stepMu.Unlock()
	return m.configurePrefill(rows)
}

func (m *Model) configurePrefill(rows int) error {
	if rows <= 0 {
		return fmt.Errorf("gpu: prefill chunk %d must be positive", rows)
	}
	if rows == m.prefillRows && m.prefill.x != nil {
		return nil
	}
	for i := len(m.prefillAll) - 1; i >= 0; i-- {
		_ = m.prefillAll[i].Close()
	}
	m.prefillAll = nil
	cfg := m.config
	p := &m.prefill
	ffnWidth := m.ffnWidth()
	maxWidth := cfg.HiddenSize
	for _, width := range []int{ffnWidth, cfg.HeadCount * cfg.HeadDim, cfg.SSMInnerSize} {
		if width > maxWidth {
			maxWidth = width
		}
	}
	// The rocBLAS routes write n_out*batch FP16 results per matmul, and their
	// widest n_out is either a projection width or the gated-delta-net QKV
	// width. The int8 route stages one Q8_1 activation tile pair for the widest
	// K-quant projection; token_embd and output are excluded because the first
	// is only gathered by the embedding op and the second only feeds the
	// single-row LM head, and their Q8_1 form would reserve memory no prefill
	// matmul reads.
	maxOutWidth := maxWidth
	if convDim := 2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize; convDim > maxOutWidth {
		maxOutWidth = convDim
	}
	// An MoE router writes one logit row per expert; the rocBLAS staging buffer
	// must hold that output when the router selects the FP16 route.
	if cfg.IsMoE() && cfg.ExpertCount > maxOutWidth {
		maxOutWidth = cfg.ExpertCount
	}
	rowElems := func(width int) uint64 { return uint64(rows) * uint64(width) * 4 }
	entries := []struct {
		dst   **hip.Buffer
		bytes uint64
	}{
		{&p.x, rowElems(cfg.HiddenSize)},
		{&p.norm, rowElems(cfg.HiddenSize)},
		{&p.q, rowElems(cfg.HeadCount * cfg.HeadDim * 2)},
		{&p.k, rowElems(cfg.KVHeadCount * cfg.HeadDim)},
		{&p.v, rowElems(cfg.KVHeadCount * cfg.HeadDim)},
		{&p.attn, rowElems(cfg.HeadCount * cfg.HeadDim)},
		{&p.tmp, rowElems(cfg.HiddenSize)},
		{&p.qkv, rowElems(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&p.z, rowElems(cfg.SSMInnerSize)},
		{&p.beta, rowElems(cfg.SSMTimeStepRank)},
		{&p.alpha, rowElems(cfg.SSMTimeStepRank)},
		{&p.conv, rowElems(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&p.gdnNorm, rowElems(cfg.SSMInnerSize)},
		{&p.ffnGate, rowElems(ffnWidth)},
		{&p.ffnUp, rowElems(ffnWidth)},
		{&p.xF16, uint64(rows) * uint64(maxWidth) * 2},
		{&p.yF16, uint64(rows) * uint64(maxOutWidth) * 2},
		{&p.tokens, uint64(rows) * 4},
	}
	if cfg.IsMoE() {
		experts := uint64(cfg.ExpertCount)
		used := uint64(cfg.ExpertUsedCount)
		rowSlots := uint64(rows) * used
		expertWidth := uint64(cfg.ExpertFFNSize)
		hiddenWidth := uint64(cfg.HiddenSize)
		entries = append(entries,
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeLogits, uint64(rows) * experts * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeIDs, rowSlots * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeWeights, rowSlots * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeCounts, experts * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeAssignRows, experts * rowSlots * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeAssignSlots, experts * rowSlots * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeAssignWeights, experts * rowSlots * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeGate, rowSlots * expertWidth * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeUp, rowSlots * expertWidth * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeOut, rowSlots * hiddenWidth * 4},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeQ8, rowSlots * expertWidth},
			struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.moeDS, rowSlots * (expertWidth / 32) * 8},
		)
	}
	p.attentionBlas = nil
	p.attentionSplit = nil
	arch, _, _ := strings.Cut(m.device.Info().GCNArch, ":")
	measuredRows := rows == 512 || rows == 1024 || rows == 2048
	if arch == "gfx1151" && measuredRows && m.capacity >= 6144+512 &&
		m.kvType == tensor.F16 && cfg.HeadDim == 256 && cfg.HeadCount == 4*cfg.KVHeadCount &&
		!m.forceTiledAttention {
		values := 4*(cfg.HeadDim+2) + 1
		bytes := uint64(rows) * uint64(cfg.HeadCount) * uint64(values) * 4
		entries = append(entries, struct {
			dst   **hip.Buffer
			bytes uint64
		}{&p.attentionSplit, bytes})
	}
	m.attentionBlasEnabled = m.attentionBlasCapable
	if m.attentionBlasEnabled {
		// Per-chunk reusable FP16 Q/K/V, attention-score, and PV matrices.
		// The bound keeps the default path from reserving an unbounded amount
		// of device memory for large KV pools.
		qElems := uint64(cfg.HeadCount) * uint64(rows) * uint64(cfg.HeadDim)
		kvElems := uint64(cfg.KVHeadCount) * uint64(m.capacity) * uint64(cfg.HeadDim)
		scoreElems := uint64(cfg.HeadCount) * uint64(rows) * uint64(m.capacity)
		bytes := 6*qElems + 4*kvElems + 6*scoreElems
		if bytes > maxAttentionBlasScratch {
			m.attentionBlasEnabled = false
		} else {
			entries = append(entries, struct {
				dst   **hip.Buffer
				bytes uint64
			}{&p.attentionBlas, bytes})
		}
	}
	// Q8_1 activations for the int8 prefill GEMM: one int8 value per element
	// and one float2 block scale pair per 32 values.
	entries = append(entries,
		struct {
			dst   **hip.Buffer
			bytes uint64
		}{&p.xQ8, uint64(rows) * uint64(maxOutWidth)},
		struct {
			dst   **hip.Buffer
			bytes uint64
		}{&p.xDS, uint64(rows) * ((uint64(maxOutWidth) + 31) / 32) * 8})
	for _, entry := range entries {
		buf, err := m.allocTransient(entry.bytes)
		if err != nil {
			return fmt.Errorf("allocate prefill chunk buffer: %w", err)
		}
		*entry.dst = buf
		m.prefillAll = append(m.prefillAll, buf)
	}
	m.prefillRows = rows
	m.prefillGen++
	return nil
}

func (m *Model) weight(name string) (weight, error) {
	w, ok := m.weights[name]
	if !ok {
		return weight{}, fmt.Errorf("missing device weight %q", name)
	}
	return w, nil
}

// CopyBlock copies one complete KV block for every full-attention layer. It is
// synchronous with the model stream and is used for private prefix blocks.
func (m *Model) CopyBlock(dst, src int) error {
	if dst < 0 || dst >= m.blocks || src < 0 || src >= m.blocks {
		return fmt.Errorf("gpu: KV block copy %d <- %d is out of range", dst, src)
	}
	m.stepMu.Lock()
	defer m.stepMu.Unlock()
	rowBytes := uint64(m.blockSize) * m.kvRow
	dstOff := uint64(dst) * rowBytes
	srcOff := uint64(src) * rowBytes
	copies := 0
	for i := range m.poolK {
		if m.poolK[i] == nil {
			continue
		}
		if err := m.device.CopyBuffer(m.stream, m.poolK[i], dstOff, m.poolK[i], srcOff, rowBytes, nil); err != nil {
			return err
		}
		if err := m.device.CopyBuffer(m.stream, m.poolV[i], dstOff, m.poolV[i], srcOff, rowBytes, nil); err != nil {
			return err
		}
		copies++
	}
	if copies == 0 {
		return nil
	}
	return m.stream.Synchronize()
}

// Allocator returns the shared KV block allocator.
func (m *Model) Allocator() *memory.Allocator { return m.alloc }

// BlockSize returns the KV block size.
func (m *Model) BlockSize() int { return m.blockSize }

// VocabSize returns the model vocabulary size.
func (m *Model) VocabSize() int { return m.config.VocabSize }

// KVType returns the full-attention KV pool storage type.
func (m *Model) KVType() tensor.DType { return m.kvType }

// Capacity returns the maximum number of positions in the pool.
func (m *Model) Capacity() int { return m.capacity }

// PrefillRows returns the current prompt chunk size.
func (m *Model) PrefillRows() int { return m.prefillRows }

// Close releases the stream, device, and every allocation. Sessions must be
// closed first.
func (m *Model) Close() error {
	if m == nil || m.closed {
		return nil
	}
	m.closed = true
	for i := len(m.prefillAll) - 1; i >= 0; i-- {
		_ = m.prefillAll[i].Close()
	}
	m.prefillAll = nil
	for i := len(m.all) - 1; i >= 0; i-- {
		_ = m.all[i].Close()
	}
	m.all = nil
	if m.event != nil {
		_ = m.event.Close()
	}
	if m.stream != nil {
		_ = m.stream.Close()
	}
	if m.device != nil {
		_ = m.device.Close()
	}
	return nil
}
