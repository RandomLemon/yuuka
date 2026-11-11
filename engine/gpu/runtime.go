//go:build hip && linux

// Package gpu executes the Qwen3.5 decoder on the HIP backend. It builds a
// device plan once at load time and submits one plan per token.
package gpu

import (
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

type weight struct {
	buf   *hip.Buffer
	dtype tensor.DType
}

type layerState struct {
	kvK  *hip.Buffer
	kvV  *hip.Buffer
	conv *hip.Buffer
	rec  *hip.Buffer
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
}

type prefillActivations struct {
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
	gdnNorm *hip.Buffer
	ffnGate *hip.Buffer
	ffnUp   *hip.Buffer
	xF16    *hip.Buffer
	tokens  *hip.Buffer
}

// Runtime is a HIP execution backend implementing engine.Runtime.
type Runtime struct {
	device    *hip.Device
	stream    *hip.Stream
	event     *hip.Event
	plan      *hip.Plan
	tracePlan *hip.Plan
	traceBufs []*hip.Buffer
	config    *qwen35.Config
	weights   map[string]weight
	layers    []layerState
	act       activations
	capacity  int
	pos       int
	logits    []float32
	all       []*hip.Buffer
	closed    bool

	prefill      prefillActivations
	prefillRows  int
	prefillPlans map[int]*hip.Plan
}

// New uploads model weights and builds the device plan.
func New(reader *gguf.Reader, model *qwen35.Model, capacity int) (*Runtime, error) {
	if capacity <= 0 || capacity > model.Config.ContextLength {
		return nil, fmt.Errorf("invalid context capacity %d", capacity)
	}
	if err := validateModelSupport(model); err != nil {
		return nil, err
	}
	device, err := hip.Open(0)
	if err != nil {
		return nil, err
	}
	stream, err := device.NewStream()
	if err != nil {
		device.Close()
		return nil, err
	}
	event, err := device.NewEvent()
	if err != nil {
		stream.Close()
		device.Close()
		return nil, err
	}
	r := &Runtime{
		device:   device,
		stream:   stream,
		event:    event,
		config:   model.Config,
		weights:  make(map[string]weight, len(reader.Tensors())),
		layers:   make([]layerState, model.Config.BlockCount),
		capacity: capacity,
		pos:      0,
		logits:   make([]float32, model.Config.VocabSize),
	}
	if err := r.setup(reader, model); err != nil {
		r.Close()
		return nil, err
	}
	chunk := defaultPrefillChunk
	if chunk > capacity {
		chunk = capacity
	}
	if err := r.configurePrefill(chunk); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

const defaultPrefillChunk = 512

func (r *Runtime) setup(reader *gguf.Reader, model *qwen35.Model) error {
	cfg := r.config
	for _, info := range reader.Tensors() {
		data, err := info.Data()
		if err != nil {
			return err
		}
		buf, err := r.alloc(uint64(len(data)))
		if err != nil {
			return fmt.Errorf("allocate %s: %w", info.Name, err)
		}
		if err := r.device.Upload(buf, data); err != nil {
			return fmt.Errorf("upload %s: %w", info.Name, err)
		}
		r.weights[info.Name] = weight{buf: buf, dtype: info.DType}
	}
	if err := r.allocActivations(); err != nil {
		return err
	}
	if err := r.allocLayers(); err != nil {
		return err
	}
	r.traceBufs = make([]*hip.Buffer, cfg.BlockCount)
	for i := range r.traceBufs {
		buf, err := r.alloc(uint64(cfg.HiddenSize) * 4)
		if err != nil {
			return err
		}
		r.traceBufs[i] = buf
	}
	plan, err := r.buildPlan(model, false)
	if err != nil {
		return err
	}
	r.plan = plan
	tracePlan, err := r.buildPlan(model, true)
	if err != nil {
		return err
	}
	r.tracePlan = tracePlan
	return nil
}

func (r *Runtime) alloc(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	buf, err := r.device.Alloc(bytes)
	if err != nil {
		return nil, err
	}
	r.all = append(r.all, buf)
	return buf, nil
}

func (r *Runtime) allocActivations() error {
	cfg := r.config
	sizes := []struct {
		dst **hip.Buffer
		n   uint64
	}{
		{&r.act.x, uint64(cfg.HiddenSize)},
		{&r.act.norm, uint64(cfg.HiddenSize)},
		{&r.act.q, uint64(cfg.HeadCount * cfg.HeadDim * 2)},
		{&r.act.k, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&r.act.v, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&r.act.attn, uint64(cfg.HeadCount * cfg.HeadDim)},
		{&r.act.tmp, uint64(cfg.HiddenSize)},
		{&r.act.qkv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&r.act.z, uint64(cfg.SSMInnerSize)},
		{&r.act.beta, uint64(cfg.SSMTimeStepRank)},
		{&r.act.alpha, uint64(cfg.SSMTimeStepRank)},
		{&r.act.conv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&r.act.gdn, uint64(cfg.SSMInnerSize)},
		{&r.act.gdnNorm, uint64(cfg.SSMInnerSize)},
		{&r.act.ffnGate, uint64(cfg.FFNSize)},
		{&r.act.ffnUp, uint64(cfg.FFNSize)},
		{&r.act.logits, uint64(cfg.VocabSize)},
		{&r.act.scores, uint64(cfg.HeadCount * r.capacity)},
	}
	for _, entry := range sizes {
		buf, err := r.alloc(entry.n * 4)
		if err != nil {
			return err
		}
		*entry.dst = buf
	}
	return nil
}

const maxPrefillChunk = 4096

// SetPrefillChunk selects the prompt chunk size. It clamps the request to the
// context capacity and to maxPrefillChunk. The call reallocates chunk buffers
// and rebuilds cached plans; call Reset before reusing the runtime.
func (r *Runtime) SetPrefillChunk(rows int) error {
	if r == nil || r.closed {
		return fmt.Errorf("runtime is closed")
	}
	if rows <= 0 {
		return fmt.Errorf("prefill chunk %d must be positive", rows)
	}
	if rows > maxPrefillChunk {
		rows = maxPrefillChunk
	}
	if rows > r.capacity {
		rows = r.capacity
	}
	return r.configurePrefill(rows)
}

// configurePrefill allocates the chunk activations and resets the plan cache.
func (r *Runtime) configurePrefill(rows int) error {
	if rows <= 0 {
		return fmt.Errorf("prefill chunk %d must be positive", rows)
	}
	if rows == r.prefillRows && r.prefill.x != nil {
		return nil
	}
	for _, plan := range r.prefillPlans {
		_ = plan.Close()
	}
	r.prefillPlans = make(map[int]*hip.Plan)
	cfg := r.config
	p := &r.prefill
	maxWidth := cfg.HiddenSize
	for _, width := range []int{cfg.FFNSize, cfg.HeadCount * cfg.HeadDim, cfg.SSMInnerSize} {
		if width > maxWidth {
			maxWidth = width
		}
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
		{&p.ffnGate, rowElems(cfg.FFNSize)},
		{&p.ffnUp, rowElems(cfg.FFNSize)},
		{&p.xF16, uint64(rows) * uint64(maxWidth) * 2},
		{&p.tokens, uint64(rows) * 4},
	}
	for _, entry := range entries {
		buf, err := r.alloc(entry.bytes)
		if err != nil {
			return fmt.Errorf("allocate prefill chunk buffer: %w", err)
		}
		*entry.dst = buf
	}
	r.prefillRows = rows
	return nil
}

// prefillMatmul appends the measured implementation for one prefill matmul.
func (r *Runtime) prefillMatmul(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	switch hip.SelectPrefillPath(w.dtype, rows, nIn, nOut) {
	case hip.PrefillBatchedGemv:
		return plan.AddGemvBatchAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
	case hip.PrefillBLAS:
		if err := plan.AddCastF16At(x, xOff, r.prefill.xF16, 0, rows*nIn); err != nil {
			return err
		}
		return plan.AddGemmPathAt(w.buf, 0, r.prefill.xF16, 0, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmBLAS)
	default:
		return plan.AddGemmPathAt(w.buf, 0, x, xOff, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmWMMA)
	}
}

// prefillPlan returns a cached plan for exactly rows prompt rows.
func (r *Runtime) prefillPlan(rows int) (*hip.Plan, error) {
	if plan, ok := r.prefillPlans[rows]; ok {
		return plan, nil
	}
	plan, err := r.buildPrefillPlan(rows)
	if err != nil {
		return nil, err
	}
	r.prefillPlans[rows] = plan
	return plan, nil
}

// buildPrefillPlan constructs the chunked prompt graph. Every row-parallel
// operator keeps the full chunk; only the output tail selects the last row.
func (r *Runtime) buildPrefillPlan(rows int) (*hip.Plan, error) {
	plan, err := r.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := r.config
	p := &r.prefill
	rows64 := int64(rows)
	embedding, err := r.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbeddingBatchAt(embedding.buf, p.x, p.tokens, 0, rows64,
		int64(cfg.HiddenSize), int64(rowBytes), int64(cfg.VocabSize), embedding.dtype); err != nil {
		return nil, err
	}
	for i := 0; i < cfg.BlockCount; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		attnNorm, err := r.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := r.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			if err := r.addPrefillRecurrent(plan, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		} else {
			if err := r.addPrefillFullAttention(plan, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		}
		if err := r.addPrefillFFN(plan, prefix, postNorm, rows64); err != nil {
			return nil, err
		}
	}
	plan.MarkTail()
	outputNorm, err := r.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	lastOff := uint64(rows-1) * uint64(cfg.HiddenSize) * 4
	if err := plan.AddRMSNormAt(p.x, lastOff, outputNorm.buf, p.norm, 0,
		int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := r.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := r.weight(outputName)
	if err != nil {
		return nil, err
	}
	if err := plan.AddGemv(output.buf, p.norm, r.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

// addPrefillFullAttention appends one full GQA layer over a prompt chunk.
func (r *Runtime) addPrefillFullAttention(plan *hip.Plan, layerIndex int, attnNorm weight, rows int64) error {
	cfg := r.config
	p := &r.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := r.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := r.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := r.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := r.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := r.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := r.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	state := &r.layers[layerIndex]
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, q, p.norm, 0, p.q, 0, hidden, qDim, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, k, p.norm, 0, p.k, 0, hidden, kvDim, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, v, p.norm, 0, p.v, 0, hidden, kvDim, rows); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.q, 0, qNorm.buf, p.q, 0,
		rows, int64(cfg.HeadCount), headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.k, 0, kNorm.buf, p.k, 0,
		rows, int64(cfg.KVHeadCount), headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(p.q, 0, int(rows), cfg.HeadCount, int(2*headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.HeadCount*2*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(p.k, 0, int(rows), cfg.KVHeadCount, int(headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.KVHeadCount*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddKVWriteBatchAt(p.k, 0, p.v, 0, state.kvK, state.kvV,
		kvDim, rows, int64(r.capacity)); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPrefillAt(p.q, 0, state.kvK, state.kvV, p.attn, 0,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(rows),
		r.capacity, scale); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, o, p.attn, 0, p.tmp, 0, oDim, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

// addPrefillRecurrent appends one gated delta-net layer over a prompt chunk.
func (r *Runtime) addPrefillRecurrent(plan *hip.Plan, layerIndex int, attnNorm weight, rows int64) error {
	cfg := r.config
	p := &r.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := r.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := r.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := r.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := r.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := r.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := r.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := r.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := r.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := r.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	state := &r.layers[layerIndex]
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, qkv, p.norm, 0, p.qkv, 0, hidden, convDim, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, gate, p.norm, 0, p.z, 0, hidden, inner, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, beta, p.norm, 0, p.beta, 0, hidden, rank, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, alpha, p.norm, 0, p.alpha, 0, hidden, rank, rows); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if err := plan.AddRecurrentChunkAt(hip.RecurrentChunkArgs{
		ConvState:   state.conv,
		State:       state.rec,
		QKV:         p.qkv,
		Z:           p.z,
		Alpha:       p.alpha,
		Beta:        p.beta,
		ConvWeights: conv.buf,
		DT:          dt.buf,
		A:           a.buf,
		Norm:        norm.buf,
		Conv:        p.conv,
		Out:         p.gdnNorm,
		Rows:        rows,
		Groups:      int64(cfg.SSMGroupCount),
		Heads:       rank,
		StateSize:   stateSize,
		ConvDim:     convDim,
		Inner:       inner,
		Kernel:      int64(cfg.SSMConvKernel),
		Eps:         cfg.RMSNormEps,
		QScale:      qScale,
	}); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, out, p.gdnNorm, 0, p.tmp, 0, inner, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

// addPrefillFFN appends one SwiGLU feed-forward block over a prompt chunk.
func (r *Runtime) addPrefillFFN(plan *hip.Plan, prefix string, postNorm weight, rows int64) error {
	cfg := r.config
	p := &r.prefill
	gate, err := r.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := r.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := r.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, postNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, gate, p.norm, 0, p.ffnGate, 0, hidden, ffn, rows); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, up, p.norm, 0, p.ffnUp, 0, hidden, ffn, rows); err != nil {
		return err
	}
	if err := plan.AddSiluMul(p.ffnGate, p.ffnUp, p.ffnGate, rows*ffn); err != nil {
		return err
	}
	if err := r.prefillMatmul(plan, down, p.ffnGate, 0, p.tmp, 0, ffn, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

func (r *Runtime) allocLayers() error {
	cfg := r.config
	kvDim := uint64(cfg.KVHeadCount * cfg.HeadDim)
	convDim := uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	convState := uint64(cfg.SSMConvKernel-1) * convDim
	recState := uint64(cfg.SSMTimeStepRank * cfg.SSMStateSize * cfg.SSMStateSize)
	for i := range r.layers {
		layer := &r.layers[i]
		if cfg.IsRecurrent(i) {
			var err error
			if layer.conv, err = r.alloc(convState * 4); err != nil {
				return err
			}
			if layer.rec, err = r.alloc(recState * 4); err != nil {
				return err
			}
			if err := r.device.Memset(r.stream, layer.conv, 0, convState*4); err != nil {
				return err
			}
			if err := r.device.Memset(r.stream, layer.rec, 0, recState*4); err != nil {
				return err
			}
		} else {
			var err error
			if layer.kvK, err = r.alloc(uint64(r.capacity) * kvDim * 4); err != nil {
				return err
			}
			if layer.kvV, err = r.alloc(uint64(r.capacity) * kvDim * 4); err != nil {
				return err
			}
			if err := r.device.Memset(r.stream, layer.kvK, 0, uint64(r.capacity)*kvDim*4); err != nil {
				return err
			}
			if err := r.device.Memset(r.stream, layer.kvV, 0, uint64(r.capacity)*kvDim*4); err != nil {
				return err
			}
		}
	}
	return r.stream.Synchronize()
}

func (r *Runtime) weight(name string) (weight, error) {
	w, ok := r.weights[name]
	if !ok {
		return weight{}, fmt.Errorf("missing device weight %q", name)
	}
	return w, nil
}

func (r *Runtime) buildPlan(model *qwen35.Model, trace bool) (*hip.Plan, error) {
	plan, err := r.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := r.config
	embedding, err := r.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbedding(embedding.buf, r.act.x, int64(cfg.HiddenSize), int64(rowBytes), embedding.dtype); err != nil {
		return nil, err
	}
	layerBytes := func(layer int) string { return fmt.Sprintf("blk.%d.", layer) }
	for i := range model.Layers {
		layer := &model.Layers[i]
		prefix := layerBytes(i)
		attnNorm, err := r.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := r.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if layer.Full != nil {
			if err := r.addFullAttention(plan, i, layer, attnNorm); err != nil {
				return nil, err
			}
		} else {
			if err := r.addRecurrent(plan, i, layer, attnNorm); err != nil {
				return nil, err
			}
		}
		if err := r.addFFN(plan, prefix, postNorm); err != nil {
			return nil, err
		}
		if trace {
			if err := plan.AddCopy(r.traceBufs[i], r.act.x, int64(cfg.HiddenSize)); err != nil {
				return nil, err
			}
		}
	}
	plan.MarkTail()
	outputNorm, err := r.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	if err := plan.AddRMSNorm(r.act.x, outputNorm.buf, r.act.norm, int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := r.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := r.weight(outputName)
	if err != nil {
		return nil, err
	}
	if err := plan.AddGemv(output.buf, r.act.norm, r.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (r *Runtime) addFullAttention(plan *hip.Plan, layerIndex int, layer *qwen35.Layer, attnNorm weight) error {
	cfg := r.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := r.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := r.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := r.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := r.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := r.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := r.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	state := &r.layers[layerIndex]
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNorm(r.act.x, attnNorm.buf, r.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(q.buf, r.act.norm, r.act.q, hidden, qDim, q.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(k.buf, r.act.norm, r.act.k, hidden, kvDim, k.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(v.buf, r.act.norm, r.act.v, hidden, kvDim, v.dtype); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(r.act.q, 0, qNorm.buf, r.act.q, 0,
		cfg.HeadCount, headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(r.act.k, 0, kNorm.buf, r.act.k, 0,
		cfg.KVHeadCount, headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPE(r.act.q, cfg.HeadCount, int(2*headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddRoPE(r.act.k, cfg.KVHeadCount, int(headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddKVWrite(r.act.k, r.act.v, state.kvK, state.kvV, kvDim); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttention(r.act.q, state.kvK, state.kvV, r.act.attn, r.act.scores,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), r.capacity, scale); err != nil {
		return err
	}
	if err := plan.AddGemv(o.buf, r.act.attn, r.act.tmp, oDim, hidden, o.dtype); err != nil {
		return err
	}
	return plan.Add(r.act.x, r.act.tmp, hidden)
}

func (r *Runtime) addRecurrent(plan *hip.Plan, layerIndex int, layer *qwen35.Layer, attnNorm weight) error {
	cfg := r.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := r.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := r.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := r.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := r.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := r.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := r.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := r.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := r.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := r.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	state := &r.layers[layerIndex]
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	keyDim := int64(cfg.SSMGroupCount * cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNorm(r.act.x, attnNorm.buf, r.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(qkv.buf, r.act.norm, r.act.qkv, hidden, convDim, qkv.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, r.act.norm, r.act.z, hidden, inner, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(beta.buf, r.act.norm, r.act.beta, hidden, rank, beta.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(alpha.buf, r.act.norm, r.act.alpha, hidden, rank, alpha.dtype); err != nil {
		return err
	}
	if err := plan.AddConv1D(state.conv, conv.buf, r.act.qkv, r.act.conv, int(convDim), cfg.SSMConvKernel); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if err := plan.AddL2NormAt(r.act.conv, 0, r.act.conv, 0,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, qScale); err != nil {
		return err
	}
	if err := plan.AddL2NormAt(r.act.conv, uint64(keyDim)*4, r.act.conv, uint64(keyDim)*4,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, 1); err != nil {
		return err
	}
	if err := plan.AddGDNAt(state.rec,
		r.act.conv, 0,
		r.act.conv, uint64(keyDim)*4,
		r.act.conv, uint64(2*keyDim)*4,
		r.act.alpha, r.act.beta, dt.buf, a.buf, r.act.gdn,
		cfg.SSMGroupCount, cfg.SSMTimeStepRank, int(stateSize)); err != nil {
		return err
	}
	if err := plan.AddGatedRMSNorm(r.act.gdn, norm.buf, r.act.z, r.act.gdnNorm,
		rank, stateSize, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(out.buf, r.act.gdnNorm, r.act.tmp, inner, hidden, out.dtype); err != nil {
		return err
	}
	return plan.Add(r.act.x, r.act.tmp, hidden)
}

func (r *Runtime) addFFN(plan *hip.Plan, prefix string, postNorm weight) error {
	cfg := r.config
	gate, err := r.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := r.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := r.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNorm(r.act.x, postNorm.buf, r.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, r.act.norm, r.act.ffnGate, hidden, ffn, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(up.buf, r.act.norm, r.act.ffnUp, hidden, ffn, up.dtype); err != nil {
		return err
	}
	if err := plan.AddSiluMul(r.act.ffnGate, r.act.ffnUp, r.act.ffnGate, ffn); err != nil {
		return err
	}
	if err := plan.AddGemv(down.buf, r.act.ffnGate, r.act.tmp, ffn, hidden, down.dtype); err != nil {
		return err
	}
	return plan.Add(r.act.x, r.act.tmp, hidden)
}

func (r *Runtime) Reset() {
	if r == nil || r.closed {
		return
	}
	cfg := r.config
	kvDim := uint64(cfg.KVHeadCount * cfg.HeadDim)
	convDim := uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	convState := uint64(cfg.SSMConvKernel-1) * convDim
	recState := uint64(cfg.SSMTimeStepRank * cfg.SSMStateSize * cfg.SSMStateSize)
	for i := range r.layers {
		layer := &r.layers[i]
		if cfg.IsRecurrent(i) {
			_ = r.device.Memset(r.stream, layer.conv, 0, convState*4)
			_ = r.device.Memset(r.stream, layer.rec, 0, recState*4)
		} else {
			_ = r.device.Memset(r.stream, layer.kvK, 0, uint64(r.capacity)*kvDim*4)
			_ = r.device.Memset(r.stream, layer.kvV, 0, uint64(r.capacity)*kvDim*4)
		}
	}
	_ = r.stream.Synchronize()
	r.pos = 0
}

func (r *Runtime) Forward(tokens []int32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if r.pos+len(tokens) > r.capacity {
		return nil, fmt.Errorf("token sequence exceeds remaining context")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= r.config.VocabSize {
			return nil, fmt.Errorf("token id %d is out of range", token)
		}
	}
	if len(tokens) == 1 {
		if err := r.plan.Execute(r.stream, tokens[0], uint32(r.pos), true, r.event); err != nil {
			return nil, err
		}
		if err := r.event.Wait(); err != nil {
			return nil, err
		}
		r.pos++
		return r.downloadLogits()
	}
	return r.forwardPrefill(tokens)
}

// forwardPrefill processes a prompt in measured chunks. Each chunk submits one
// batched plan; only the final chunk runs the output tail.
func (r *Runtime) forwardPrefill(tokens []int32) ([]float32, error) {
	for start := 0; start < len(tokens); {
		n := r.prefillRows
		if remaining := len(tokens) - start; remaining < n {
			n = remaining
		}
		plan, err := r.prefillPlan(n)
		if err != nil {
			return nil, fmt.Errorf("build prefill plan for %d rows: %w", n, err)
		}
		if err := r.device.UploadInt32(r.prefill.tokens, tokens[start:start+n]); err != nil {
			return nil, err
		}
		last := start+n == len(tokens)
		if err := plan.Execute(r.stream, 0, uint32(r.pos), last, r.event); err != nil {
			return nil, err
		}
		if err := r.event.Wait(); err != nil {
			return nil, err
		}
		r.pos += n
		start += n
	}
	return r.downloadLogits()
}

func (r *Runtime) downloadLogits() ([]float32, error) {
	if err := r.device.DownloadFloats(r.logits, r.act.logits); err != nil {
		return nil, err
	}
	return r.logits, nil
}

func (r *Runtime) Step(token int32) ([]float32, error) {
	return r.Forward([]int32{token})
}

// ProfileStep executes one token through the ordinary plan with per-operation
// HIP events and returns the submitted operation timings. The runtime position
// advances exactly as it does for Forward.
func (r *Runtime) ProfileStep(token int32) ([]hip.OpTime, error) {
	if token < 0 || int(token) >= r.config.VocabSize {
		return nil, fmt.Errorf("token id %d is out of range", token)
	}
	if r.pos >= r.capacity {
		return nil, fmt.Errorf("context capacity %d exhausted", r.capacity)
	}
	times, err := r.plan.ExecuteProfiled(r.stream, token, uint32(r.pos), true, r.event)
	if err != nil {
		return nil, err
	}
	if err := r.event.Wait(); err != nil {
		return nil, err
	}
	r.pos++
	return times, nil
}

// ForwardTrace processes tokens and returns the last-token logits together
// with the hidden state after every decoder layer for the last token. It is a
// diagnostic path for CPU/GPU layer comparison and allocates no new device
// buffers after construction.
func (r *Runtime) ForwardTrace(tokens []int32) ([]float32, [][]float32, error) {
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("empty token sequence")
	}
	if r.pos+len(tokens) > r.capacity {
		return nil, nil, fmt.Errorf("token sequence exceeds remaining context")
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= r.config.VocabSize {
			return nil, nil, fmt.Errorf("token id %d is out of range", token)
		}
		if err := r.tracePlan.Execute(r.stream, token, uint32(r.pos), i == len(tokens)-1, r.event); err != nil {
			return nil, nil, err
		}
		if err := r.event.Wait(); err != nil {
			return nil, nil, err
		}
		r.pos++
	}
	if err := r.device.DownloadFloats(r.logits, r.act.logits); err != nil {
		return nil, nil, err
	}
	traces := make([][]float32, len(r.traceBufs))
	for i, buf := range r.traceBufs {
		trace := make([]float32, r.config.HiddenSize)
		if err := r.device.DownloadFloats(trace, buf); err != nil {
			return nil, nil, err
		}
		traces[i] = trace
	}
	return r.logits, traces, nil
}

func (r *Runtime) Capacity() int  { return r.capacity }
func (r *Runtime) VocabSize() int { return r.config.VocabSize }

// Close releases the device plan and every allocation. It is idempotent.
func (r *Runtime) Close() error {
	if r == nil || r.closed {
		return nil
	}
	r.closed = true
	if r.plan != nil {
		_ = r.plan.Close()
	}
	if r.tracePlan != nil {
		_ = r.tracePlan.Close()
	}
	for _, plan := range r.prefillPlans {
		_ = plan.Close()
	}
	r.prefillPlans = nil
	for i := len(r.all) - 1; i >= 0; i-- {
		_ = r.all[i].Close()
	}
	r.all = nil
	if r.event != nil {
		_ = r.event.Close()
	}
	if r.stream != nil {
		_ = r.stream.Close()
	}
	if r.device != nil {
		_ = r.device.Close()
	}
	return nil
}

var _ engine.Runtime = (*Runtime)(nil)
