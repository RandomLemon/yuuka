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

// Runtime is a HIP execution backend implementing engine.Runtime.
type Runtime struct {
	device   *hip.Device
	stream   *hip.Stream
	event    *hip.Event
	plan     *hip.Plan
	config   *qwen35.Config
	weights  map[string]weight
	layers   []layerState
	act      activations
	capacity int
	pos      int
	logits   []float32
	all      []*hip.Buffer
	closed   bool
}

// New uploads model weights and builds the device plan.
func New(reader *gguf.Reader, model *qwen35.Model, capacity int) (*Runtime, error) {
	if capacity <= 0 || capacity > model.Config.ContextLength {
		return nil, fmt.Errorf("invalid context capacity %d", capacity)
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
	return r, nil
}

func (r *Runtime) setup(reader *gguf.Reader, model *qwen35.Model) error {
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
	return r.buildPlan(model)
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

func (r *Runtime) buildPlan(model *qwen35.Model) error {
	plan, err := r.device.NewPlan()
	if err != nil {
		return err
	}
	r.plan = plan
	cfg := r.config
	embedding, err := r.weight("token_embd.weight")
	if err != nil {
		return err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return err
	}
	if err := plan.AddEmbedding(embedding.buf, r.act.x, int64(cfg.HiddenSize), int64(rowBytes), embedding.dtype); err != nil {
		return err
	}
	layerBytes := func(layer int) string { return fmt.Sprintf("blk.%d.", layer) }
	for i := range model.Layers {
		layer := &model.Layers[i]
		prefix := layerBytes(i)
		attnNorm, err := r.weight(prefix + "attn_norm.weight")
		if err != nil {
			return err
		}
		postNorm, err := r.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return err
		}
		if layer.Full != nil {
			if err := r.addFullAttention(plan, i, layer, attnNorm); err != nil {
				return err
			}
		} else {
			if err := r.addRecurrent(plan, i, layer, attnNorm); err != nil {
				return err
			}
		}
		if err := r.addFFN(plan, prefix, postNorm); err != nil {
			return err
		}
	}
	plan.MarkTail()
	outputNorm, err := r.weight("output_norm.weight")
	if err != nil {
		return err
	}
	if err := plan.AddRMSNorm(r.act.x, outputNorm.buf, r.act.norm, int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return err
	}
	outputName := "output.weight"
	if _, ok := r.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := r.weight(outputName)
	if err != nil {
		return err
	}
	return plan.AddGemv(output.buf, r.act.norm, r.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype)
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
	for h := 0; h < cfg.HeadCount; h++ {
		off := uint64(h) * uint64(headDim) * 2 * 4
		if err := plan.AddRMSNormAt(r.act.q, off, qNorm.buf, r.act.q, off, headDim, cfg.RMSNormEps); err != nil {
			return err
		}
	}
	for h := 0; h < cfg.KVHeadCount; h++ {
		off := uint64(h) * uint64(headDim) * 4
		if err := plan.AddRMSNormAt(r.act.k, off, kNorm.buf, r.act.k, off, headDim, cfg.RMSNormEps); err != nil {
			return err
		}
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
	for i, token := range tokens {
		if token < 0 || int(token) >= r.config.VocabSize {
			return nil, fmt.Errorf("token id %d is out of range", token)
		}
		if err := r.plan.Execute(r.stream, token, uint32(r.pos), i == len(tokens)-1, r.event); err != nil {
			return nil, err
		}
		if err := r.event.Wait(); err != nil {
			return nil, err
		}
		r.pos++
	}
	if err := r.device.DownloadFloats(r.logits, r.act.logits); err != nil {
		return nil, err
	}
	return r.logits, nil
}

func (r *Runtime) Step(token int32) ([]float32, error) {
	return r.Forward([]int32{token})
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
