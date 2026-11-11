package dense

import (
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/tensor"
)

// Forward processes tokens and returns the logits for the last token. The
// returned slice is owned by st and is valid until the next forward call.
func (m *Model) Forward(tokens []int32, st *State) ([]float32, error) {
	return m.ForwardTrace(tokens, st, nil)
}

// ForwardTrace is Forward with an optional per-layer hidden-state callback.
func (m *Model) ForwardTrace(tokens []int32, st *State, trace func(layer int, x []float32)) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if st.Pos+len(tokens) > st.Capacity {
		return nil, fmt.Errorf("token sequence exceeds remaining context")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= m.Config.VocabSize {
			return nil, fmt.Errorf("token id %d is out of range", token)
		}
		if err := m.forwardOne(token, st, token == tokens[len(tokens)-1], trace); err != nil {
			return nil, err
		}
	}
	return st.scratch.logits, nil
}

// Step processes one token and returns its logits.
func (m *Model) Step(token int32, st *State) ([]float32, error) {
	return m.Forward([]int32{token}, st)
}

// LastHidden returns the normalized hidden state from the last forward call.
func (m *Model) LastHidden(st *State) []float32 { return st.scratch.norm }

func (m *Model) forwardOne(token int32, st *State, wantLogits bool, trace func(int, []float32)) error {
	if st.Pos >= st.Capacity {
		return fmt.Errorf("context capacity %d exhausted", st.Capacity)
	}
	rowBytes := int64(m.embeddingRowB)
	start := int64(token) * rowBytes
	end := start + rowBytes
	data := m.Embedding.Data
	if start < 0 || end > int64(len(data)) {
		return fmt.Errorf("embedding row for token %d is out of range", token)
	}
	if err := tensor.DequantizeRow(st.scratch.x, m.Embedding.DType, data[start:end]); err != nil {
		return fmt.Errorf("embedding token %d: %w", token, err)
	}

	for i := range m.Layers {
		layer := &m.Layers[i]
		if err := m.forwardLayer(i, layer, st); err != nil {
			return err
		}
		if trace != nil {
			trace(i, st.scratch.x)
		}
	}
	rmsNorm(st.scratch.norm, st.scratch.x, m.OutputNorm, m.Config.RMSNormEps)
	if wantLogits {
		if err := m.Output.MatVec(st.scratch.logits, st.scratch.norm); err != nil {
			return fmt.Errorf("lm head: %w", err)
		}
	}
	st.Pos++
	return nil
}

func (m *Model) forwardLayer(index int, layer *Layer, st *State) error {
	cfg := m.Config
	s := &st.scratch
	rmsNorm(s.norm, s.x, layer.AttnNorm, cfg.RMSNormEps)
	if err := layer.Q.MatVec(s.q, s.norm); err != nil {
		return fmt.Errorf("attn_q: %w", err)
	}
	if err := layer.K.MatVec(s.k, s.norm); err != nil {
		return fmt.Errorf("attn_k: %w", err)
	}
	if err := layer.V.MatVec(s.v, s.norm); err != nil {
		return fmt.Errorf("attn_v: %w", err)
	}
	if cfg.HasQKNorm {
		for h := 0; h < cfg.HeadCount; h++ {
			q := s.q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
			rmsNormInPlace(q, layer.QNorm, cfg.RMSNormEps)
		}
		for h := 0; h < cfg.KVHeadCount; h++ {
			k := s.k[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
			rmsNormInPlace(k, layer.KNorm, cfg.RMSNormEps)
		}
	}
	applyRoPE(s.q, cfg.HeadCount, cfg.HeadDim, cfg.HeadDim, cfg.RoPEDim, st.Pos, cfg.RoPEFreqBase)
	applyRoPE(s.k, cfg.KVHeadCount, cfg.HeadDim, cfg.HeadDim, cfg.RoPEDim, st.Pos, cfg.RoPEFreqBase)

	kvDim := cfg.KVHeadCount * cfg.HeadDim
	cache := st.kv[index]
	copy(cache.k[st.Pos*kvDim:(st.Pos+1)*kvDim], s.k)
	copy(cache.v[st.Pos*kvDim:(st.Pos+1)*kvDim], s.v)

	group := cfg.KVHeadsPerGroup()
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	length := st.Pos + 1
	for h := 0; h < cfg.HeadCount; h++ {
		kvh := h / group
		q := s.q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		scores := s.scores[:length]
		for p := 0; p < length; p++ {
			base := p*kvDim + kvh*cfg.HeadDim
			scores[p] = dot(q, cache.k[base:base+cfg.HeadDim]) * scale
		}
		softmaxInPlace(scores)
		out := s.attn[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		clear(out)
		for p := 0; p < length; p++ {
			weight := scores[p]
			base := p*kvDim + kvh*cfg.HeadDim
			v := cache.v[base : base+cfg.HeadDim]
			for d := range out {
				out[d] += weight * v[d]
			}
		}
	}
	if err := layer.O.MatVec(s.tmp, s.attn); err != nil {
		return fmt.Errorf("attn_output: %w", err)
	}
	addInPlace(s.x, s.tmp)

	rmsNorm(s.norm, s.x, layer.FFNNorm, cfg.RMSNormEps)
	if err := layer.Gate.MatVec(s.gate, s.norm); err != nil {
		return fmt.Errorf("ffn_gate: %w", err)
	}
	if err := layer.Up.MatVec(s.up, s.norm); err != nil {
		return fmt.Errorf("ffn_up: %w", err)
	}
	for i := range s.gate {
		s.gate[i] = silu(s.gate[i]) * s.up[i]
	}
	if err := layer.Down.MatVec(s.tmp, s.gate); err != nil {
		return fmt.Errorf("ffn_down: %w", err)
	}
	addInPlace(s.x, s.tmp)
	return nil
}

func rmsNorm(dst, src, weight []float32, eps float32) {
	var sum float64
	for _, value := range src {
		sum += float64(value) * float64(value)
	}
	scale := float32(1.0 / math.Sqrt(sum/float64(len(src))+float64(eps)))
	for i := range src {
		dst[i] = src[i] * scale * weight[i]
	}
}

func rmsNormInPlace(x, weight []float32, eps float32) {
	var sum float64
	for _, value := range x {
		sum += float64(value) * float64(value)
	}
	scale := float32(1.0 / math.Sqrt(sum/float64(len(x))+float64(eps)))
	for i := range x {
		x[i] *= scale * weight[i]
	}
}

func applyRoPE(vec []float32, heads, stride, headDim, rotDim int, pos int, base float64) {
	half := rotDim / 2
	if half == 0 {
		return
	}
	for i := 0; i < half; i++ {
		theta := float64(pos) * math.Pow(base, -2*float64(i)/float64(rotDim))
		c := float32(math.Cos(theta))
		s := float32(math.Sin(theta))
		for h := 0; h < heads; h++ {
			off := h*stride + i
			x0 := vec[off]
			x1 := vec[off+half]
			vec[off] = x0*c - x1*s
			vec[off+half] = x0*s + x1*c
		}
	}
}

func softmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, value := range x[1:] {
		if value > max {
			max = value
		}
	}
	var sum float32
	for i, value := range x {
		e := float32(math.Exp(float64(value - max)))
		x[i] = e
		sum += e
	}
	inv := 1.0 / sum
	for i := range x {
		x[i] *= inv
	}
}

func dot(a, b []float32) float32 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
