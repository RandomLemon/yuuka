package qwen35

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

// ForwardTrace is Forward with an optional per-layer hidden-state callback. The
// callback must copy x if it needs the values after the call returns.
func (m *Model) ForwardTrace(tokens []int32, st *State, trace func(layer int, x []float32)) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if st == nil {
		return nil, fmt.Errorf("nil state")
	}
	if st.Pos+len(tokens) > st.Capacity {
		return nil, fmt.Errorf("token sequence of %d exceeds remaining context %d",
			len(tokens), st.Capacity-st.Pos)
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= m.Config.VocabSize {
			return nil, fmt.Errorf("token id %d is out of range", token)
		}
		if err := m.forwardOne(token, st, i == len(tokens)-1, trace); err != nil {
			return nil, err
		}
	}
	return st.scratch.logits, nil
}

// Step processes one token and returns its logits.
func (m *Model) Step(token int32, st *State) ([]float32, error) {
	return m.Forward([]int32{token}, st)
}

// Prefill processes tokens and optionally computes logits for the final token.
// Chunked prefill uses wantLogits=false for every chunk that does not complete
// the prompt, which skips the LM head.
func (m *Model) Prefill(tokens []int32, st *State, wantLogits bool) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if st == nil {
		return nil, fmt.Errorf("nil state")
	}
	if st.Pos+len(tokens) > st.Capacity {
		return nil, fmt.Errorf("token sequence of %d exceeds remaining context %d",
			len(tokens), st.Capacity-st.Pos)
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= m.Config.VocabSize {
			return nil, fmt.Errorf("token id %d is out of range", token)
		}
		if err := m.forwardOne(token, st, wantLogits && i == len(tokens)-1, nil); err != nil {
			return nil, err
		}
	}
	if wantLogits {
		return st.scratch.logits, nil
	}
	return nil, nil
}

// MatVecOutput computes the LM-head logits for a normalized hidden vector.
func (m *Model) MatVecOutput(dst, hidden []float32) error {
	if len(hidden) != m.Config.HiddenSize {
		return fmt.Errorf("hidden vector length %d, want %d", len(hidden), m.Config.HiddenSize)
	}
	if len(dst) != m.Config.VocabSize {
		return fmt.Errorf("logit vector length %d, want %d", len(dst), m.Config.VocabSize)
	}
	return m.Output.MatVec(dst, hidden)
}

// LastHidden returns the normalized hidden state produced by the last forward
// call. The slice is owned by st and is valid until the next forward call.
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
		if layer.Full != nil {
			if err := m.forwardFullAttention(i, layer, st); err != nil {
				return err
			}
		} else {
			if err := m.forwardRecurrent(i, layer, st); err != nil {
				return err
			}
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

func (m *Model) forwardFullAttention(index int, layer *Layer, st *State) error {
	cfg := m.Config
	s := &st.scratch
	attn := layer.Full
	rmsNorm(s.norm, s.x, layer.AttnNorm, cfg.RMSNormEps)
	if err := attn.Q.MatVec(s.q, s.norm); err != nil {
		return fmt.Errorf("attn_q: %w", err)
	}
	if err := attn.K.MatVec(s.k, s.norm); err != nil {
		return fmt.Errorf("attn_k: %w", err)
	}
	if err := attn.V.MatVec(s.v, s.norm); err != nil {
		return fmt.Errorf("attn_v: %w", err)
	}

	headDim := cfg.HeadDim
	for h := 0; h < cfg.HeadCount; h++ {
		q := s.q[h*2*headDim : h*2*headDim+headDim]
		rmsNormInPlace(q, attn.QNorm, cfg.RMSNormEps)
	}
	for h := 0; h < cfg.KVHeadCount; h++ {
		k := s.k[h*headDim : (h+1)*headDim]
		rmsNormInPlace(k, attn.KNorm, cfg.RMSNormEps)
	}
	applyRoPE(s.q, cfg.HeadCount, 2*headDim, headDim, cfg.RoPEDim, st.Pos, cfg.RoPEFreqBase)
	applyRoPE(s.k, cfg.KVHeadCount, headDim, headDim, cfg.RoPEDim, st.Pos, cfg.RoPEFreqBase)

	kvDim := cfg.KVHeadCount * headDim
	keys := st.pool.k[index]
	values := st.pool.v[index]
	if keys == nil || values == nil {
		return fmt.Errorf("layer %d is recurrent and has no KV cache", index)
	}
	offset, err := st.pool.elementOffset(st.table, st.Pos)
	if err != nil {
		return err
	}
	copy(keys[offset:offset+kvDim], s.k)
	copy(values[offset:offset+kvDim], s.v)

	group := cfg.KVHeadsPerGroup()
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	length := st.Pos + 1
	for h := 0; h < cfg.HeadCount; h++ {
		kvh := h / group
		q := s.q[h*2*headDim : h*2*headDim+headDim]
		scores := s.scores[:length]
		for p := 0; p < length; p++ {
			base, err := st.pool.elementOffset(st.table, p)
			if err != nil {
				return err
			}
			base += kvh * headDim
			scores[p] = dot(q, keys[base:base+headDim]) * scale
		}
		softmaxInPlace(scores)
		out := s.attn[h*headDim : (h+1)*headDim]
		clear(out)
		for p := 0; p < length; p++ {
			weight := scores[p]
			base, err := st.pool.elementOffset(st.table, p)
			if err != nil {
				return err
			}
			base += kvh * headDim
			v := values[base : base+headDim]
			for d := range out {
				out[d] += weight * v[d]
			}
		}
		gate := s.q[h*2*headDim+headDim : h*2*headDim+2*headDim]
		for d := range out {
			out[d] *= sigmoid(gate[d])
		}
	}
	if err := attn.O.MatVec(s.tmp, s.attn); err != nil {
		return fmt.Errorf("attn_output: %w", err)
	}
	addInPlace(s.x, s.tmp)
	return m.forwardFFN(layer, st)
}

func (m *Model) forwardRecurrent(index int, layer *Layer, st *State) error {
	cfg := m.Config
	s := &st.scratch
	rec := layer.Rec
	rmsNorm(s.norm, s.x, layer.AttnNorm, cfg.RMSNormEps)
	if err := rec.QKV.MatVec(s.qkv, s.norm); err != nil {
		return fmt.Errorf("attn_qkv: %w", err)
	}
	if err := rec.Gate.MatVec(s.z, s.norm); err != nil {
		return fmt.Errorf("attn_gate: %w", err)
	}
	if err := rec.Beta.MatVec(s.beta, s.norm); err != nil {
		return fmt.Errorf("ssm_beta: %w", err)
	}
	if err := rec.Alpha.MatVec(s.alpha, s.norm); err != nil {
		return fmt.Errorf("ssm_alpha: %w", err)
	}
	for h := 0; h < cfg.SSMTimeStepRank; h++ {
		s.beta[h] = sigmoid(s.beta[h])
		s.alpha[h] = softplus(s.alpha[h]+rec.DT[h]) * rec.A[h]
	}

	state := st.rec[index]
	convDim := len(s.qkv)
	kernel := cfg.SSMConvKernel
	for c := 0; c < convDim; c++ {
		var sum float32
		base := c * kernel
		stateBase := c * (kernel - 1)
		for t := 0; t < kernel-1; t++ {
			sum += rec.Conv[base+t] * state.conv[stateBase+t]
		}
		sum += rec.Conv[base+kernel-1] * s.qkv[c]
		s.conv[c] = silu(sum)
		for t := 0; t < kernel-2; t++ {
			state.conv[stateBase+t] = state.conv[stateBase+t+1]
		}
		if kernel > 1 {
			state.conv[stateBase+kernel-2] = s.qkv[c]
		}
	}

	keyDim := cfg.SSMGroupCount * cfg.SSMStateSize
	valueDim := cfg.SSMInnerSize
	qConv := s.conv[:keyDim]
	kConv := s.conv[keyDim : 2*keyDim]
	vConv := s.conv[2*keyDim : 2*keyDim+valueDim]
	stateSize := cfg.SSMStateSize
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	for h := 0; h < cfg.SSMGroupCount; h++ {
		base := h * stateSize
		l2NormHead(qConv[base:base+stateSize], cfg.RMSNormEps, qScale)
		l2NormHead(kConv[base:base+stateSize], cfg.RMSNormEps, 1)
	}

	for h := 0; h < cfg.SSMTimeStepRank; h++ {
		kh := h % cfg.SSMGroupCount
		q := qConv[kh*stateSize : (kh+1)*stateSize]
		k := kConv[kh*stateSize : (kh+1)*stateSize]
		v := vConv[h*stateSize : (h+1)*stateSize]
		sBase := h * stateSize * stateSize
		matrix := state.s[sBase : sBase+stateSize*stateSize]
		decay := float32(math.Exp(float64(s.alpha[h])))
		for i := range matrix {
			matrix[i] *= decay
		}
		clear(s.sk)
		for i := 0; i < stateSize; i++ {
			ki := k[i]
			row := matrix[i*stateSize : (i+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				s.sk[j] += row[j] * ki
			}
		}
		beta := s.beta[h]
		for j := 0; j < stateSize; j++ {
			s.d[j] = (v[j] - s.sk[j]) * beta
		}
		for i := 0; i < stateSize; i++ {
			ki := k[i]
			row := matrix[i*stateSize : (i+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				row[j] += ki * s.d[j]
			}
		}
		out := s.gdn[h*stateSize : (h+1)*stateSize]
		clear(out)
		for i := 0; i < stateSize; i++ {
			qi := q[i]
			row := matrix[i*stateSize : (i+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				out[j] += row[j] * qi
			}
		}
		rmsNormInPlace(out, rec.Norm, cfg.RMSNormEps)
		for j := 0; j < stateSize; j++ {
			out[j] *= silu(s.z[h*stateSize+j])
		}
	}
	if err := rec.Out.MatVec(s.tmp, s.gdn); err != nil {
		return fmt.Errorf("ssm_out: %w", err)
	}
	addInPlace(s.x, s.tmp)
	return m.forwardFFN(layer, st)
}

func (m *Model) forwardFFN(layer *Layer, st *State) error {
	s := &st.scratch
	rmsNorm(s.norm, s.x, layer.PostNorm, m.Config.RMSNormEps)
	if err := layer.FFN.Gate.MatVec(s.ffnGate, s.norm); err != nil {
		return fmt.Errorf("ffn_gate: %w", err)
	}
	if err := layer.FFN.Up.MatVec(s.ffnUp, s.norm); err != nil {
		return fmt.Errorf("ffn_up: %w", err)
	}
	for i := range s.ffnGate {
		s.ffnGate[i] = silu(s.ffnGate[i]) * s.ffnUp[i]
	}
	if err := layer.FFN.Down.MatVec(s.tmp, s.ffnGate); err != nil {
		return fmt.Errorf("ffn_down: %w", err)
	}
	addInPlace(s.x, s.tmp)
	return nil
}

func rmsNorm(dst, src, weight []float32, eps float32) {
	var sum float64
	for _, v := range src {
		sum += float64(v) * float64(v)
	}
	mean := float32(sum / float64(len(src)))
	scale := float32(1.0 / math.Sqrt(float64(mean+eps)))
	for i := range dst {
		dst[i] = src[i] * scale * weight[i]
	}
}

func rmsNormInPlace(x, weight []float32, eps float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	mean := float32(sum / float64(len(x)))
	scale := float32(1.0 / math.Sqrt(float64(mean+eps)))
	for i := range x {
		x[i] *= scale * weight[i]
	}
}

func l2NormHead(x []float32, eps float32, extra float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	scale := float32(1.0/math.Sqrt(sum+float64(eps))) * extra
	for i := range x {
		x[i] *= scale
	}
}

func applyRoPE(vec []float32, heads, stride, headDim, rotDim int, pos int, base float64) {
	half := rotDim / 2
	if half == 0 {
		return
	}
	// The Qwen3.5 MRoPE sections collapse to standard RoPE when all position
	// components are equal, which is the text-only path.
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
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	inv := 1.0 / sum
	for i := range x {
		x[i] *= inv
	}
}

func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func silu(x float32) float32 {
	return x * sigmoid(x)
}

func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}
