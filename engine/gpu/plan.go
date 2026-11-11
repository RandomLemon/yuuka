//go:build hip && linux

package gpu

import (
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

// prefillMatmul appends the measured implementation for one prefill matmul.
func (m *Model) prefillMatmul(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	switch hip.SelectPrefillPath(w.dtype, rows, nIn, nOut) {
	case hip.PrefillBatchedGemv:
		return plan.AddGemvBatchAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
	case hip.PrefillBLAS:
		if err := plan.AddCastF16At(x, xOff, m.prefill.xF16, 0, rows*nIn); err != nil {
			return err
		}
		return plan.AddGemmPathAt(w.buf, 0, m.prefill.xF16, 0, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmBLAS)
	default:
		return plan.AddGemmPathAt(w.buf, 0, x, xOff, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmWMMA)
	}
}

// buildDecodePlan constructs the single-token graph over one session's block
// table and recurrent state.
func (m *Model) buildDecodePlan(s *Session, trace bool) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	embedding, err := m.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbedding(embedding.buf, m.act.x, int64(cfg.HiddenSize), int64(rowBytes), embedding.dtype); err != nil {
		return nil, err
	}
	for i := range cfg.BlockCount {
		prefix := fmt.Sprintf("blk.%d.", i)
		attnNorm, err := m.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := m.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			if err := m.addRecurrent(plan, s, i, attnNorm); err != nil {
				return nil, err
			}
		} else {
			if err := m.addFullAttention(plan, s, i, attnNorm); err != nil {
				return nil, err
			}
		}
		if err := m.addFFN(plan, prefix, postNorm); err != nil {
			return nil, err
		}
		if trace {
			if err := plan.AddCopy(s.traceBufs[i], m.act.x, int64(cfg.HiddenSize)); err != nil {
				return nil, err
			}
		}
	}
	plan.MarkTail()
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	if err := plan.AddRMSNorm(m.act.x, outputNorm.buf, m.act.norm, int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := m.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := m.weight(outputName)
	if err != nil {
		return nil, err
	}
	if err := plan.AddGemv(output.buf, m.act.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (m *Model) addFullAttention(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight) error {
	cfg := m.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := m.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := m.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := m.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := m.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := m.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := m.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNorm(m.act.x, attnNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(q.buf, m.act.norm, m.act.q, hidden, qDim, q.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(k.buf, m.act.norm, m.act.k, hidden, kvDim, k.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(v.buf, m.act.norm, m.act.v, hidden, kvDim, v.dtype); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(m.act.q, 0, qNorm.buf, m.act.q, 0,
		cfg.HeadCount, headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(m.act.k, 0, kNorm.buf, m.act.k, 0,
		cfg.KVHeadCount, headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPE(m.act.q, cfg.HeadCount, int(2*headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddRoPE(m.act.k, cfg.KVHeadCount, int(headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddKVWritePagedFormatAt(m.act.k, m.act.v, m.poolK[layerIndex], m.poolV[layerIndex],
		s.tableBuf, kvDim, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPagedFormatAt(m.act.q, m.poolK[layerIndex], m.poolV[layerIndex],
		m.act.attn, m.act.scores, s.tableBuf,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim),
		m.capacity, s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := plan.AddGemv(o.buf, m.act.attn, m.act.tmp, oDim, hidden, o.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

func (m *Model) addRecurrent(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight) error {
	cfg := m.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := m.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := m.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := m.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := m.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := m.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := m.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := m.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := m.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	keyDim := int64(cfg.SSMGroupCount * cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNorm(m.act.x, attnNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(qkv.buf, m.act.norm, m.act.qkv, hidden, convDim, qkv.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, m.act.norm, m.act.z, hidden, inner, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(beta.buf, m.act.norm, m.act.beta, hidden, rank, beta.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(alpha.buf, m.act.norm, m.act.alpha, hidden, rank, alpha.dtype); err != nil {
		return err
	}
	if err := plan.AddConv1D(s.conv[layerIndex], conv.buf, m.act.qkv, m.act.conv, int(convDim), cfg.SSMConvKernel); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if err := plan.AddL2NormAt(m.act.conv, 0, m.act.conv, 0,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, qScale); err != nil {
		return err
	}
	if err := plan.AddL2NormAt(m.act.conv, uint64(keyDim)*4, m.act.conv, uint64(keyDim)*4,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, 1); err != nil {
		return err
	}
	if err := plan.AddGDNAt(s.recState[layerIndex],
		m.act.conv, 0,
		m.act.conv, uint64(keyDim)*4,
		m.act.conv, uint64(2*keyDim)*4,
		m.act.alpha, m.act.beta, dt.buf, a.buf, m.act.gdn,
		cfg.SSMGroupCount, cfg.SSMTimeStepRank, int(stateSize)); err != nil {
		return err
	}
	if err := plan.AddGatedRMSNorm(m.act.gdn, norm.buf, m.act.z, m.act.gdnNorm,
		rank, stateSize, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(out.buf, m.act.gdnNorm, m.act.tmp, inner, hidden, out.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

func (m *Model) addFFN(plan *hip.Plan, prefix string, postNorm weight) error {
	cfg := m.config
	gate, err := m.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNorm(m.act.x, postNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, m.act.norm, m.act.ffnGate, hidden, ffn, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(up.buf, m.act.norm, m.act.ffnUp, hidden, ffn, up.dtype); err != nil {
		return err
	}
	if err := plan.AddSiluMul(m.act.ffnGate, m.act.ffnUp, m.act.ffnGate, ffn); err != nil {
		return err
	}
	if err := plan.AddGemv(down.buf, m.act.ffnGate, m.act.tmp, ffn, hidden, down.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

// buildPrefillPlan constructs the chunked prompt graph for one session.
func (m *Model) buildPrefillPlan(s *Session, rows int, allLogits bool) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	p := &m.prefill
	rows64 := int64(rows)
	embedding, err := m.weight("token_embd.weight")
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
		attnNorm, err := m.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := m.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			if err := m.addPrefillRecurrent(plan, s, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		} else {
			if err := m.addPrefillFullAttention(plan, s, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		}
		if err := m.addPrefillFFN(plan, prefix, postNorm, rows64); err != nil {
			return nil, err
		}
	}
	plan.MarkTail()
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := m.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := m.weight(outputName)
	if err != nil {
		return nil, err
	}
	if allLogits {
		if rows > maxSpecDraft {
			return nil, fmt.Errorf("gpu: speculative logits rows %d exceed %d", rows, maxSpecDraft)
		}
		if err := plan.AddRMSNormBatchAt(p.x, 0, outputNorm.buf, p.norm, 0,
			rows64, int64(cfg.HiddenSize), int64(cfg.HiddenSize), int64(cfg.HiddenSize),
			cfg.RMSNormEps); err != nil {
			return nil, err
		}
		for row := range rows {
			if err := plan.AddGemvAt(output.buf, 0, p.norm,
				uint64(row)*uint64(cfg.HiddenSize)*4, m.specLogits,
				uint64(row)*uint64(cfg.VocabSize)*4, int64(cfg.HiddenSize),
				int64(cfg.VocabSize), output.dtype); err != nil {
				return nil, err
			}
		}
		return plan, nil
	}
	lastOff := uint64(rows-1) * uint64(cfg.HiddenSize) * 4
	if err := plan.AddRMSNormAt(p.x, lastOff, outputNorm.buf, p.norm, 0,
		int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	if err := plan.AddGemv(output.buf, p.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (m *Model) addPrefillFullAttention(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight, rows int64) error {
	cfg := m.config
	p := &m.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := m.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := m.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := m.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := m.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := m.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := m.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, q, p.norm, 0, p.q, 0, hidden, qDim, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, k, p.norm, 0, p.k, 0, hidden, kvDim, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, v, p.norm, 0, p.v, 0, hidden, kvDim, rows); err != nil {
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
	if err := plan.AddKVWriteBatchPagedFormatAt(p.k, 0, p.v, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], s.tableBuf, kvDim, rows, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPrefillPagedFormatAt(p.q, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], p.attn, 0, s.tableBuf,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(rows),
		s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, o, p.attn, 0, p.tmp, 0, oDim, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

func (m *Model) addPrefillRecurrent(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight, rows int64) error {
	cfg := m.config
	p := &m.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := m.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := m.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := m.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := m.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := m.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := m.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := m.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := m.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, qkv, p.norm, 0, p.qkv, 0, hidden, convDim, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, gate, p.norm, 0, p.z, 0, hidden, inner, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, beta, p.norm, 0, p.beta, 0, hidden, rank, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, alpha, p.norm, 0, p.alpha, 0, hidden, rank, rows); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if err := plan.AddRecurrentChunkAt(hip.RecurrentChunkArgs{
		ConvState:   s.conv[layerIndex],
		State:       s.recState[layerIndex],
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
	if err := m.prefillMatmul(plan, out, p.gdnNorm, 0, p.tmp, 0, inner, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

func (m *Model) addPrefillFFN(plan *hip.Plan, prefix string, postNorm weight, rows int64) error {
	cfg := m.config
	p := &m.prefill
	gate, err := m.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, postNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, gate, p.norm, 0, p.ffnGate, 0, hidden, ffn, rows); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, up, p.norm, 0, p.ffnUp, 0, hidden, ffn, rows); err != nil {
		return err
	}
	if err := plan.AddSiluMul(p.ffnGate, p.ffnUp, p.ffnGate, rows*ffn); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, down, p.ffnGate, 0, p.tmp, 0, ffn, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}
