//go:build hip && linux

package gpu

import (
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

func (m *MoEModel) moePrefillMatmul(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	switch hip.SelectPrefillPath(w.dtype, rows, nIn, nOut) {
	case hip.PrefillBatchedGemv:
		return plan.AddGemvBatchAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
	case hip.PrefillBLAS:
		if err := plan.AddCastF16At(x, xOff, m.act.xF16, 0, rows*nIn); err != nil {
			return err
		}
		return plan.AddGemmPathAt(w.buf, 0, m.act.xF16, 0, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmBLAS)
	default:
		return plan.AddGemmPathAt(w.buf, 0, x, xOff, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmWMMA)
	}
}

func (m *MoEModel) embeddingWeight() (weight, uint64, error) {
	w, err := m.weight("token_embd.weight")
	if err != nil {
		return weight{}, 0, err
	}
	rowBytes, err := w.dtype.RowBytes(int64(m.config.HiddenSize))
	return w, rowBytes, err
}

func (m *MoEModel) outputWeight() (weight, error) {
	if _, ok := m.weights["output.weight"]; ok {
		return m.weight("output.weight")
	}
	return m.weight("token_embd.weight")
}

// buildMoEDecodePlan builds the single-token graph for one session.
func (m *MoEModel) buildMoEDecodePlan(s *MoESession, trace bool) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	a := &m.act
	embedding, rowBytes, err := m.embeddingWeight()
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbedding(embedding.buf, a.x, int64(cfg.HiddenSize),
		int64(rowBytes), embedding.dtype); err != nil {
		return nil, err
	}
	for i := range cfg.BlockCount {
		if err := m.addMoEAttentionDecode(plan, s, i); err != nil {
			return nil, err
		}
		if err := m.addMoEDecodeFFN(plan, i); err != nil {
			return nil, err
		}
		if trace {
			if err := plan.AddCopy(s.traceBufs[i], a.x, int64(cfg.HiddenSize)); err != nil {
				return nil, err
			}
		}
	}
	plan.MarkTail()
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	if err := plan.AddRMSNorm(a.x, outputNorm.buf, a.norm, int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	output, err := m.outputWeight()
	if err != nil {
		return nil, err
	}
	return plan, plan.AddGemv(output.buf, a.norm, a.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype)
}

func (m *MoEModel) addMoEAttentionDecode(plan *hip.Plan, s *MoESession, layerIndex int) error {
	cfg := m.config
	a := &m.act
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	attnNorm, err := m.weight(prefix + "attn_norm.weight")
	if err != nil {
		return err
	}
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
	qDim := int64(cfg.HeadCount * cfg.HeadDim)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	if err := plan.AddRMSNorm(a.x, attnNorm.buf, a.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(q.buf, a.norm, a.q, hidden, qDim, q.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(k.buf, a.norm, a.k, hidden, kvDim, k.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(v.buf, a.norm, a.v, hidden, kvDim, v.dtype); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(a.q, 0, qNorm.buf, a.q, 0,
		cfg.HeadCount, headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(a.k, 0, kNorm.buf, a.k, 0,
		cfg.KVHeadCount, headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPE(a.q, cfg.HeadCount, int(headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddRoPE(a.k, cfg.KVHeadCount, int(headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddKVWritePagedFormatAt(a.k, a.v, m.poolK[layerIndex], m.poolV[layerIndex],
		s.tableBuf, kvDim, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPagedDenseFormatAt(a.q, m.poolK[layerIndex], m.poolV[layerIndex],
		a.attn, a.scores, s.tableBuf, cfg.KVHeadCount, cfg.HeadCount, int(headDim),
		int(kvDim), m.capacity, s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := plan.AddGemv(o.buf, a.attn, a.tmp, int64(cfg.HeadCount*cfg.HeadDim), hidden, o.dtype); err != nil {
		return err
	}
	return plan.Add(a.x, a.tmp, hidden)
}

func (m *MoEModel) addMoEDecodeFFN(plan *hip.Plan, layerIndex int) error {
	cfg := m.config
	a := &m.act
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	postNorm, err := m.weight(prefix + "ffn_norm.weight")
	if err != nil {
		return err
	}
	gateInp, err := m.weight(prefix + "ffn_gate_inp.weight")
	if err != nil {
		return err
	}
	gateExps, err := m.weight(prefix + "ffn_gate_exps.weight")
	if err != nil {
		return err
	}
	upExps, err := m.weight(prefix + "ffn_up_exps.weight")
	if err != nil {
		return err
	}
	downExps, err := m.weight(prefix + "ffn_down_exps.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.ExpertFFNSize)
	experts := int64(cfg.ExpertCount)
	used := int64(cfg.ExpertUsedCount)
	if err := plan.AddRMSNorm(a.x, postNorm.buf, a.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(gateInp.buf, a.norm, a.router, hidden, experts, gateInp.dtype); err != nil {
		return err
	}
	if err := plan.AddMoERoute(a.router, a.ids, a.weights, a.counts,
		a.assignRows, a.assignSlots, a.assignWeights, 1, experts, used); err != nil {
		return err
	}
	if err := plan.AddMoEGateUpGrouped(a.norm, gateExps.buf, upExps.buf, a.counts,
		a.assignRows, a.assignSlots, a.gateOut, a.upOut, hidden, ffn, 1,
		experts, used, gateExps.dtype); err != nil {
		return err
	}
	if err := plan.AddSiluMul(a.gateOut, a.upOut, a.actOut, used*ffn); err != nil {
		return err
	}
	if err := plan.AddMoEDownGrouped(a.actOut, downExps.buf, a.counts,
		a.assignRows, a.assignSlots, a.expertOut, ffn, hidden, 1, experts,
		used, downExps.dtype); err != nil {
		return err
	}
	return plan.AddMoEReduceAdd(a.x, a.expertOut, a.weights, hidden, 1, used)
}

// buildMoEPrefillPlan builds the chunked prompt graph for one session.
func (m *MoEModel) buildMoEPrefillPlan(s *MoESession, rows int) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	a := &m.act
	rows64 := int64(rows)
	embedding, rowBytes, err := m.embeddingWeight()
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbeddingBatchAt(embedding.buf, a.x, a.tokens, 0, rows64,
		int64(cfg.HiddenSize), int64(rowBytes), int64(cfg.VocabSize), embedding.dtype); err != nil {
		return nil, err
	}
	for i := range cfg.BlockCount {
		if err := m.addMoEAttentionPrefill(plan, s, i, rows64); err != nil {
			return nil, err
		}
		if err := m.addMoEPrefillFFN(plan, i, rows64); err != nil {
			return nil, err
		}
	}
	plan.MarkTail()
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	lastOff := uint64(rows-1) * uint64(cfg.HiddenSize) * 4
	if err := plan.AddRMSNormAt(a.x, lastOff, outputNorm.buf, a.norm, 0,
		int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	output, err := m.outputWeight()
	if err != nil {
		return nil, err
	}
	return plan, plan.AddGemv(output.buf, a.norm, a.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype)
}

func (m *MoEModel) addMoEAttentionPrefill(plan *hip.Plan, s *MoESession, layerIndex int, rows int64) error {
	cfg := m.config
	a := &m.act
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	attnNorm, err := m.weight(prefix + "attn_norm.weight")
	if err != nil {
		return err
	}
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
	qDim := int64(cfg.HeadCount * cfg.HeadDim)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	if err := plan.AddRMSNormBatchAt(a.x, 0, attnNorm.buf, a.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.moePrefillMatmul(plan, q, a.norm, 0, a.q, 0, hidden, qDim, rows); err != nil {
		return err
	}
	if err := m.moePrefillMatmul(plan, k, a.norm, 0, a.k, 0, hidden, kvDim, rows); err != nil {
		return err
	}
	if err := m.moePrefillMatmul(plan, v, a.norm, 0, a.v, 0, hidden, kvDim, rows); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(a.q, 0, qNorm.buf, a.q, 0,
		rows, int64(cfg.HeadCount), headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(a.k, 0, kNorm.buf, a.k, 0,
		rows, int64(cfg.KVHeadCount), headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(a.q, 0, int(rows), cfg.HeadCount, int(headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.HeadCount*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(a.k, 0, int(rows), cfg.KVHeadCount, int(headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.KVHeadCount*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddKVWriteBatchPagedFormatAt(a.k, 0, a.v, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], s.tableBuf, kvDim, rows, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPrefillPagedDenseFormatAt(a.q, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], a.attn, 0, s.tableBuf, cfg.KVHeadCount, cfg.HeadCount,
		int(headDim), int(kvDim), int(rows), s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := m.moePrefillMatmul(plan, o, a.attn, 0, a.tmp, 0,
		int64(cfg.HeadCount*cfg.HeadDim), hidden, rows); err != nil {
		return err
	}
	return plan.Add(a.x, a.tmp, rows*hidden)
}

func (m *MoEModel) addMoEPrefillFFN(plan *hip.Plan, layerIndex int, rows int64) error {
	cfg := m.config
	a := &m.act
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	postNorm, err := m.weight(prefix + "ffn_norm.weight")
	if err != nil {
		return err
	}
	gateInp, err := m.weight(prefix + "ffn_gate_inp.weight")
	if err != nil {
		return err
	}
	gateExps, err := m.weight(prefix + "ffn_gate_exps.weight")
	if err != nil {
		return err
	}
	upExps, err := m.weight(prefix + "ffn_up_exps.weight")
	if err != nil {
		return err
	}
	downExps, err := m.weight(prefix + "ffn_down_exps.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.ExpertFFNSize)
	experts := int64(cfg.ExpertCount)
	used := int64(cfg.ExpertUsedCount)
	if err := plan.AddRMSNormBatchAt(a.x, 0, postNorm.buf, a.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.moePrefillMatmul(plan, gateInp, a.norm, 0, a.router, 0, hidden, experts, rows); err != nil {
		return err
	}
	if err := plan.AddMoERoute(a.router, a.ids, a.weights, a.counts,
		a.assignRows, a.assignSlots, a.assignWeights, rows, experts, used); err != nil {
		return err
	}
	if err := plan.AddMoEGateUpGrouped(a.norm, gateExps.buf, upExps.buf, a.counts,
		a.assignRows, a.assignSlots, a.gateOut, a.upOut, hidden, ffn, rows,
		experts, used, gateExps.dtype); err != nil {
		return err
	}
	if err := plan.AddSiluMul(a.gateOut, a.upOut, a.actOut, rows*used*ffn); err != nil {
		return err
	}
	if err := plan.AddMoEDownGrouped(a.actOut, downExps.buf, a.counts,
		a.assignRows, a.assignSlots, a.expertOut, ffn, hidden, rows, experts,
		used, downExps.dtype); err != nil {
		return err
	}
	return plan.AddMoEReduceAdd(a.x, a.expertOut, a.weights, hidden, rows, used)
}
