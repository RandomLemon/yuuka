//go:build hip && linux

package gpu

import (
	"fmt"
	"math"
	"strings"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// prefillMatmul appends the measured implementation for one prefill matmul.
func (m *Model) prefillMatmul(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	return m.prefillMatmulPath(plan, w, x, xOff, out, outOff, nIn, nOut, rows,
		hip.SelectPrefillPath(w.dtype, rows, nIn, nOut))
}

func (m *Model) prefillMatmulPath(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64, path hip.PrefillPath) error {
	switch path {
	case hip.PrefillBatchedGemv:
		return plan.AddGemvBatchAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
	case hip.PrefillBLAS:
		if err := plan.AddCastF16At(x, xOff, m.prefill.xF16, 0, rows*nIn); err != nil {
			return err
		}
		return plan.AddGemmBlasAt(w.buf, 0, m.prefill.xF16, m.prefill.yF16, out,
			outOff, nIn, nOut, rows)
	case hip.PrefillMMQ:
		return m.prefillMMQPrepared(plan, w, x, xOff, out, outOff, nIn, nOut, rows)
	default:
		return plan.AddGemmPathAt(w.buf, 0, x, xOff, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmWMMA)
	}
}

// decodeMatmul appends one decode matvec. The channel-parallel kernel serves
// every shape the decode graph dispatches except the 32-output recurrent
// projections, where the per-row GEMV stays; within that set the K-quants and
// Q8_0 dot the Q8_1 activation with int8 v_dot4 and the other dtypes keep FP32
// (docs/kernels.md section 5.3).
//
// The int8 route reads digits that decodeActGroup quantized for this group of
// matvecs, so a matvec on a row that was not prepared is a planning error and
// fails here rather than reading stale activations.
func (m *Model) decodeMatmul(plan *hip.Plan, w weight, x, out *hip.Buffer,
	nIn, nOut int64) error {
	path := hip.DecodeGemvPerRow
	if !m.forceDecodeGemv {
		path = hip.SelectDecodeGemvPath(w.dtype, nIn, nOut, m.int8Matvec)
	}
	switch path {
	case hip.DecodeGemvInt8:
		if !m.prequantizedInt8 {
			// The block quantizes the row it reads, so no quantization
			// operation is submitted for the group.
			return plan.AddGemvQ8_1FusedAt(w.buf, 0, x, out, nIn, nOut, w.dtype)
		}
		if m.int8Act != x || m.int8ActWidth != nIn {
			return fmt.Errorf("gpu: %s matvec on an activation row that was not"+
				" quantized for the group", hip.DecodeGemvInt8)
		}
		return plan.AddGemvQ8_1At(w.buf, 0, m.act8.qs, m.act8.ds, out, nIn, nOut, w.dtype)
	case hip.DecodeGemvStreaming:
		return plan.AddLMHeadAt(w.buf, 0, x, 0, out, 0, nIn, nOut, 1, w.dtype)
	default:
		return plan.AddGemv(w.buf, x, out, nIn, nOut, w.dtype)
	}
}

// decodeMatmulResidual appends the decode matvec that ends a sub-layer together
// with its residual add. out = W*x + res in one launch when the streaming kernel
// is selected; the per-row route
// keeps the two operations it has always had, because that kernel has no
// epilogue and the choice is made here, at plan build time, not at run time.
func (m *Model) decodeMatmulResidual(plan *hip.Plan, w weight, x, out, res *hip.Buffer,
	nIn, nOut int64) error {
	if !m.noResidualFusion && !m.forceDecodeGemv && !m.int8Matvec &&
		hip.SelectDecodeGemvPath(w.dtype, nIn, nOut, false) == hip.DecodeGemvStreaming {
		return plan.AddGemvResidualAt(w.buf, 0, x, 0, out, 0, res, 0, nIn, nOut, 1, w.dtype)
	}
	if err := m.decodeMatmul(plan, w, x, m.act.tmp, nIn, nOut); err != nil {
		return err
	}
	return plan.Add(out, m.act.tmp, nOut)
}

// decodeMatmulPair appends two projections that read the same activation row as
// one launch. It needs both shapes on the streaming route, because the fused
// block forms both dots over one traversal of the row; otherwise the plan keeps
// the two matvecs, and with the SwiGLU epilogue the separate SILU_MUL as well.
//
// The choice is made here, at plan build time, and the two forms are asserted to
// agree by the differential tests in decode_fusion_test.go.
func (m *Model) decodeMatmulPair(plan *hip.Plan, a, b weight, x, outA, outB *hip.Buffer,
	nIn, nOutA, nOutB int64, epilogue int) error {
	// The pair kernel decodes both matrices with one instantiation, so it needs
	// one weight dtype; a mixed pair keeps the two matvecs. That is a plan-build
	// decision, not a silent route change.
	streaming := hip.SelectDecodeGemvPath(a.dtype, nIn, nOutA, false) ==
		hip.DecodeGemvStreaming &&
		hip.SelectDecodeGemvPath(b.dtype, nIn, nOutB, false) == hip.DecodeGemvStreaming
	// A narrow equal-shape pair (the recurrent beta/alpha rank projection) takes
	// the same fused launch even though each matrix alone is below the streaming
	// crossover: one launch over both rows is what pays, and the pair kernel
	// covers the shape when the K step is a multiple of 512.
	narrow := epilogue == hip.PairWriteBoth && nOutA == nOutB && nIn%512 == 0 &&
		a.dtype == b.dtype && (a.dtype == tensor.Q4_K || a.dtype == tensor.Q5_K ||
		a.dtype == tensor.Q6_K || a.dtype == tensor.Q8_0)
	usable := !m.noPairFusion && !m.forceDecodeGemv && !m.int8Matvec &&
		a.dtype == b.dtype && (streaming || narrow) &&
		(epilogue != hip.PairSwiglu || nOutA == nOutB)
	if usable {
		// The SwiGLU epilogue consumes the second row inside the kernel, so the
		// fused form takes no second output.
		second := outB
		if epilogue == hip.PairSwiglu {
			second = nil
		}
		return plan.AddGemvPairAt(a.buf, 0, b.buf, 0, x, 0, outA, 0, second, 0, nIn,
			nOutA, nOutB, a.dtype, b.dtype, epilogue)
	}
	if err := m.decodeMatmul(plan, a, x, outA, nIn, nOutA); err != nil {
		return err
	}
	if err := m.decodeMatmul(plan, b, x, outB, nIn, nOutB); err != nil {
		return err
	}
	if epilogue == hip.PairSwiglu {
		return plan.AddSiluMul(outA, outB, outA, nOutA)
	}
	return nil
}

// decodeMatmulSwiglu appends the gate and up projections and their SwiGLU
// product, either as one fused launch or as two matvecs and a SILU_MUL.
func (m *Model) decodeMatmulSwiglu(plan *hip.Plan, gate, up weight, x, out *hip.Buffer,
	nIn, nOut int64) error {
	return m.decodeMatmulPair(plan, gate, up, x, out, m.act.ffnUp, nIn, nOut, nOut,
		hip.PairSwiglu)
}

// decodeActGroup prepares the Q8_1 digits of one decode activation row for the
// matvecs that read it. It emits one quantization operation for the group, which
// is what keeps the activation from being re-quantized per projection, and skips
// it when no consumer takes the int8 route so a model whose weights are all FP32
// pays nothing. The plan is sequential, so the digits stay valid until the next
// group.
func (m *Model) decodeActGroup(plan *hip.Plan, x *hip.Buffer, nIn int64,
	consumers ...tensor.DType) error {
	m.int8Act = nil
	m.int8ActWidth = 0
	if m.forceDecodeGemv || !m.int8Matvec || !m.prequantizedInt8 {
		return nil
	}
	needed := false
	for _, dtype := range consumers {
		if hip.IsInt8DecodeDType(dtype) {
			needed = true
			break
		}
	}
	if !needed {
		return nil
	}
	if err := plan.AddQuantizeQ8_1At(x, 0, m.act8.qs, m.act8.ds, nIn, 1); err != nil {
		return err
	}
	m.int8Act = x
	m.int8ActWidth = nIn
	return nil
}

// prefillProjection is one prefill matmul that reads the shared activation
// tile at offset zero.
type prefillProjection struct {
	w    weight
	out  *hip.Buffer
	nOut int64
}

// prefillProjections runs several projections that read the same activation
// tile. When two or more of them select the MMQ route, the Q8_1 quantization of
// that tile runs once for the group instead of once per projection; the q/k/v
// and qkv/gate groups are the measured case. Projections on other routes keep
// their own implementation and their own input handling.
func (m *Model) prefillProjections(plan *hip.Plan, x *hip.Buffer, nIn, rows int64,
	projections []prefillProjection) error {
	mmq := 0
	for _, projection := range projections {
		if hip.SelectPrefillPath(projection.w.dtype, rows, nIn, projection.nOut) == hip.PrefillMMQ {
			mmq++
		}
	}
	if mmq > 1 {
		if err := plan.AddQuantizeQ8_1At(x, 0, m.prefill.xQ8, m.prefill.xDS, nIn, rows); err != nil {
			return err
		}
	}
	for _, projection := range projections {
		if mmq > 1 && hip.SelectPrefillPath(projection.w.dtype, rows, nIn, projection.nOut) == hip.PrefillMMQ {
			if err := plan.AddGemmMmqAt(projection.w.buf, 0, m.prefill.xQ8, m.prefill.xDS,
				projection.out, 0, nIn, projection.nOut, rows, projection.w.dtype); err != nil {
				return err
			}
			continue
		}
		if err := m.prefillMatmul(plan, projection.w, x, 0, projection.out, 0, nIn,
			projection.nOut, rows); err != nil {
			return err
		}
	}
	return nil
}

// prefillMMQPrepared quantizes the activation tile to Q8_1 and runs the int8
// GEMM. The quantization is a separate plan operation so a gate/up pair that
// reads the same normalized tile can share it.
func (m *Model) prefillMMQPrepared(plan *hip.Plan, w weight, x *hip.Buffer,
	xOff uint64, out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	if err := plan.AddQuantizeQ8_1At(x, xOff, m.prefill.xQ8, m.prefill.xDS, nIn, rows); err != nil {
		return err
	}
	return plan.AddGemmMmqAt(w.buf, 0, m.prefill.xQ8, m.prefill.xDS, out, outOff,
		nIn, nOut, rows, w.dtype)
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
	if err := m.decodeActGroup(plan, m.act.norm, int64(cfg.HiddenSize), output.dtype); err != nil {
		return nil, err
	}
	if err := m.decodeMatmul(plan, output, m.act.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize)); err != nil {
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
	// q, k, and v read the same normalized row, so its Q8_1 digits are produced
	// once for the group.
	if err := m.decodeActGroup(plan, m.act.norm, hidden, q.dtype, k.dtype, v.dtype); err != nil {
		return err
	}
	// q, k, and v are independent and read the same row, but only a pair of
	// comparable widths pays for a merge: the q and k pair was measured 0.4 ms
	// slower per step than two launches, because the grid follows the wider
	// matrix and the narrow one then shares its staging. They keep
	// their own launches.
	if err := m.decodeMatmul(plan, q, m.act.norm, m.act.q, hidden, qDim); err != nil {
		return err
	}
	if err := m.decodeMatmul(plan, k, m.act.norm, m.act.k, hidden, kvDim); err != nil {
		return err
	}
	if err := m.decodeMatmul(plan, v, m.act.norm, m.act.v, hidden, kvDim); err != nil {
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
	if err := m.decodeActGroup(plan, m.act.attn, oDim, o.dtype); err != nil {
		return err
	}
	return m.decodeMatmulResidual(plan, o, m.act.attn, m.act.x, m.act.x, oDim, hidden)
}

// fusedDeltaNet reports whether the recurrent layer can take the fused
// conv_l2norm and recurrent_token pair. Both kernels are compiled for a 128-wide
// state and for norm groups and channel blocks that do not cross, so a model
// with any other recurrent shape keeps the separate operators instead of failing
// the plan.
func (m *Model) fusedDeltaNet(convDim, qkDim, keyDim, stateSize int64) bool {
	if m.legacyDeltaNet {
		return false
	}
	return stateSize == 128 && convDim%256 == 0 && qkDim%256 == 0 && keyDim%128 == 0
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
	// The qkv, gate, alpha, and beta projections all read the normalized row.
	if err := m.decodeActGroup(plan, m.act.norm, hidden, qkv.dtype, gate.dtype,
		alpha.dtype, beta.dtype); err != nil {
		return err
	}
	// The qkv projection and the delta-net gate read the same row, but their
	// dtypes differ in this model (Q5_K against Q4_K) and their widths are 2:1, so
	// they keep separate launches.
	if err := m.decodeMatmul(plan, qkv, m.act.norm, m.act.qkv, hidden, convDim); err != nil {
		return err
	}
	if err := m.decodeMatmul(plan, gate, m.act.norm, m.act.z, hidden, inner); err != nil {
		return err
	}
	// beta and alpha are the same narrow shape over the same row. One launch
	// reads the activation once for both instead of once per matrix; the pair
	// form is the same kernel the gate/up projection uses, and the two per-row
	// launches stay for a shape it does not cover.
	if err := m.decodeMatmulPair(plan, beta, alpha, m.act.norm, m.act.beta, m.act.alpha,
		hidden, rank, rank, hip.PairWriteBoth); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if m.fusedDeltaNet(convDim, 2*keyDim, keyDim, stateSize) {
		// conv_l2norm advances the convolution state and normalizes the q and k
		// channels; recurrent_token keeps the head's state in registers and
		// writes the gated RMSNorm of the head directly, so the raw delta-net
		// row (m.act.gdn) is not materialized.
		if err := plan.AddConvL2Norm(s.conv[layerIndex], conv.buf, m.act.qkv, m.act.conv,
			int(convDim), cfg.SSMConvKernel, int(2*keyDim), int(keyDim), int(stateSize),
			cfg.RMSNormEps, qScale); err != nil {
			return err
		}
		if err := plan.AddRecurrentTokenAt(s.recState[layerIndex],
			m.act.conv, 0,
			m.act.conv, uint64(keyDim)*4,
			m.act.conv, uint64(2*keyDim)*4,
			m.act.alpha, m.act.beta, dt.buf, a.buf, m.act.gdnNorm, norm.buf, m.act.z,
			cfg.SSMGroupCount, cfg.SSMTimeStepRank, int(stateSize),
			cfg.RMSNormEps); err != nil {
			return err
		}
	} else {
		if err := plan.AddConv1D(s.conv[layerIndex], conv.buf, m.act.qkv, m.act.conv, int(convDim), cfg.SSMConvKernel); err != nil {
			return err
		}
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
	}
	if err := m.decodeActGroup(plan, m.act.gdnNorm, inner, out.dtype); err != nil {
		return err
	}
	return m.decodeMatmulResidual(plan, out, m.act.gdnNorm, m.act.x, m.act.x, inner, hidden)
}

func (m *Model) addFFN(plan *hip.Plan, prefix string, postNorm weight) error {
	cfg := m.config
	if cfg.IsMoE() {
		return m.addMoEDecodeFFN(plan, prefix, postNorm)
	}
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
	// Gate and up read the same normalized row.
	if err := m.decodeActGroup(plan, m.act.norm, hidden, gate.dtype, up.dtype); err != nil {
		return err
	}
	if err := m.decodeMatmulSwiglu(plan, gate, up, m.act.norm, m.act.ffnGate, hidden, ffn); err != nil {
		return err
	}
	if err := m.decodeActGroup(plan, m.act.ffnGate, ffn, down.dtype); err != nil {
		return err
	}
	// The projection that ends the sub-layer adds the residual itself.
	return m.decodeMatmulResidual(plan, down, m.act.ffnGate, m.act.x, m.act.x, ffn, hidden)
}

// addMoEDecodeFFN builds one MoE sub-layer for a single token. The router picks
// cfg.ExpertUsedCount experts, the selected-expert kernels compute only those
// expert rows, the routing weights accumulate into the residual stream, and the
// shared expert adds its sigmoid-gated result. The grouped kernels stay on the
// prefill path: their grid is (experts, ffn) blocks, which would schedule every
// unselected expert row for one token.
func (m *Model) addMoEDecodeFFN(plan *hip.Plan, prefix string, postNorm weight) error {
	cfg := m.config
	router, err := m.weight(prefix + "ffn_gate_inp.weight")
	if err != nil {
		return err
	}
	sharedGateVec, err := m.weight(prefix + "ffn_gate_inp_shexp.weight")
	if err != nil {
		return err
	}
	sharedGate, err := m.weight(prefix + "ffn_gate_shexp.weight")
	if err != nil {
		return err
	}
	sharedUp, err := m.weight(prefix + "ffn_up_shexp.weight")
	if err != nil {
		return err
	}
	sharedDown, err := m.weight(prefix + "ffn_down_shexp.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "ffn_gate_exps.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up_exps.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down_exps.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	experts := int64(cfg.ExpertCount)
	used := int64(cfg.ExpertUsedCount)
	ffn := int64(cfg.ExpertFFNSize)
	sharedFFN := int64(cfg.SharedFFNSize)
	// One selected-expert launch decodes the gate and the up matrix with the
	// same dtype instantiation; a mixed pair would need two grids per row.
	if gate.dtype != up.dtype {
		return fmt.Errorf("gpu: %s expert gate dtype %s differs from up dtype %s",
			prefix, gate.dtype, up.dtype)
	}
	if err := plan.AddRMSNorm(m.act.x, postNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.decodeActGroup(plan, m.act.norm, hidden, router.dtype,
		sharedGate.dtype, sharedUp.dtype); err != nil {
		return err
	}
	// The router is one dense row over the experts; every MoE model in the
	// supported set stores it as F32, which keeps the per-row GEMV.
	if err := m.decodeMatmul(plan, router, m.act.norm, m.act.moeLogits, hidden, experts); err != nil {
		return err
	}
	if err := plan.AddMoERoute(m.act.moeLogits, m.act.moeIDs, m.act.moeWeights,
		m.act.moeCounts, m.act.moeAssignRows, m.act.moeAssignSlots,
		m.act.moeAssignWeights, 1, experts, used); err != nil {
		return err
	}
	if err := plan.AddMoEGateUpSelected(m.act.norm, m.act.moeIDs, gate.buf, up.buf,
		m.act.moeAct, hidden, ffn, 1, used, experts, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddMoEDownSelected(m.act.moeAct, m.act.moeIDs, down.buf,
		m.act.moeOut, ffn, hidden, 1, used, experts, down.dtype); err != nil {
		return err
	}
	if err := plan.AddMoEReduceAdd(m.act.x, m.act.moeOut, m.act.moeWeights,
		hidden, 1, used); err != nil {
		return err
	}
	// The shared expert is a dense SwiGLU FFN over the same normalized row.
	if err := m.decodeMatmulSwiglu(plan, sharedGate, sharedUp, m.act.norm,
		m.act.ffnGate, hidden, sharedFFN); err != nil {
		return err
	}
	if err := m.decodeMatmul(plan, sharedDown, m.act.ffnGate, m.act.tmp,
		sharedFFN, hidden); err != nil {
		return err
	}
	return plan.AddMoESharedAdd(m.act.x, m.act.tmp, sharedGateVec.buf, m.act.norm,
		hidden, 1)
}

type prefillOutput uint8

const (
	prefillLastLogit prefillOutput = iota
	prefillAllLogits
	prefillLastLogitFullFFN // Full-row reference for correctness comparisons.
)

// buildPrefillPlan constructs the chunked prompt graph for one session.
func (m *Model) buildPrefillPlan(s *Session, rows int, outputMode prefillOutput) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	p := &m.prefill
	rows64 := int64(rows)
	lastOff := uint64(rows-1) * uint64(cfg.HiddenSize) * 4
	crop := outputMode == prefillLastLogit && cfg.BlockCount > 0 && !cfg.IsRecurrent(cfg.BlockCount-1)
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
		cropLayer := crop && i == cfg.BlockCount-1
		if cfg.IsRecurrent(i) {
			if err := m.addPrefillRecurrent(plan, s, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		} else {
			if err := m.addPrefillFullAttention(plan, s, i, attnNorm, rows64, cropLayer); err != nil {
				return nil, err
			}
		}
		ffnRows := rows64
		if cropLayer {
			ffnRows, lastOff = 1, 0
		}
		// Preserve the parent chunk's projection arithmetic for the compact row.
		if err := m.addPrefillFFN(plan, prefix, postNorm, ffnRows, rows64); err != nil {
			return nil, err
		}
	}
	if !crop {
		plan.MarkTail()
	}
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
	if outputMode == prefillAllLogits {
		if rows > maxSpecDraft {
			return nil, fmt.Errorf("gpu: speculative logits rows %d exceed %d", rows, maxSpecDraft)
		}
		if err := plan.AddRMSNormBatchAt(p.x, 0, outputNorm.buf, p.norm, 0,
			rows64, int64(cfg.HiddenSize), int64(cfg.HiddenSize), int64(cfg.HiddenSize),
			cfg.RMSNormEps); err != nil {
			return nil, err
		}
		if hip.SelectLMHeadPath(output.dtype, rows64, int64(cfg.HiddenSize), int64(cfg.VocabSize)) {
			// One streaming pass over the output matrix for every candidate
			// row instead of one GEMV per row.
			if err := plan.AddLMHeadAt(output.buf, 0, p.norm, 0, m.specLogits, 0,
				int64(cfg.HiddenSize), int64(cfg.VocabSize), rows64, output.dtype); err != nil {
				return nil, err
			}
			return plan, nil
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
	if err := plan.AddRMSNormAt(p.x, lastOff, outputNorm.buf, p.norm, 0,
		int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	if hip.SelectLMHeadPath(output.dtype, 1, int64(cfg.HiddenSize), int64(cfg.VocabSize)) {
		// The chunk tail needs the last row only. The projection reads the
		// output matrix once at streaming bandwidth instead of running the
		// decode GEMV, which is latency bound at a single row.
		if err := plan.AddLMHeadAt(output.buf, 0, p.norm, 0, m.act.logits, 0,
			int64(cfg.HiddenSize), int64(cfg.VocabSize), 1, output.dtype); err != nil {
			return nil, err
		}
		return plan, nil
	}
	if err := plan.AddGemv(output.buf, p.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (m *Model) addPrefillFullAttention(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight, rows int64, cropQuery bool) error {
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
	projections := [...]prefillProjection{
		{w: q, out: p.q, nOut: qDim},
		{w: k, out: p.k, nOut: kvDim},
		{w: v, out: p.v, nOut: kvDim},
	}
	firstProjection := 0
	if cropQuery {
		firstProjection = 1
	}
	if err := m.prefillProjections(plan, p.norm, hidden, rows, projections[firstProjection:]); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.k, 0, kNorm.buf, p.k, 0,
		rows, int64(cfg.KVHeadCount), headDim, headDim, cfg.RMSNormEps); err != nil {
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
	queryRows := rows
	var displacement uint32
	if cropQuery {
		// All K/V rows are ready for subsequent chunks and decode. The
		// remaining operations only produce the requested final output.
		plan.MarkTail()
		queryRows = 1
		displacement = uint32(rows - 1)
		lastOff := uint64(rows-1) * uint64(hidden) * 4
		if rows > 1 {
			if err := plan.AddCopyAt(p.x, 0, p.x, lastOff, hidden); err != nil {
				return err
			}
		}
		qPath := hip.SelectPrefillPath(q.dtype, rows, hidden, qDim)
		if err := m.prefillMatmulPath(plan, q, p.norm, lastOff, p.q, 0, hidden, qDim, queryRows, qPath); err != nil {
			return err
		}
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.q, 0, qNorm.buf, p.q, 0,
		queryRows, int64(cfg.HeadCount), headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchOffsetAt(p.q, 0, int(queryRows), cfg.HeadCount, int(2*headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), qDim, displacement); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	// The rocBLAS path only runs where the FP16 MMA kernel does not cover the
	// shape, so F16 KV never selects the slower kernel by default.
	useBlas := m.attentionBlasEnabled && !m.forceTiledAttention &&
		!hip.PrefillAttentionMMA(cfg.HeadCount, cfg.KVHeadCount, int(headDim),
			int(rows), s.position, m.kvType)
	if useBlas {
		if err := plan.AddAttentionPrefillPagedBlasOffsetFormatAt(p.q, 0, m.poolK[layerIndex],
			m.poolV[layerIndex], p.attn, 0, s.tableBuf, p.attentionBlas,
			cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(queryRows),
			s.capacity, m.blockSize, scale, displacement); err != nil {
			return err
		}
	} else if p.attentionSplit != nil && queryRows >= 512 {
		if err := plan.AddAttentionPrefillPagedSplitFormatAt(p.q, m.poolK[layerIndex],
			m.poolV[layerIndex], p.attn, s.tableBuf, p.attentionSplit,
			cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(queryRows),
			s.capacity, m.blockSize, scale); err != nil {
			return err
		}
	} else if err := plan.AddAttentionPrefillPagedOffsetFormatAt(p.q, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], p.attn, 0, s.tableBuf,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(queryRows),
		s.capacity, m.blockSize, scale, m.kvType, displacement, m.forceTiledAttention); err != nil {
		return err
	}
	oPath := hip.SelectPrefillPath(o.dtype, rows, oDim, hidden)
	if err := m.prefillMatmulPath(plan, o, p.attn, 0, p.tmp, 0, oDim, hidden, queryRows, oPath); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, queryRows*hidden)
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
	if err := m.prefillProjections(plan, p.norm, hidden, rows, []prefillProjection{
		{w: qkv, out: p.qkv, nOut: convDim},
		{w: gate, out: p.z, nOut: inner},
		{w: beta, out: p.beta, nOut: rank},
		{w: alpha, out: p.alpha, nOut: rank},
	}); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	var prepared *hip.Buffer
	arch, _, _ := strings.Cut(m.device.Info().GCNArch, ":")
	preparedWidth := uint64(convDim - inner + 2*rank)
	if arch == "gfx1151" && stateSize == 128 && rows >= 128 &&
		preparedWidth <= p.ffnGate.Bytes()/4/uint64(rows) {
		// The FFN writes this buffer after the recurrent op on the same stream.
		prepared = p.ffnGate
	}
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
		Prepared:    prepared,
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

func (m *Model) addPrefillFFN(plan *hip.Plan, prefix string, postNorm weight, rows, dispatchRows int64) error {
	cfg := m.config
	if cfg.IsMoE() {
		return m.addMoEPrefillFFN(plan, prefix, postNorm, rows, dispatchRows)
	}
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
	if err := m.addPrefillSwiGLU(plan, gate, up, down, p.norm, rows, dispatchRows,
		hidden, ffn); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

// addPrefillSwiGLU runs one gate/up/down feed-forward block over the activation
// tile norm and leaves the result in m.prefill.tmp. It does not touch the
// residual stream, so both the dense FFN and the MoE shared expert call it
// before adding their own contributions.
func (m *Model) addPrefillSwiGLU(plan *hip.Plan, gate, up, down weight, norm *hip.Buffer,
	rows, dispatchRows, hidden, ffn int64) error {
	p := &m.prefill
	// Fuse the gate/up pair when both matmuls select the custom WMMA path.
	// The fused kernel shares one activation tile and applies SwiGLU in the
	// epilogue, so neither intermediate is written.
	gatePath := hip.SelectPrefillPath(gate.dtype, dispatchRows, hidden, ffn)
	upPath := hip.SelectPrefillPath(up.dtype, dispatchRows, hidden, ffn)
	downPath := hip.SelectPrefillPath(down.dtype, dispatchRows, ffn, hidden)
	fused := gate.dtype == up.dtype && gatePath == hip.PrefillWMMA
	if fused {
		if err := plan.AddGemmSwigluAt(gate.buf, 0, up.buf, 0, norm, 0,
			p.ffnGate, 0, hidden, ffn, rows, gate.dtype); err != nil {
			return err
		}
	} else if gatePath == hip.PrefillMMQ && upPath == hip.PrefillMMQ {
		// The two projections read exactly the same activation tile, so the
		// Q8_1 quantization runs once for the pair.
		if err := plan.AddQuantizeQ8_1At(norm, 0, p.xQ8, p.xDS, hidden, rows); err != nil {
			return err
		}
		if err := plan.AddGemmMmqAt(gate.buf, 0, p.xQ8, p.xDS, p.ffnGate, 0, hidden, ffn, rows, gate.dtype); err != nil {
			return err
		}
		if err := plan.AddGemmMmqAt(up.buf, 0, p.xQ8, p.xDS, p.ffnUp, 0, hidden, ffn, rows, up.dtype); err != nil {
			return err
		}
		if downPath == hip.PrefillMMQ {
			// Both projections have consumed the activation tile. Reuse its
			// Q8_1 scratch for the down projection on the same stream.
			if err := plan.AddSiluMulQuantQ8_1At(p.ffnGate, 0, p.ffnUp, 0, p.xQ8, 0, p.xDS, 0, ffn, rows); err != nil {
				return err
			}
			return plan.AddGemmMmqAt(down.buf, 0, p.xQ8, p.xDS, p.tmp, 0, ffn, hidden, rows, down.dtype)
		}
	} else {
		if err := m.prefillMatmulPath(plan, gate, norm, 0, p.ffnGate, 0, hidden, ffn, rows, gatePath); err != nil {
			return err
		}
		if err := m.prefillMatmulPath(plan, up, norm, 0, p.ffnUp, 0, hidden, ffn, rows, upPath); err != nil {
			return err
		}
	}
	if !fused {
		if err := plan.AddSiluMul(p.ffnGate, p.ffnUp, p.ffnGate, rows*ffn); err != nil {
			return err
		}
	}
	return m.prefillMatmulPath(plan, down, p.ffnGate, 0, p.tmp, 0, ffn, hidden, rows, downPath)
}

// isIQExpertDType reports whether an expert matrix uses one of the IQ codebooks
// whose 256-value blocks the int8 expert route decodes to int8 digits.
func isIQExpertDType(dtype tensor.DType) bool {
	switch dtype {
	case tensor.IQ2XXS, tensor.IQ2S, tensor.IQ3S:
		return true
	default:
		return false
	}
}

// addMoEPrefillFFN builds one MoE sub-layer for a prompt chunk. The routed
// experts use the grouped kernels, which schedule one grid over all experts and
// let each expert read its binned assignment range; the shared expert reuses the
// dense prefill SwiGLU and its sigmoid gate.
func (m *Model) addMoEPrefillFFN(plan *hip.Plan, prefix string, postNorm weight, rows, dispatchRows int64) error {
	cfg := m.config
	p := &m.prefill
	router, err := m.weight(prefix + "ffn_gate_inp.weight")
	if err != nil {
		return err
	}
	sharedGateVec, err := m.weight(prefix + "ffn_gate_inp_shexp.weight")
	if err != nil {
		return err
	}
	sharedGate, err := m.weight(prefix + "ffn_gate_shexp.weight")
	if err != nil {
		return err
	}
	sharedUp, err := m.weight(prefix + "ffn_up_shexp.weight")
	if err != nil {
		return err
	}
	sharedDown, err := m.weight(prefix + "ffn_down_shexp.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "ffn_gate_exps.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up_exps.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down_exps.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	experts := int64(cfg.ExpertCount)
	used := int64(cfg.ExpertUsedCount)
	ffn := int64(cfg.ExpertFFNSize)
	sharedFFN := int64(cfg.SharedFFNSize)
	if gate.dtype != up.dtype {
		return fmt.Errorf("gpu: %s expert gate dtype %s differs from up dtype %s",
			prefix, gate.dtype, up.dtype)
	}
	if err := plan.AddRMSNormBatchAt(p.x, 0, postNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	routerPath := hip.SelectPrefillPath(router.dtype, dispatchRows, hidden, experts)
	if router.dtype == tensor.F32 {
		// The router decides which experts run, and this model is routing
		// sensitive: perturbing its router weights by 1e-4 relative moves the
		// first-token logits by 0.6, because a perturbation flips near-tie
		// expert selections. The FP16 activation staging of the WMMA tile is
		// that much noise, so an F32 router takes the FP32 batched GEMM, which
		// keeps the activations at full width. Other router dtypes keep the
		// measured prefill dispatch.
		if err := plan.AddGemmPathAt(router.buf, 0, p.norm, 0, p.moeLogits, 0,
			hidden, experts, rows, tensor.F32, hip.GemmAuto); err != nil {
			return err
		}
	} else if err := m.prefillMatmulPath(plan, router, p.norm, 0, p.moeLogits, 0,
		hidden, experts, rows, routerPath); err != nil {
		return err
	}
	if err := plan.AddMoERoute(p.moeLogits, p.moeIDs, p.moeWeights, p.moeCounts,
		p.moeAssignRows, p.moeAssignSlots, p.moeAssignWeights, rows, experts, used); err != nil {
		return err
	}
	// The int8 route needs a whole 256-value K step and a whole sixteen-
	// assignment tile. It also needs both expert matrices in an IQ codebook,
	// which is what the dp4a decoders serve; every other shape or dtype keeps
	// the FP32 grouped path.
	q8 := rows >= 16 && hidden%256 == 0 && ffn%256 == 0 &&
		gate.dtype == up.dtype && isIQExpertDType(gate.dtype) &&
		isIQExpertDType(down.dtype)
	if q8 {
		if err := plan.AddQuantizeQ8_1At(p.norm, 0, p.xQ8, p.xDS, hidden, rows); err != nil {
			return err
		}
		if err := plan.AddMoEGateUpQ8(p.xQ8, p.xDS, gate.buf, up.buf, p.moeCounts,
			p.moeAssignRows, p.moeAssignSlots, p.moeGate, p.moeUp, hidden, ffn,
			rows, experts, used, gate.dtype); err != nil {
			return err
		}
		if err := plan.AddSiluMulQuantQ8_1At(p.moeGate, 0, p.moeUp, 0, p.moeQ8, 0,
			p.moeDS, 0, ffn, rows*used); err != nil {
			return err
		}
		if err := plan.AddMoEDownQ8(p.moeQ8, p.moeDS, down.buf, p.moeCounts,
			p.moeAssignRows, p.moeAssignSlots, p.moeOut, ffn, hidden, rows,
			experts, used, down.dtype); err != nil {
			return err
		}
	} else {
		if err := plan.AddMoEGateUpGrouped(p.norm, gate.buf, up.buf, p.moeCounts,
			p.moeAssignRows, p.moeAssignSlots, p.moeGate, p.moeUp, hidden, ffn, rows,
			experts, used, gate.dtype); err != nil {
			return err
		}
		if err := plan.AddSiluMul(p.moeGate, p.moeUp, p.moeGate, rows*used*ffn); err != nil {
			return err
		}
		if err := plan.AddMoEDownGrouped(p.moeGate, down.buf, p.moeCounts,
			p.moeAssignRows, p.moeAssignSlots, p.moeOut, ffn, hidden, rows, experts,
			used, down.dtype); err != nil {
			return err
		}
	}
	if err := plan.AddMoEReduceAdd(p.x, p.moeOut, p.moeWeights, hidden, rows, used); err != nil {
		return err
	}
	// The shared expert reads the normalized tile, which none of the routed
	// expert operations write.
	if err := m.addPrefillSwiGLU(plan, sharedGate, sharedUp, sharedDown, p.norm,
		rows, dispatchRows, hidden, sharedFFN); err != nil {
		return err
	}
	return plan.AddMoESharedAdd(p.x, p.tmp, sharedGateVec.buf, p.norm, hidden, rows)
}
