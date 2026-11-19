//go:build hip && linux

package hip

/*
#include "yuuka_hip.h"
*/
import "C"

// Op codes from the yuuka_op_code enum. They are re-declared so non-cgo Go code
// and tests can name an operation instead of using a number.
const (
	OpGemv                  uint32 = uint32(C.YUUKA_OP_GEMV)
	OpRMSNorm               uint32 = uint32(C.YUUKA_OP_RMSNORM)
	OpL2Norm                uint32 = uint32(C.YUUKA_OP_L2NORM)
	OpGatedRMSNorm          uint32 = uint32(C.YUUKA_OP_GATED_RMSNORM)
	OpRoPE                  uint32 = uint32(C.YUUKA_OP_ROPE)
	OpAttention             uint32 = uint32(C.YUUKA_OP_ATTENTION)
	OpConv1D                uint32 = uint32(C.YUUKA_OP_CONV1D)
	OpGDN                   uint32 = uint32(C.YUUKA_OP_GDN)
	OpSiluMul               uint32 = uint32(C.YUUKA_OP_SILU_MUL)
	OpAdd                   uint32 = uint32(C.YUUKA_OP_ADD)
	OpCopy                  uint32 = uint32(C.YUUKA_OP_COPY)
	OpScale                 uint32 = uint32(C.YUUKA_OP_SCALE)
	OpEmbedding             uint32 = uint32(C.YUUKA_OP_EMBEDDING)
	OpKVWrite               uint32 = uint32(C.YUUKA_OP_KV_WRITE)
	OpRMSNormHeads          uint32 = uint32(C.YUUKA_OP_RMSNORM_HEADS)
	OpGemm                  uint32 = uint32(C.YUUKA_OP_GEMM)
	OpGemvBatch             uint32 = uint32(C.YUUKA_OP_GEMV_BATCH)
	OpRMSNormBatch          uint32 = uint32(C.YUUKA_OP_RMSNORM_BATCH)
	OpRMSNormHeadsBatch     uint32 = uint32(C.YUUKA_OP_RMSNORM_HEADS_BATCH)
	OpRoPEBatch             uint32 = uint32(C.YUUKA_OP_ROPE_BATCH)
	OpKVWriteBatch          uint32 = uint32(C.YUUKA_OP_KV_WRITE_BATCH)
	OpAttentionPrefill      uint32 = uint32(C.YUUKA_OP_ATTENTION_PREFILL)
	OpRecurrentChunk        uint32 = uint32(C.YUUKA_OP_RECURRENT_CHUNK)
	OpEmbeddingBatch        uint32 = uint32(C.YUUKA_OP_EMBEDDING_BATCH)
	OpCastF16               uint32 = uint32(C.YUUKA_OP_CAST_F16)
	OpKVWritePaged          uint32 = uint32(C.YUUKA_OP_KV_WRITE_PAGED)
	OpAttentionPaged        uint32 = uint32(C.YUUKA_OP_ATTENTION_PAGED)
	OpKVWriteBatchPaged     uint32 = uint32(C.YUUKA_OP_KV_WRITE_BATCH_PAGED)
	OpAttentionPrefillPaged uint32 = uint32(C.YUUKA_OP_ATTENTION_PREFILL_PAGED)
	OpMoERoute              uint32 = uint32(C.YUUKA_OP_MOE_ROUTE)
	OpMoEGateUp             uint32 = uint32(C.YUUKA_OP_MOE_GATE_UP)
	OpMoEDown               uint32 = uint32(C.YUUKA_OP_MOE_DOWN)
	OpMoEReduceAdd          uint32 = uint32(C.YUUKA_OP_MOE_REDUCE_ADD)
	OpGemmSwiglu            uint32 = uint32(C.YUUKA_OP_GEMM_SWIGLU)
	OpDequantF16            uint32 = uint32(C.YUUKA_OP_DEQUANT_F16)
	OpQuantQ8_1             uint32 = uint32(C.YUUKA_OP_QUANT_Q8_1)
	OpLMHead                uint32 = uint32(C.YUUKA_OP_LM_HEAD)
	OpGemvQ8_1              uint32 = uint32(C.YUUKA_OP_GEMV_Q8_1)
	OpRecurrentToken        uint32 = uint32(C.YUUKA_OP_RECURRENT_TOKEN)
	OpConvL2Norm            uint32 = uint32(C.YUUKA_OP_CONV_L2NORM)
	OpGemvResidual          uint32 = uint32(C.YUUKA_OP_GEMV_RESIDUAL)
	OpGemvPair              uint32 = uint32(C.YUUKA_OP_GEMV_PAIR)
	OpSiluMulQuantQ8_1      uint32 = uint32(C.YUUKA_OP_SILU_MUL_QUANT_Q8_1)
	OpMoEGateUpSelected     uint32 = uint32(C.YUUKA_OP_MOE_GATE_UP_SELECTED)
	OpMoEDownSelected       uint32 = uint32(C.YUUKA_OP_MOE_DOWN_SELECTED)
	OpMoESharedAdd          uint32 = uint32(C.YUUKA_OP_MOE_SHARED_ADD)
)

// opFlagTail mirrors YUUKA_OP_FLAG_TAIL.
const opFlagTail = 0x1

// MaxOpCode is the highest op code the bound ABI declares. The shadow covers
// every code in 1..MaxOpCode, and opinfo_test.go fails when one of them has no
// name or role.
const MaxOpCode = OpMoESharedAdd

// OpRole names what a plan operation holds in one of its buffer slots. The role
// is a property of the ABI opcode, so it is recorded once per code in opRoles
// instead of at every Add* call site.
type OpRole string

const (
	// RoleUnused marks a slot the operation does not read or write.
	RoleUnused OpRole = ""
	// RoleWeight is a packed matrix or a per-channel scale the kernel reads.
	RoleWeight OpRole = "weight"
	// RoleActivation is an FP32 input the kernel reads without writing it.
	RoleActivation OpRole = "activation"
	// RoleAccumulator is an FP32 input the kernel reads and writes in place.
	RoleAccumulator OpRole = "accumulator"
	// RoleOutput is the destination the kernel writes.
	RoleOutput OpRole = "output"
	// RoleState is persistent per-request state such as the gated delta-net
	// matrix or the conv1d history.
	RoleState OpRole = "state"
	// RolePool is paged KV pool storage.
	RolePool OpRole = "pool"
	// RoleTable is an index list: a block table or a token id buffer.
	RoleTable OpRole = "table"
	// RoleScratch is a staging buffer whose contents are not an output.
	RoleScratch OpRole = "scratch"
	// RoleControl is routing data such as per-expert counts or slot lists.
	RoleControl OpRole = "control"
)

// OpInfo is the Go-side shadow of one planned device operation. The plan keeps
// it next to the opaque C plan so diagnostic tooling can name an operation and
// the buffers it touches without a C ABI change. It carries the fields the
// attribution needs, not a copy of the whole yuuka_op.
type OpInfo struct {
	// Index is the submission index of the operation in the plan.
	Index int
	// Code is the yuuka_op_code of the operation.
	Code uint32
	// Flags is the yuuka_op flags word, including YUUKA_OP_FLAG_TAIL.
	Flags uint32
	// N is the dimension block the ABI passes to the kernel.
	N [6]int64
	// U is the selector block: dtype, GEMM path, capacity, or block size
	// depending on the operation.
	U [4]uint32
	// Bufs identifies buf[0..2] with Buffer.ID so a caller can map a slot back
	// to the tensor it reads. Zero means the slot is unused. The values are
	// diagnostics-only identities and must not be dereferenced.
	Bufs [3]uintptr
	// Roles names the semantics of buf[0..2].
	Roles [3]OpRole
}

// Name returns the ABI name of the operation code.
func (o OpInfo) Name() string { return OpName(o.Code) }

// Tail reports whether the operation belongs to the output tail, which an
// Execute call can skip.
func (o OpInfo) Tail() bool { return o.Flags&opFlagTail != 0 }

// OpName returns the short ABI name of an op code. The names match the columns
// of the attribution tables in docs/benchmarks.md.
func OpName(code uint32) string {
	switch code {
	case OpGemv:
		return "gemv"
	case OpRMSNorm:
		return "rmsnorm"
	case OpL2Norm:
		return "l2norm"
	case OpGatedRMSNorm:
		return "gated_rmsnorm"
	case OpRoPE:
		return "rope"
	case OpAttention:
		return "attention"
	case OpConv1D:
		return "conv1d"
	case OpGDN:
		return "gdn"
	case OpSiluMul:
		return "silu_mul"
	case OpAdd:
		return "add"
	case OpCopy:
		return "copy"
	case OpScale:
		return "scale"
	case OpEmbedding:
		return "embedding"
	case OpKVWrite:
		return "kv_write"
	case OpRMSNormHeads:
		return "rmsnorm_heads"
	case OpGemm:
		return "gemm"
	case OpGemvBatch:
		return "gemv_batch"
	case OpRMSNormBatch:
		return "rmsnorm_batch"
	case OpRMSNormHeadsBatch:
		return "rmsnorm_heads_batch"
	case OpRoPEBatch:
		return "rope_batch"
	case OpKVWriteBatch:
		return "kv_write_batch"
	case OpAttentionPrefill:
		return "attention_prefill"
	case OpRecurrentChunk:
		return "recurrent_chunk"
	case OpEmbeddingBatch:
		return "embedding_batch"
	case OpCastF16:
		return "cast_f16"
	case OpKVWritePaged:
		return "kv_write_paged"
	case OpAttentionPaged:
		return "attention_paged"
	case OpKVWriteBatchPaged:
		return "kv_write_batch_paged"
	case OpAttentionPrefillPaged:
		return "attention_prefill_paged"
	case OpMoERoute:
		return "moe_route"
	case OpMoEGateUp:
		return "moe_gate_up"
	case OpMoEDown:
		return "moe_down"
	case OpMoEReduceAdd:
		return "moe_reduce_add"
	case OpGemmSwiglu:
		return "gemm_swiglu"
	case OpDequantF16:
		return "dequant_f16"
	case OpQuantQ8_1:
		return "quant_q8_1"
	case OpLMHead:
		return "lm_head"
	case OpGemvQ8_1:
		return "gemv_q8_1"
	case OpRecurrentToken:
		return "recurrent_token"
	case OpConvL2Norm:
		return "conv_l2norm"
	case OpGemvResidual:
		return "gemv_residual"
	case OpGemvPair:
		return "gemv_pair"
	case OpSiluMulQuantQ8_1:
		return "silu_mul_quant_q8_1"
	case OpMoEGateUpSelected:
		return "moe_gate_up_selected"
	case OpMoEDownSelected:
		return "moe_down_selected"
	case OpMoESharedAdd:
		return "moe_shared_add"
	default:
		return "unknown"
	}
}

// opRoles returns the buffer roles of an op code. The order follows the buffer
// indices each launcher reads in backend/hip/bridge/bridge.cpp.
func opRoles(code uint32) [3]OpRole {
	switch code {
	case OpGemv, OpGemvBatch, OpLMHead, OpGemvResidual:
		return [3]OpRole{RoleWeight, RoleActivation, RoleOutput}
	case OpGemvPair:
		// Two weight matrices and one activation row; the outputs are buf[3]
		// and buf[4].
		return [3]OpRole{RoleWeight, RoleWeight, RoleActivation}
	case OpGemvQ8_1:
		// Weights, Q8_1 digits, and their block scales; the output is buf[3].
		return [3]OpRole{RoleWeight, RoleActivation, RoleScratch}
	case OpRMSNorm, OpRMSNormHeads, OpRMSNormBatch, OpRMSNormHeadsBatch:
		return [3]OpRole{RoleActivation, RoleWeight, RoleOutput}
	case OpL2Norm:
		return [3]OpRole{RoleActivation, RoleUnused, RoleOutput}
	case OpGatedRMSNorm:
		return [3]OpRole{RoleActivation, RoleWeight, RoleActivation}
	case OpRoPE, OpRoPEBatch, OpScale:
		return [3]OpRole{RoleAccumulator, RoleUnused, RoleUnused}
	case OpAttention:
		// q, k, v, out, scores: the first three slots are all activation rows
		// and the scores buffer is scratch.
		return [3]OpRole{RoleActivation, RoleActivation, RoleActivation}
	case OpConv1D:
		return [3]OpRole{RoleState, RoleWeight, RoleActivation}
	case OpGDN:
		// state, q, k, v, alpha, beta, dt, a, out.
		return [3]OpRole{RoleState, RoleActivation, RoleActivation}
	case OpRecurrentToken:
		// state, q, k, v, alpha, beta, dt, a, out, norm weight, gate.
		return [3]OpRole{RoleState, RoleActivation, RoleActivation}
	case OpConvL2Norm:
		// conv state, conv weights, qkv, out.
		return [3]OpRole{RoleState, RoleWeight, RoleActivation}
	case OpSiluMul, OpSiluMulQuantQ8_1:
		return [3]OpRole{RoleActivation, RoleActivation, RoleOutput}
	case OpAdd:
		return [3]OpRole{RoleAccumulator, RoleActivation, RoleUnused}
	case OpCopy:
		return [3]OpRole{RoleOutput, RoleActivation, RoleUnused}
	case OpEmbedding:
		return [3]OpRole{RoleWeight, RoleOutput, RoleUnused}
	case OpKVWrite, OpKVWriteBatch:
		// k, v, k cache, v cache.
		return [3]OpRole{RoleActivation, RoleActivation, RolePool}
	case OpGemm:
		// weights, x, out, fp16 scratch.
		return [3]OpRole{RoleWeight, RoleActivation, RoleOutput}
	case OpAttentionPrefill:
		return [3]OpRole{RoleActivation, RolePool, RolePool}
	case OpRecurrentChunk:
		// The exact slot list is in AddRecurrentChunkAt: state in, state out,
		// then the per-token inputs.
		return [3]OpRole{RoleState, RoleState, RoleActivation}
	case OpEmbeddingBatch:
		return [3]OpRole{RoleWeight, RoleOutput, RoleTable}
	case OpCastF16, OpDequantF16:
		return [3]OpRole{RoleActivation, RoleOutput, RoleUnused}
	case OpKVWritePaged, OpKVWriteBatchPaged:
		// k, v, k pool, v pool, table.
		return [3]OpRole{RoleActivation, RoleActivation, RolePool}
	case OpAttentionPaged, OpAttentionPrefillPaged:
		// q, k pool, v pool, out, scores, table.
		return [3]OpRole{RoleActivation, RolePool, RolePool}
	case OpMoERoute, OpMoEGateUp:
		// activation first, then the expert weights.
		return [3]OpRole{RoleActivation, RoleWeight, RoleWeight}
	case OpMoEDown:
		// activation, weights, counts, assignment, out.
		return [3]OpRole{RoleActivation, RoleWeight, RoleControl}
	case OpMoEReduceAdd:
		return [3]OpRole{RoleAccumulator, RoleOutput, RoleWeight}
	case OpMoEGateUpSelected:
		// norm, ids, gate weights; the id list is routing control data.
		return [3]OpRole{RoleActivation, RoleControl, RoleWeight}
	case OpMoEDownSelected:
		// act, ids, down weights.
		return [3]OpRole{RoleActivation, RoleControl, RoleWeight}
	case OpMoESharedAdd:
		// x, shared-expert output, gate vector.
		return [3]OpRole{RoleAccumulator, RoleActivation, RoleWeight}
	case OpGemmSwiglu:
		// gate weights, up weights, x, out.
		return [3]OpRole{RoleWeight, RoleWeight, RoleActivation}
	case OpQuantQ8_1:
		return [3]OpRole{RoleActivation, RoleOutput, RoleOutput}
	default:
		return [3]OpRole{}
	}
}
