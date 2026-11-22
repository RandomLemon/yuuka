//go:build hip && linux

package hip

import (
	"github.com/yuuka-ai/yuuka/tensor"
)

// PrefillPath is the selected prefill matmul implementation.
type PrefillPath int

const (
	// PrefillBatchedGemv reuses each weight element across a small batch.
	PrefillBatchedGemv PrefillPath = iota
	// PrefillWMMA is the custom rocWMMA GEMM with FP32 activations.
	PrefillWMMA
	// PrefillBLAS is the rocBLAS route for F16 weights.
	PrefillBLAS
	// PrefillMMQ is the int8 MMQ route: Q8_1 activations, int8-decoded
	// K-quant weights, exact int32 accumulation, and FP32 weight scales.
	PrefillMMQ
)

// String returns the diagnostic name of a prefill path.
func (p PrefillPath) String() string {
	switch p {
	case PrefillBatchedGemv:
		return "batched-gemv"
	case PrefillWMMA:
		return "wmma"
	case PrefillBLAS:
		return "rocblas"
	case PrefillMMQ:
		return "mmq"
	default:
		return "unknown"
	}
}

// SelectPrefillPath returns the measured prefill matmul choice for a dtype,
// row count, and shape. Measurements are in docs/benchmarks.md. Batched GEMV
// is selected below four rows and for the tall-K projections at the smallest
// WMMA batch; at one row the sweep shows per-row submission is faster for F16,
// Q8_0, and Q6_K, which remains a small-row follow-up. Above those points the
// custom WMMA GEMM beats per-row submission; the rocBLAS FP16 baseline wins for
// F16 only in the measured shape classes below.
//
// K-quant weights use the int8 MMQ route at 128 or more rows and a multiple-of
// 256 input width, which is every K-quant block width. The route keeps the
// weights packed, quantizes the activations to Q8_1, accumulates exact int32
// products in the 16x16x16 int8 WMMA, and applies the FP32 weight scales in the
// epilogue, so it has none of the FP16-destination reduction error that keeps
// rocBLAS off the tall projections. The measured sweep puts it at 10.4-14.8
// TF/s against 6.3-9.7 TF/s for WMMA across the 128-512 row classes, and 7-60%
// above the dequant-plus-rocBLAS route at 128-256 rows; at 512 rows that route
// is 2-17% faster on wide shapes, which the blanket rule intentionally trades
// for halving the tall-projection time and dropping the FP16 weight scratch.
// Below 128 rows the Q8_1 pass and the 128-token tile waste more work than they
// save, so WMMA stays.
func SelectPrefillPath(dtype tensor.DType, rows, nIn, nOut int64) PrefillPath {
	if rows <= 0 {
		return PrefillBatchedGemv
	}
	if dtype == tensor.F16 {
		if rows < 4 {
			return PrefillBatchedGemv
		}
		if nIn > nOut {
			if rows <= 64 {
				return PrefillBLAS
			}
			return PrefillWMMA
		}
		// Wide F16 projections keep batched GEMV below M=8: WMMA is 17%
		// slower at 2560->9216 and only within noise at 2560->8192.
		if rows < 8 {
			return PrefillBatchedGemv
		}
		if rows >= 128 {
			return PrefillBLAS
		}
		return PrefillWMMA
	}
	if rows < 4 {
		return PrefillBatchedGemv
	}
	if nIn > nOut && rows < 8 {
		return PrefillBatchedGemv
	}
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		if nIn%256 == 0 && nOut >= 64 && rows >= 128 {
			return PrefillMMQ
		}
	}
	return PrefillWMMA
}

// MoEPrefillPath is the selected routed-expert prefill implementation.
type MoEPrefillPath int

const (
	// MoEPrefillF32Grouped is the grouped kernel over FP32 activations. It
	// serves every dtype the MoE kernels decode but pays the scalar decode and
	// dot per weight value, so it is the fallback for shapes the quantized
	// routes do not cover.
	MoEPrefillF32Grouped MoEPrefillPath = iota
	// MoEPrefillQ8 is the grouped Q8_1 dp4a route. It serves the IQ codebooks
	// and the shapes whose K or output width does not fit the WMMA tile.
	MoEPrefillQ8
	// MoEPrefillMMQ is the grouped int8 WMMA route: the same Q8_1 activation
	// tile dotted by 16x16x16 int8 WMMA over an assignment tile of TileJ rows,
	// so one decoded weight tile serves the whole assignment tile. It decodes
	// the K-quants and the IQ2_XXS/IQ2_S/IQ3_S codebooks.
	MoEPrefillMMQ
)

// String returns the diagnostic name of a MoE prefill path.
func (p MoEPrefillPath) String() string {
	switch p {
	case MoEPrefillF32Grouped:
		return "fp32-grouped"
	case MoEPrefillQ8:
		return "q8-dp4a"
	case MoEPrefillMMQ:
		return "mmq"
	default:
		return "unknown"
	}
}

// MoEMMQMaxTileJ is the widest measured assignment tile of the int8 WMMA
// expert route.
const MoEMMQMaxTileJ = 128

// SelectMoEPrefillPath returns the routed-expert prefill route and, for the
// int8 WMMA route, its assignment tile width.
//
// The int8 WMMA route serves the K-quant and the IQ expert dtypes when every
// weight row is a whole number of 256-value blocks and the output width fills
// the 128-channel tile; its assignment tile width comes from the measured table
// in moeAssignmentTile. Both quantized routes require at least
// MoEPrefillMinRows rows: below that the Q8_1 activation pass and the grouped
// tile cost more than they save, which is also why the one-row FFN of the last
// prefill chunk stays on the FP32 grouped route. Everything else falls back to
// the Q8_1 dp4a route for the IQ codebooks and to the FP32 grouped route
// otherwise. These are plan-build-time decisions; the measured stage times that
// motivate them are in docs/benchmarks.md and
// docs/benchmarks/qwen3.6_35b_a3b.md.
//
// forceQ8 keeps the Q8_1 dp4a route for the routed experts (Options.ForceMoEQ8).
// It exists so the two routes can be compared on identical plan inputs and does
// not change what a production session runs.
func SelectMoEPrefillPath(dtype tensor.DType, rows, used, experts, nIn, nOut int64, forceQ8 bool) (MoEPrefillPath, int64) {
	if rows <= 0 || used <= 0 || experts <= 0 || nIn <= 0 || nOut <= 0 {
		return MoEPrefillF32Grouped, 0
	}
	if rows < MoEPrefillMinRows {
		return MoEPrefillF32Grouped, 0
	}
	if !forceQ8 && MoEMMQDType(dtype) && nIn%256 == 0 && nOut%128 == 0 {
		return MoEPrefillMMQ, moeAssignmentTile(rows)
	}
	switch dtype {
	case tensor.IQ2XXS, tensor.IQ2S, tensor.IQ3S:
		if nIn%256 == 0 && nOut%256 == 0 {
			return MoEPrefillQ8, 0
		}
	}
	return MoEPrefillF32Grouped, 0
}

// MoEMMQDType reports whether the int8 WMMA expert route decodes a weight
// dtype: the K-quants through the shared MMQ tile staging and the IQ codebooks
// through its IQ span decoder. The shape rules are separate; see
// SelectMoEPrefillPath.
func MoEMMQDType(dtype tensor.DType) bool {
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K,
		tensor.IQ2XXS, tensor.IQ2S, tensor.IQ3S:
		return true
	default:
		return false
	}
}

// MoEPrefillMinRows is the smallest chunk either quantized expert route serves.
// Below it the 16-row assignment tile and the Q8_1 activation pass waste more
// work than they save, and the FP32 grouped route is the measured choice.
const MoEPrefillMinRows = 16

// moeTileTable is the measured assignment-tile table of the int8 WMMA expert
// route: a chunk of at least minRows rows stages tileJ assignments per weight
// pass. Each entry is the smallest width that won its row class in the sweep of
// docs/benchmarks/qwen3.6_35b_a3b.md at the qwen35moe shape (256 experts, top-8,
// hidden 2048, expert FFN 512, IQ2_XXS gate/up): 16 through 384 rows, 32 from
// 512 to 2048, and 64 from 4096. A wider tile than the bucket mean needs costs
// one weight pass per expert, which is cheap; a tile narrower than the bucket
// costs one more pass, which is not, so the table sits at the measured size
// rather than at the bucket mean.
var moeTileTable = [...]struct {
	minRows int64
	tileJ   int64
}{
	{1, 16},
	{512, 32},
	{4096, 64},
}

// moeAssignmentTile returns the measured assignment tile for a chunk of rows.
// The table is measured at the shape the benchmark record covers; a model whose
// bucket mean differs by more than 2x keeps the smallest measured width, which
// costs extra passes, never correctness.
func moeAssignmentTile(rows int64) int64 {
	tileJ := moeTileTable[0].tileJ
	for _, entry := range moeTileTable {
		if rows >= entry.minRows {
			tileJ = entry.tileJ
		}
	}
	return tileJ
}

// SelectLMHeadPath reports whether the streaming LM head kernel is selected for
// an output projection. The kernel serves 1..16 rows and streams every packed
// weight row once, which is what makes both output tails bandwidth bound
// instead of latency bound; it is measured against the per-row GEMV in
// docs/benchmarks.md. The prefill tail and the decode tail use it directly, and
// SelectDecodeGemvPath uses it for the whole decode matvec family; any shape it
// does not cover keeps the per-row GEMV, which stays the documented fallback
// rather than a silent substitution.
func SelectLMHeadPath(dtype tensor.DType, rows, nIn, nOut int64) bool {
	if rows <= 0 || rows > MaxLMHeadRows {
		return false
	}
	if nIn <= 0 || nOut < lmHeadMinOutputsFor(dtype) || nIn%256 != 0 {
		return false
	}
	switch dtype {
	case tensor.F32, tensor.F16, tensor.Q4_0, tensor.Q5_0, tensor.Q8_0,
		tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		return true
	default:
		return false
	}
}

// lmHeadMinOutputsFor is the measured crossover below which the per-row GEMV is
// still the better route. The threshold is shared by the one-row decode matvec
// and the 1-16-row prefill tail, so it uses the value measured for the
// multi-row channel-parallel form: at 2560 input values that form wins from 512
// outputs upward for Q4_K, Q5_K, and Q6_K and from 1536 upward for Q8_0 and F16.
// The 1024-output cell of Q8_0 and F16 is level with the per-row form and 1536
// is the measured win, so they keep the higher threshold. After the one-row
// remap (gemv_row_kernel) the decode sweep measures the streaming route ahead
// from 256 outputs for every dtype, but the shared predicate cannot use that
// without a multi-row measurement for the same widths. Sweeps:
// build/yuuka_prefill_sweep (LMHEAD cells) and
// build/yuuka_gemv_decode_sweep --threshold.
func lmHeadMinOutputsFor(dtype tensor.DType) int64 {
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		return 512
	default:
		return 1024
	}
}

// DecodeGemvPath is the selected decode matvec implementation.
type DecodeGemvPath int

const (
	// DecodeGemvPerRow is one 256-thread block per output row with a block-wide
	// tree reduction and, for Q4_K/Q5_K, two barriers per 256-value K step. It
	// is the fallback for shapes the streaming kernel does not cover and the
	// reference the kernel test compares against.
	DecodeGemvPerRow DecodeGemvPath = iota
	// DecodeGemvStreaming is the streaming kernel family. A one-row block with
	// a 512-value K step takes the remapped form (gemv_row_kernel): one wave per
	// output row, sixteen K values per lane, no shared-memory staging and no
	// barrier inside K. Other row counts and K widths take the channel-parallel
	// form (lm_head_kernel): one wave owns four output channels over a
	// 256-value staged activation step. Both reduce with one wave shuffle per
	// output and keep the packed weights quantized.
	DecodeGemvStreaming
	// DecodeGemvInt8 is the same mapping over an int8 activation: the four
	// packed digits of a lane are dotted against the Q8_1 activation word with
	// one v_dot4 instead of being decoded to FP32 first, and the block scale
	// and minimum are applied once per sub-block.
	DecodeGemvInt8
)

// IsInt8DecodeDType reports whether the int8 decode matvec serves a weight
// dtype. The route decodes the packed digits straight to int8, so it exists
// only for the dtypes whose per-value form is an integer plus an affine block
// scale: the K-quants and Q8_0. F32, F16, Q4_0, and Q5_0 keep the FP32
// streaming kernel.
func IsInt8DecodeDType(dtype tensor.DType) bool {
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K, tensor.Q8_0:
		return true
	default:
		return false
	}
}

// String returns the diagnostic name of a decode matvec path.
func (p DecodeGemvPath) String() string {
	switch p {
	case DecodeGemvPerRow:
		return "per-row"
	case DecodeGemvStreaming:
		return "streaming"
	case DecodeGemvInt8:
		return "int8"
	default:
		return "unknown"
	}
}

// SelectDecodeGemvPath returns the measured decode matvec choice for a weight
// dtype and shape. It shares the output projection's shape predicate: the
// streaming mapping is bandwidth bound on every shape the decode graph
// dispatches (measured 93-230 GB/s in step against 25-76 GB/s for the per-row
// form), and the per-row GEMV stays below the crossover and for the shapes it
// cannot serve (a K step that is not 512 values, an unsupported dtype, or more
// than one row). Sharing one predicate keeps the two tails and the layer
// matvecs from drifting apart; the crossover measurement is in
// docs/benchmarks.md.
//
// The FP32 streaming kernel is the selected route for every dtype. The int8
// route is implemented and reachable through int8Matvec, and it is measured
// slower end to end on gfx1151: the decode step reads one row of weights per
// matvec and is not instruction bound, so the ALU saving does not pay for the
// activation quantization the route needs (docs/benchmarks.md).
// It stays available so that measurement is reproducible and so a device or a
// later fusion can be re-evaluated against it.
//
// The narrow recurrent projections (ssm_alpha and ssm_beta, 32 outputs) stay on
// the per-row form because the streaming kernel needs 32 blocks to fill the
// device: at 32 outputs it runs one block and measures 3.7 GB/s against 9.5.
// Merging those launches is a launch-count change that is not implemented.
func SelectDecodeGemvPath(dtype tensor.DType, nIn, nOut int64, int8Matvec bool) DecodeGemvPath {
	if !SelectLMHeadPath(dtype, 1, nIn, nOut) {
		return DecodeGemvPerRow
	}
	if int8Matvec && IsInt8DecodeDType(dtype) {
		return DecodeGemvInt8
	}
	return DecodeGemvStreaming
}

// PrefillAttentionMMA reports whether the FP16 MMA paged prefill attention
// kernel is selected for a shape. It mirrors the C++ rule in
// kernels/hip/prefill/attention_wmma.hip: the kernel serves the four query
// heads that share a KV head, needs an FP16-storable pool and a head width of
// 64, 128, or 256, and only wins once the chunk or the prefix is long enough to
// fill the device. A shape it rejects keeps the tiled kernel, which is correct
// for every format.
func PrefillAttentionMMA(heads, kvHeads, headDim, rows, position int, dtype tensor.DType) bool {
	if heads != 4*kvHeads || headDim <= 0 || rows <= 0 {
		return false
	}
	if dtype != tensor.F32 && dtype != tensor.F16 {
		return false
	}
	if headDim != 64 && headDim != 128 && headDim != 256 {
		return false
	}
	return rows >= 64 || position+rows >= 256
}
