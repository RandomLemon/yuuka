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
	// PrefillBLAS is the rocBLAS FP16 baseline.
	PrefillBLAS
)

// SelectPrefillPath returns the measured prefill matmul choice for a dtype,
// row count, and shape. Measurements are in docs/benchmarks.md. Batched GEMV
// is selected below four rows and for the tall-K projections at the smallest
// WMMA batch; at one row the sweep shows per-row submission is faster for F16,
// Q8_0, and Q6_K, which remains a small-row follow-up. Above those points the
// custom WMMA GEMM beats per-row submission; the rocBLAS FP16 baseline wins for
// F16 only in the measured shape classes below.
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
		// Wide F16 projections keep batched GEMV through M=4: WMMA is 17%
		// slower at 2560->9216 and only within noise at 8192->2560.
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
	return PrefillWMMA
}
