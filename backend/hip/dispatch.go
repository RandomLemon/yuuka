//go:build hip && linux

package hip

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/tensor"
)

// GemvPath is the selected decode matrix-vector submission shape.
type GemvPath int

const (
	// GemvPerRow submits one GEMV per batch row.
	GemvPerRow GemvPath = iota
	// GemvBatched reuses each packed weight element across the batch.
	GemvBatched
)

// SelectGemvPath returns the measured dispatch choice for a dtype and batch.
// Thresholds come from bench/hip/gemv_sweep.cpp on gfx1151: K-quants switch at
// batch 2, while Q8_0 and F16 need batch 4 to offset the extra reduction work.
// Batches above 16 fall back to per-row submission.
func SelectGemvPath(dtype tensor.DType, batch int) GemvPath {
	if batch <= 0 || batch > 16 {
		return GemvPerRow
	}
	threshold := 4
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		threshold = 2
	}
	if batch >= threshold {
		return GemvBatched
	}
	return GemvPerRow
}

// AddGemvAuto appends the measured dispatch choice for a batch of input rows.
// batch must be positive. Batches above 16 fall back to one GEMV per row.
func (p *Plan) AddGemvAuto(w, x, out *Buffer, nIn, nOut, batch int64, dtype tensor.DType) error {
	if batch <= 0 {
		return fmt.Errorf("hip: GEMV batch %d must be positive", batch)
	}
	if SelectGemvPath(dtype, int(batch)) == GemvBatched {
		return p.AddGemvBatch(w, x, out, nIn, nOut, batch, dtype)
	}
	for row := int64(0); row < batch; row++ {
		if err := p.AddGemvAt(w, 0,
			x, uint64(row*nIn*4), out, uint64(row*nOut*4),
			nIn, nOut, dtype); err != nil {
			return err
		}
	}
	return nil
}

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
// wins below four rows and for the tall-K projections at the smallest WMMA
// batch. Above those points the custom WMMA GEMM beats per-row submission; the
// rocBLAS FP16 baseline wins for F16 only in the measured shape classes below.
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
