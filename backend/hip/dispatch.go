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
