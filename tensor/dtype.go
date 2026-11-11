// Package tensor defines tensor dtypes, GGML block layouts, and reference
// numeric kernels used by the CPU backend.
package tensor

import (
	"errors"
	"fmt"
)

// DType is the on-disk GGML tensor type. Values match the GGUF ggml_type enum.
type DType uint32

const (
	F32   DType = 0
	F16   DType = 1
	Q4_0  DType = 2
	Q4_1  DType = 3
	Q5_0  DType = 6
	Q5_1  DType = 7
	Q8_0  DType = 8
	Q8_1  DType = 9
	Q2_K  DType = 10
	Q3_K  DType = 11
	Q4_K  DType = 12
	Q5_K  DType = 13
	Q6_K  DType = 14
	Q8_K  DType = 15
	IQ2XS DType = 16
)

// ErrUnsupportedDType is returned when a dtype has no reader or kernel in the
// current build. Callers must surface it as a capability failure.
var ErrUnsupportedDType = errors.New("unsupported tensor dtype")

// String returns the GGML dtype name.
func (d DType) String() string {
	switch d {
	case F32:
		return "F32"
	case F16:
		return "F16"
	case Q4_0:
		return "Q4_0"
	case Q4_1:
		return "Q4_1"
	case Q5_0:
		return "Q5_0"
	case Q5_1:
		return "Q5_1"
	case Q8_0:
		return "Q8_0"
	case Q8_1:
		return "Q8_1"
	case Q2_K:
		return "Q2_K"
	case Q3_K:
		return "Q3_K"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case Q6_K:
		return "Q6_K"
	case Q8_K:
		return "Q8_K"
	case IQ2XS:
		return "IQ2_XXS"
	default:
		return fmt.Sprintf("DType(%d)", uint32(d))
	}
}

// Block describes the fixed element and byte size of one GGML storage block.
type Block struct {
	Elements int64
	Bytes    int64
}

// BlockLayout returns the known GGML storage block for a dtype. A known layout
// does not imply a decoding kernel; use Supported for execution support.
func (d DType) BlockLayout() (Block, error) {
	switch d {
	case F32:
		return Block{1, 4}, nil
	case F16:
		return Block{1, 2}, nil
	case Q4_0:
		return Block{32, 18}, nil
	case Q4_1:
		return Block{32, 20}, nil
	case Q5_0:
		return Block{32, 22}, nil
	case Q5_1:
		return Block{32, 24}, nil
	case Q8_0:
		return Block{32, 34}, nil
	case Q8_1:
		return Block{32, 40}, nil
	case Q2_K:
		return Block{256, 84}, nil
	case Q3_K:
		return Block{256, 110}, nil
	case Q4_K:
		return Block{256, 144}, nil
	case Q5_K:
		return Block{256, 176}, nil
	case Q6_K:
		return Block{256, 210}, nil
	case Q8_K:
		return Block{256, 292}, nil
	case IQ2XS:
		return Block{256, 66}, nil
	default:
		return Block{}, fmt.Errorf("%w: %s", ErrUnsupportedDType, d)
	}
}

// RowBytes returns the packed byte size of a row with columns elements.
func (d DType) RowBytes(columns int64) (uint64, error) {
	block, err := d.BlockLayout()
	if err != nil {
		return 0, err
	}
	if columns <= 0 {
		return 0, fmt.Errorf("invalid column count %d", columns)
	}
	if block.Elements > 1 && columns%block.Elements != 0 {
		return 0, fmt.Errorf("%s row width %d is not a multiple of %d",
			d, columns, block.Elements)
	}
	blocks := columns / block.Elements
	if blocks > (1<<63-1)/block.Bytes {
		return 0, fmt.Errorf("%s row byte size overflows", d)
	}
	return uint64(blocks) * uint64(block.Bytes), nil
}

// Supported reports whether this build has a decoding kernel for the dtype.
func (d DType) Supported() bool {
	switch d {
	case F32, F16, Q8_0, Q4_K, Q5_K, Q6_K:
		return true
	default:
		return false
	}
}
