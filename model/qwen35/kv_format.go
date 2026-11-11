package qwen35

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/tensor"
)

// KVFormat selects the storage format of full-attention key and value rows.
// F32 is the default and preserves the reference path. F16 halves cache
// capacity cost. Q8_0 stores each 32-value group with one FP16 scale and 32
// signed bytes; it is the higher-compression experimental format.
type KVFormat uint8

const (
	// KVF32 stores keys and values as FP32.
	KVF32 KVFormat = iota
	// KVF16 stores keys and values as FP16.
	KVF16
	// KVQ8_0 stores each 32-value group as one FP16 scale and 32 int8 values.
	KVQ8_0
)

// ParseKVFormat maps a configuration string to a KV format. The empty string
// selects F32. Names are case-insensitive at the caller boundary; this function
// accepts the canonical upper-case names.
func ParseKVFormat(name string) (KVFormat, error) {
	switch name {
	case "", "f32", "F32":
		return KVF32, nil
	case "f16", "F16":
		return KVF16, nil
	case "q8_0", "Q8_0":
		return KVQ8_0, nil
	default:
		return 0, fmt.Errorf("qwen35: unknown KV format %q", name)
	}
}

// String returns the canonical format name.
func (f KVFormat) String() string {
	switch f {
	case KVF32:
		return "F32"
	case KVF16:
		return "F16"
	case KVQ8_0:
		return "Q8_0"
	default:
		return fmt.Sprintf("KVFormat(%d)", uint8(f))
	}
}

// DType returns the corresponding tensor dtype used by HIP plans.
func (f KVFormat) DType() tensor.DType {
	switch f {
	case KVF16:
		return tensor.F16
	case KVQ8_0:
		return tensor.Q8_0
	default:
		return tensor.F32
	}
}

// RowBytes returns the stored byte size of one KV row.
func (f KVFormat) RowBytes(kvDim int) (int, error) {
	if kvDim <= 0 {
		return 0, fmt.Errorf("qwen35: invalid KV row width %d", kvDim)
	}
	switch f {
	case KVF32:
		return kvDim * 4, nil
	case KVF16:
		return kvDim * 2, nil
	case KVQ8_0:
		return (kvDim + 31) / 32 * 34, nil
	default:
		return 0, fmt.Errorf("qwen35: unknown KV format %d", f)
	}
}

// EncodeRow writes one FP32 key or value row in the selected format. dst must
// hold exactly RowBytes(kvDim) bytes.
func (f KVFormat) EncodeRow(dst []byte, src []float32) error {
	want, err := f.RowBytes(len(src))
	if err != nil {
		return err
	}
	if len(dst) != want {
		return fmt.Errorf("qwen35: KV destination has %d bytes, want %d", len(dst), want)
	}
	switch f {
	case KVF32:
		for i, value := range src {
			binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(value))
		}
	case KVF16:
		for i, value := range src {
			binary.LittleEndian.PutUint16(dst[i*2:], tensor.Float32ToFloat16(value))
		}
	case KVQ8_0:
		for block := 0; block < (len(src)+31)/32; block++ {
			start := block * 32
			end := min(start+32, len(src))
			values := src[start:end]
			var maxAbs float32
			for _, value := range values {
				if abs := float32(math.Abs(float64(value))); abs > maxAbs {
					maxAbs = abs
				}
			}
			scale := maxAbs / 127
			off := block * 34
			binary.LittleEndian.PutUint16(dst[off:], tensor.Float32ToFloat16(scale))
			for i := range 32 {
				q := int8(0)
				if i < len(values) && scale != 0 {
					q = int8(math.Round(float64(values[i] / scale)))
				}
				dst[off+2+i] = byte(q)
			}
		}
	default:
		return fmt.Errorf("qwen35: unknown KV format %d", f)
	}
	return nil
}

// DecodeRow expands one stored row into dst. dst must hold exactly kvDim
// values.
func (f KVFormat) DecodeRow(dst []float32, src []byte) error {
	kvDim := len(dst)
	want, err := f.RowBytes(kvDim)
	if err != nil {
		return err
	}
	if len(src) != want {
		return fmt.Errorf("qwen35: KV source has %d bytes, want %d", len(src), want)
	}
	switch f {
	case KVF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		}
	case KVF16:
		for i := range dst {
			dst[i] = tensor.Float16ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
		}
	case KVQ8_0:
		for i := range dst {
			off := i / 32 * 34
			scale := tensor.Float16ToFloat32(binary.LittleEndian.Uint16(src[off:]))
			dst[i] = scale * float32(int8(src[off+2+i%32]))
		}
	default:
		return fmt.Errorf("qwen35: unknown KV format %d", f)
	}
	return nil
}
