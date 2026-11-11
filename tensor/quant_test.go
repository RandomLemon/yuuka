package tensor

import (
	"encoding/binary"
	"math"
	"testing"
)

func putHalf(dst []byte, off int, v float32) {
	binary.LittleEndian.PutUint16(dst[off:], Float32ToFloat16(v))
}

func TestFloat16RoundTrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, 65504, 6.103515625e-05, 5.960464477539063e-08, float32(math.Inf(1)), float32(math.Inf(-1))}
	for _, v := range values {
		half := Float32ToFloat16(v)
		got := Float16ToFloat32(half)
		if math.IsInf(float64(v), 0) {
			if !math.IsInf(float64(got), int(math.Copysign(1, float64(v)))) {
				t.Fatalf("half round trip %v -> %v", v, got)
			}
			continue
		}
		if v == 0 {
			if got != 0 || math.Signbit(float64(got)) != math.Signbit(float64(v)) {
				t.Fatalf("zero round trip %v -> %v", v, got)
			}
			continue
		}
		if math.Abs(float64(got-v)) > math.Abs(float64(v))*0.001 {
			t.Fatalf("half round trip %v -> %v", v, got)
		}
	}
}

func TestDequantizeQ8_0(t *testing.T) {
	src := make([]byte, 34)
	putHalf(src, 0, 0.5)
	for i := range 32 {
		src[2+i] = byte(int8(i - 16))
	}
	dst := make([]float32, 32)
	if err := DequantizeRow(dst, Q8_0, src); err != nil {
		t.Fatal(err)
	}
	for i := range 32 {
		want := 0.5 * float32(i-16)
		if dst[i] != want {
			t.Fatalf("index %d: got %v want %v", i, dst[i], want)
		}
	}
}

func TestDequantizeQ4K(t *testing.T) {
	src := make([]byte, 144)
	putHalf(src, 0, 1)
	putHalf(src, 2, 0)
	// All eight sub-block scales are 1 and all mins are 0.
	for i := range 4 {
		src[4+i] = 1
	}
	for i := 12; i < 16; i++ {
		src[i] = 1
	}
	for i := 16; i < 144; i++ {
		src[i] = 0x01
	}
	dst := make([]float32, 256)
	if err := DequantizeRow(dst, Q4_K, src); err != nil {
		t.Fatal(err)
	}
	for i := range 256 {
		want := float32(0)
		if i%64 < 32 {
			want = 1
		}
		if dst[i] != want {
			t.Fatalf("index %d: got %v want %v", i, dst[i], want)
		}
	}
}

func TestDequantizeQ5K(t *testing.T) {
	src := make([]byte, 176)
	putHalf(src, 0, 1)
	putHalf(src, 2, 0)
	for i := range 4 {
		src[4+i] = 1
	}
	for i := 12; i < 16; i++ {
		src[i] = 1
	}
	// High-bit lane 0 enables +16 for the first value of every group.
	src[16] = 0x55
	dst := make([]float32, 256)
	if err := DequantizeRow(dst, Q5_K, src); err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{0, 64, 128, 192} {
		if dst[i] != 16 {
			t.Fatalf("index %d: got %v want 16", i, dst[i])
		}
	}
	if dst[1] != 0 {
		t.Fatalf("index 1: got %v want 0", dst[1])
	}
}

func TestDequantizeQ6K(t *testing.T) {
	src := make([]byte, 210)
	src[0] = 0x0f
	src[128] = 0x02
	src[192] = 1
	putHalf(src, 208, 1)
	dst := make([]float32, 256)
	if err := DequantizeRow(dst, Q6_K, src); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 15 {
		t.Fatalf("index 0: got %v want 15", dst[0])
	}
	if dst[1] != -32 {
		t.Fatalf("index 1: got %v want -32", dst[1])
	}
}

func TestMatMulMatchesDequantizedDot(t *testing.T) {
	const rows, cols = 3, 256
	cases := []DType{F32, F16, Q8_0, Q4_K, Q5_K, Q6_K}
	for _, dtype := range cases {
		t.Run(dtype.String(), func(t *testing.T) {
			block, err := dtype.BlockLayout()
			if err != nil {
				t.Fatal(err)
			}
			rowBytes := int64(cols) / block.Elements * block.Bytes
			data := make([]byte, int64(rows)*rowBytes)
			for r := range rows {
				row := data[int64(r)*rowBytes:]
				for i := range row {
					row[i] = byte((r*37 + i*11) % 255)
				}
				switch dtype {
				case F32:
					for i := range cols {
						binary.LittleEndian.PutUint32(row[i*4:], math.Float32bits(float32(i%7)-3))
					}
				case F16:
					for i := range cols {
						putHalf(row, i*2, float32(i%5)-2)
					}
				case Q8_0:
					for i := range cols / 32 {
						putHalf(row, i*34, 0.03125)
					}
				case Q4_K:
					for i := 0; i < len(row); i += 144 {
						putHalf(row, i, 0.25)
						putHalf(row, i+2, 0.125)
					}
				case Q5_K:
					for i := 0; i < len(row); i += 176 {
						putHalf(row, i, 0.25)
						putHalf(row, i+2, 0.125)
					}
				case Q6_K:
					for i := 0; i < len(row); i += 210 {
						putHalf(row, i+208, 0.25)
					}
				}
			}
			m := Matrix{Data: data, Rows: rows, Cols: cols, DType: dtype}
			x := make([]float32, cols)
			for i := range x {
				x[i] = float32(i%9) - 4
			}
			want := make([]float32, rows)
			for r := range rows {
				decoded := make([]float32, cols)
				row := data[int64(r)*rowBytes : int64(r+1)*rowBytes]
				if err := DequantizeRow(decoded, dtype, row); err != nil {
					t.Fatal(err)
				}
				var sum float32
				for i := range cols {
					sum += decoded[i] * x[i]
				}
				want[r] = sum
			}
			got := make([]float32, rows)
			if err := m.MatVec(got, x); err != nil {
				t.Fatal(err)
			}
			for r := range rows {
				if math.Abs(float64(got[r]-want[r])) > 1e-5 {
					t.Fatalf("row %d: got %v want %v", r, got[r], want[r])
				}
			}
		})
	}
}
