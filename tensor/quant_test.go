package tensor

import (
	"encoding/binary"
	"encoding/hex"
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

func TestDequantizeQ4_0(t *testing.T) {
	src := make([]byte, 18)
	putHalf(src, 0, 0.5)
	for i := range 16 {
		src[2+i] = byte((i & 0x0f) | ((15 - i) << 4))
	}
	dst := make([]float32, 32)
	if err := DequantizeRow(dst, Q4_0, src); err != nil {
		t.Fatal(err)
	}
	for i := range 16 {
		if want := 0.5 * float32(i-8); dst[i] != want {
			t.Fatalf("low index %d: got %v want %v", i, dst[i], want)
		}
		if want := 0.5 * float32(15-i-8); dst[i+16] != want {
			t.Fatalf("high index %d: got %v want %v", i, dst[i+16], want)
		}
	}
}

func TestDequantizeQ5_0(t *testing.T) {
	src := make([]byte, 22)
	putHalf(src, 0, 1)
	// Set the high bit for value 0 and value 1 and leave the rest clear.
	src[2] = 0x03
	for i := range 16 {
		src[6+i] = byte(i & 0x0f)
	}
	dst := make([]float32, 32)
	if err := DequantizeRow(dst, Q5_0, src); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 0 {
		t.Fatalf("index 0: got %v want 0", dst[0])
	}
	if dst[1] != 1 {
		t.Fatalf("index 1: got %v want 1", dst[1])
	}
	if dst[2] != -14 {
		t.Fatalf("index 2: got %v want -14", dst[2])
	}
	if dst[16] != -16 {
		t.Fatalf("index 16: got %v want -16", dst[16])
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

func TestIQDTypeCapabilities(t *testing.T) {
	for _, dtype := range []DType{IQ2XXS, IQ2S, IQ3S} {
		if !dtype.Supported() {
			t.Fatalf("%s has no CPU decoder", dtype)
		}
		if dtype.HIPSupported() {
			t.Fatalf("%s incorrectly reports a HIP kernel", dtype)
		}
		if !dtype.HIPMoESupported() {
			t.Fatalf("%s has no HIP MoE kernel", dtype)
		}
		if _, err := dtype.BlockLayout(); err != nil {
			t.Fatalf("%s block layout: %v", dtype, err)
		}
	}
	for _, dtype := range []DType{F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K} {
		if !dtype.HIPSupported() {
			t.Fatalf("%s lost HIP support", dtype)
		}
		if !dtype.HIPMoESupported() {
			t.Fatalf("%s lost HIP MoE support", dtype)
		}
	}
	for _, dtype := range []DType{Q2_K, Q3_K, Q8_K, IQ2XS, IQ3XXS, IQ1S, IQ4NL, IQ4XS} {
		if dtype.HIPMoESupported() {
			t.Fatalf("%s incorrectly reports a HIP MoE kernel", dtype)
		}
	}
}

// TestIQGridDigitsAreNonZero pins the precondition of the branch-free sign
// application in the HIP IQ decoders: they negate a packed four-digit word with
// (v ^ mask) + (mask & 0x01010101), which carries into the neighbouring byte if
// a digit is zero. Every entry is therefore checked to hold a nonzero digit in
// each byte, and the digits are checked to stay in signed int8 range so the int8
// WMMA operand can carry them.
func TestIQGridDigitsAreNonZero(t *testing.T) {
	check := func(name string, index int, word uint64, bytes int) {
		for j := 0; j < bytes; j++ {
			digit := uint8(word >> (8 * j))
			if digit == 0 {
				t.Fatalf("%s grid[%d] byte %d is zero; the sign negate carries",
					name, index, j)
			}
			if digit > 127 {
				t.Fatalf("%s grid[%d] byte %d is %d, outside int8", name, index, j, digit)
			}
		}
	}
	for i, word := range iq2xxs_grid {
		check("IQ2_XXS", i, word, 8)
	}
	for i, word := range iq2s_grid {
		check("IQ2_S", i, word, 8)
	}
	for i, word := range iq3s_grid {
		check("IQ3_S", i, uint64(word), 4)
	}
}

func TestDequantizeIQReference(t *testing.T) {
	// Reference values were produced by ggml's dequantize_row_iq2_xxs,
	// dequantize_row_iq3_s, and dequantize_row_iq2_s on the first block of
	// each type from Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf.
	cases := []struct {
		name  string
		dtype DType
		raw   string
		first []float32
		sum   float32
	}{
		{
			name:  "IQ2_XXS",
			dtype: IQ2XXS,
			raw:   "2f0c070067cad0a4b8adba52aa0456f20f7b013a0c004f7332f58f6651311d43d68b23eba5d2ece74da21c8c5623474cc08e59d71e0d6377b69c22ec15786a8338bb",
			first: []float32{0.005362272262573242, 0.005362272262573242, 0.028822213411331177, 0.005362272262573242, -0.005362272262573242, 0.005362272262573242, -0.005362272262573242, 0.005362272262573242},
			sum:   -0.09559527039527893,
		},
		{
			name:  "IQ3_S",
			dtype: IQ3S,
			raw:   "76051fd85942b4e46e38b10d380fbc2ca7c5c61373867fed8a59c099a3df55e617fe8e574dd8282e7e8d3f9c002844e737e27987199ddc6f21c6d1f4a1213f94106899a002a7dd78845721b9846cec02ee3f5e044832ff7cabb771788225979696dd19b7214cf9b6a9638b8a999f",
			first: []float32{-0.0057495832443237305, 0.013415694236755371, 0.0019165277481079102, 0.013415694236755371, 0.0019165277481079102, -0.028747916221618652, 0.0019165277481079102, 0.00958263874053955},
			sum:   -0.0661618709564209,
		},
		{
			name:  "IQ2_S",
			dtype: IQ2S,
			raw:   "57099f38722503000d3a00c3f51de39e4c4a0e193e06d87f1b3f001d4f7f4900b602489e7a4b8a3a5ea690784a9e256b50e3007be615268e7d57c371164b0f34398a00c024c1000d70b0999ba959fb9c8ee7",
			first: []float32{0.016642197966575623, 0.0030962228775024414, 0.0030962228775024414, -0.0030962228775024414, 0.0030962228775024414, 0.016642197966575623, -0.0030962228775024414, 0.0030962228775024414},
			sum:   -0.14647985994815826,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := hex.DecodeString(tc.raw)
			if err != nil {
				t.Fatal(err)
			}
			dst := make([]float32, 256)
			if err := DequantizeRow(dst, tc.dtype, raw); err != nil {
				t.Fatal(err)
			}
			for i, want := range tc.first {
				if diff := math.Abs(float64(dst[i] - want)); diff > 1e-6 {
					t.Fatalf("index %d: got %v want %v", i, dst[i], want)
				}
			}
			var sum float32
			for _, value := range dst {
				sum += value
			}
			if diff := math.Abs(float64(sum - tc.sum)); diff > 1e-5 {
				t.Fatalf("sum: got %v want %v", sum, tc.sum)
			}
		})
	}
}

func TestMatMulMatchesDequantizedDot(t *testing.T) {
	const rows, cols = 3, 256
	cases := []DType{F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ2XXS, IQ2S, IQ3S}
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
				case Q4_0:
					for i := range cols / 32 {
						putHalf(row, i*18, 0.03125)
					}
				case Q5_0:
					for i := range cols / 32 {
						putHalf(row, i*22, 0.03125)
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
				case IQ2XXS, IQ2S, IQ3S:
					blockBytes, err := dtype.BlockLayout()
					if err != nil {
						t.Fatal(err)
					}
					for i := 0; i < len(row); i += int(blockBytes.Bytes) {
						putHalf(row, i, 0.25)
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
