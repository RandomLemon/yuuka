//go:build hip && linux

package hip

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

// q6kMatrix builds a packed Q6_K matrix whose block scales stay in a range
// that keeps the reference dot products well conditioned.
func q6kMatrix(rows, cols int, seed int64) ([]byte, error) {
	rng := rand.New(rand.NewSource(seed))
	stride := (cols / 256) * 210
	data := make([]byte, rows*stride)
	for row := 0; row < rows; row++ {
		for block := 0; block < cols/256; block++ {
			base := row*stride + block*210
			for i := 0; i < 192; i++ {
				data[base+i] = byte(rng.Intn(256))
			}
			for i := 0; i < 16; i++ {
				data[base+192+i] = byte(int8(rng.Intn(21) - 10))
			}
			binary.LittleEndian.PutUint16(data[base+208:], tensor.Float32ToFloat16(0.002))
		}
	}
	return data, nil
}

func q8_0Matrix(rows, cols int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	stride := (cols / 32) * 34
	data := make([]byte, rows*stride)
	for row := 0; row < rows; row++ {
		for block := 0; block < cols/32; block++ {
			base := row*stride + block*34
			binary.LittleEndian.PutUint16(data[base:], tensor.Float32ToFloat16(0.25))
			for i := 0; i < 32; i++ {
				data[base+2+i] = byte(int8(rng.Intn(255) - 127))
			}
		}
	}
	return data
}

// TestPlanLMHeadMatchesDequant checks the streaming prefill LM head against the
// dequantized CPU dot product for a single row and for a multi-row block, and
// checks that the shape rules the kernel cannot serve are rejected. The kernel
// and the reference differ only in the summation order, so the gate is the
// float32 accumulation noise of a 2560-term dot product.
func TestPlanLMHeadMatchesDequant(t *testing.T) {
	dev := openTestDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer event.Close()

	const nIn, nOut = 2560, 400
	cases := []struct {
		name    string
		dtype   tensor.DType
		weights []byte
	}{
		{"Q6_K", tensor.Q6_K, nil},
		{"Q8_0", tensor.Q8_0, q8_0Matrix(nOut, nIn, 7)},
	}
	cases[0].weights, err = q6kMatrix(nOut, nIn, 11)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		for _, rows := range []int{1, 5, 16} {
			x := make([]float32, nIn*rows)
			for i := range x {
				x[i] = float32((i%13)-6) * 0.125
			}
			// Reference: dequantize each packed row and dot it with every row
			// of the activation block.
			blocks, err := tc.dtype.BlockLayout()
			if err != nil {
				t.Fatal(err)
			}
			rowBytes := (nIn / int(blocks.Elements)) * int(blocks.Bytes)
			row := make([]float32, nIn)
			want := make([]float32, nOut*rows)
			for col := 0; col < nOut; col++ {
				if err := tensor.DequantizeRow(row, tc.dtype,
					tc.weights[col*rowBytes:(col+1)*rowBytes]); err != nil {
					t.Fatal(err)
				}
				for m := 0; m < rows; m++ {
					var sum float64
					for k := 0; k < nIn; k++ {
						sum += float64(row[k]) * float64(x[m*nIn+k])
					}
					want[m*nOut+col] = float32(sum)
				}
			}

			wBuf, err := dev.Alloc(uint64(len(tc.weights)))
			if err != nil {
				t.Fatal(err)
			}
			xBuf, err := dev.Alloc(uint64(len(x) * 4))
			if err != nil {
				t.Fatal(err)
			}
			outBuf, err := dev.Alloc(uint64(len(want) * 4))
			if err != nil {
				t.Fatal(err)
			}
			if err := dev.Upload(wBuf, tc.weights); err != nil {
				t.Fatal(err)
			}
			if err := dev.Upload(xBuf, floatsToBytes(x)); err != nil {
				t.Fatal(err)
			}
			plan, err := dev.NewPlan()
			if err != nil {
				t.Fatal(err)
			}
			if err := plan.AddLMHeadAt(wBuf, 0, xBuf, 0, outBuf, 0, nIn, nOut,
				int64(rows), tc.dtype); err != nil {
				t.Fatalf("%s rows=%d: %v", tc.name, rows, err)
			}
			if err := plan.Execute(stream, 0, 0, true, event); err != nil {
				t.Fatal(err)
			}
			if err := event.Wait(); err != nil {
				t.Fatal(err)
			}
			out := make([]byte, len(want)*4)
			if err := dev.Download(out, outBuf); err != nil {
				t.Fatal(err)
			}
			got := bytesToFloats(out)
			maxDiff := 0.0
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 2e-4*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%s rows=%d index %d: got %v want %v (diff %g)",
						tc.name, rows, i, got[i], want[i], diff)
				}
			}
			t.Logf("%s rows=%d max abs diff %.3g", tc.name, rows, maxDiff)
			_ = plan.Close()
			_ = outBuf.Close()
			_ = xBuf.Close()
			_ = wBuf.Close()
		}
	}

	// Shape rules: the kernel covers 1..MaxLMHeadRows rows and a multiple of 256
	// input width. Anything else must be rejected before submission so the
	// caller can fall back to the per-row GEMV.
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	weights, err := dev.Alloc(uint64(512 * nOut))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	x, err := dev.Alloc(512 * 17 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := dev.Alloc(uint64(nOut * 17 * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err := plan.AddLMHeadAt(weights, 0, x, 0, out, 0, 512, nOut,
		MaxLMHeadRows+1, tensor.Q8_0); err == nil {
		t.Fatal("row block above the kernel cap was accepted")
	}
	if err := plan.AddLMHeadAt(weights, 0, x, 0, out, 0, 384, nOut, 1,
		tensor.Q8_0); err == nil {
		t.Fatal("input width that is not a multiple of 256 was accepted")
	}
	if err := plan.AddLMHeadAt(weights, 0, x, 0, out, 0, 512, nOut, 1,
		tensor.Q2_K); err == nil {
		t.Fatal("unsupported LM head dtype was accepted")
	}
	// The decoders read 128-bit groups, so a misaligned base would change what
	// they read; the plan rejects it instead of decoding the wrong bytes.
	if err := plan.AddLMHeadAt(weights, 4, x, 0, out, 0, 512, nOut, 1,
		tensor.Q8_0); err == nil {
		t.Fatal("misaligned weight offset was accepted")
	}
	if plan.OpCount() != 0 {
		t.Fatalf("rejected operations were added: %d", plan.OpCount())
	}
}

// TestSelectLMHeadPath pins the dispatch rule both output tails use.
func TestSelectLMHeadPath(t *testing.T) {
	cases := []struct {
		name      string
		dtype     tensor.DType
		rows      int64
		nIn, nOut int64
		want      bool
	}{
		{"single row", tensor.Q6_K, 1, 2560, 248320, true},
		{"speculative block", tensor.Q6_K, 16, 2560, 248320, true},
		{"above row cap", tensor.Q6_K, 17, 2560, 248320, false},
		{"zero rows", tensor.Q6_K, 0, 2560, 248320, false},
		{"narrow output", tensor.Q6_K, 1, 2048, 511, false},
		{"k-quant 512 output", tensor.Q6_K, 1, 2048, 512, true},
		{"q4k 511", tensor.Q4_K, 1, 2048, 511, false},
		{"q5k 512", tensor.Q5_K, 1, 2048, 512, true},
		{"q6k 512", tensor.Q6_K, 1, 2048, 512, true},
		{"q8_0 512", tensor.Q8_0, 1, 2048, 512, false},
		{"f16 512", tensor.F16, 1, 2048, 512, false},
		{"q4k 1024", tensor.Q4_K, 1, 2048, 1024, true},
		{"smallest selected width", tensor.Q6_K, 1, 2560, 1024, true},
		{"width not a K step", tensor.Q6_K, 1, 2560 + 128, 248320, false},
		{"q8_0", tensor.Q8_0, 1, 2560, 151936, true},
		{"f16", tensor.F16, 1, 4096, 151936, true},
		{"unsupported dtype", tensor.Q2_K, 1, 2560, 248320, false},
	}
	for _, c := range cases {
		got := SelectLMHeadPath(c.dtype, c.rows, c.nIn, c.nOut)
		if got != c.want {
			t.Errorf("%s: got %v want %v", c.name, got, c.want)
		}
	}
}

// TestSelectDecodeGemvPathKQuant512 pins the decode matvec route for the
// 512-output projections of the K-quant weights: the streaming kernel serves
// them, and every other dtype keeps the per-row block because its measured
// crossover stays at 1024 outputs.
func TestSelectDecodeGemvPathKQuant512(t *testing.T) {
	cases := []struct {
		dtype      tensor.DType
		nIn, nOut  int64
		int8Matvec bool
		want       DecodeGemvPath
	}{
		{tensor.Q5_K, 2048, 512, false, DecodeGemvStreaming},
		{tensor.Q6_K, 2048, 512, false, DecodeGemvStreaming},
		{tensor.Q4_K, 2048, 511, false, DecodeGemvPerRow},
		{tensor.Q8_0, 2048, 512, false, DecodeGemvPerRow},
		{tensor.F16, 2048, 512, false, DecodeGemvPerRow},
	}
	for _, c := range cases {
		got := SelectDecodeGemvPath(c.dtype, c.nIn, c.nOut, c.int8Matvec)
		if got != c.want {
			t.Errorf("%v nIn=%d nOut=%d: got %v want %v", c.dtype, c.nIn, c.nOut,
				got, c.want)
		}
	}
}
