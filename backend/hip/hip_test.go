//go:build hip && linux

package hip

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

func openTestDevice(t *testing.T) *Device {
	t.Helper()
	dev, err := Open(0)
	if err != nil {
		t.Skipf("no HIP device: %v", err)
	}
	t.Cleanup(func() { _ = dev.Close() })
	return dev
}

func TestDeviceInfo(t *testing.T) {
	dev := openTestDevice(t)
	info := dev.Info()
	if info.Name == "" || info.TotalMemory == 0 {
		t.Fatalf("unexpected device info: %+v", info)
	}
	if ABIVersion() == 0 {
		t.Fatal("ABI version is zero")
	}
	t.Logf("%+v", info)
}

func TestPlanCopy(t *testing.T) {
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
	src, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	dst, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	input := []float32{1, -2, 3, 4}
	if err := dev.Upload(src, floatsToBytes(input)); err != nil {
		t.Fatal(err)
	}
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if err := plan.AddCopy(dst, src, 4); err != nil {
		t.Fatal(err)
	}
	if err := plan.Execute(stream, 0, 0, true, event); err != nil {
		t.Fatal(err)
	}
	if err := event.Wait(); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 16)
	if err := dev.Download(out, dst); err != nil {
		t.Fatal(err)
	}
	got := bytesToFloats(out)
	for i := range input {
		if got[i] != input[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], input[i])
		}
	}
}

func TestPlanGemvQ8_0(t *testing.T) {
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

	const nIn, nOut = 64, 3
	weights := make([]byte, nOut*(nIn/32)*34)
	for row := 0; row < nOut; row++ {
		for block := 0; block < nIn/32; block++ {
			base := row*(nIn/32)*34 + block*34
			binary.LittleEndian.PutUint16(weights[base:], tensor.Float32ToFloat16(0.25))
			for i := 0; i < 32; i++ {
				weights[base+2+i] = byte(int8((row+1)*(i%7) - 3))
			}
		}
	}
	x := make([]float32, nIn)
	for i := range x {
		x[i] = float32(i%9) - 4
	}
	matrix := tensor.Matrix{Data: weights, Rows: nOut, Cols: nIn, DType: tensor.Q8_0}
	want := make([]float32, nOut)
	if err := matrix.MatVec(want, x); err != nil {
		t.Fatal(err)
	}

	wBuf, _ := dev.Alloc(uint64(len(weights)))
	defer wBuf.Close()
	xBuf, _ := dev.Alloc(uint64(len(x) * 4))
	defer xBuf.Close()
	outBuf, _ := dev.Alloc(nOut * 4)
	defer outBuf.Close()
	if err := dev.Upload(wBuf, weights); err != nil {
		t.Fatal(err)
	}
	if err := dev.Upload(xBuf, floatsToBytes(x)); err != nil {
		t.Fatal(err)
	}
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if err := plan.AddGemv(wBuf, xBuf, outBuf, nIn, nOut, tensor.Q8_0); err != nil {
		t.Fatal(err)
	}
	if err := plan.Execute(stream, 0, 0, true, event); err != nil {
		t.Fatal(err)
	}
	if err := event.Wait(); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, nOut*4)
	if err := dev.Download(out, outBuf); err != nil {
		t.Fatal(err)
	}
	got := bytesToFloats(out)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("row %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestPlanOffsets(t *testing.T) {
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
	src, _ := dev.Alloc(2 * 4 * 4)
	defer src.Close()
	dst, _ := dev.Alloc(2 * 4 * 4)
	defer dst.Close()
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := dev.Upload(src, floatsToBytes(input)); err != nil {
		t.Fatal(err)
	}
	if err := dev.Upload(dst, make([]byte, 32)); err != nil {
		t.Fatal(err)
	}
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	// Normalize only the second four-element head.
	if err := plan.AddL2NormAt(src, 16, dst, 16, 1, 4, 1e-6, 1); err != nil {
		t.Fatal(err)
	}
	if err := plan.Execute(stream, 0, 0, true, event); err != nil {
		t.Fatal(err)
	}
	if err := event.Wait(); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 32)
	if err := dev.Download(out, dst); err != nil {
		t.Fatal(err)
	}
	got := bytesToFloats(out)
	if got[0] != 0 || got[1] != 0 || got[2] != 0 || got[3] != 0 {
		t.Fatalf("unexpected first head: %v", got[:4])
	}
	norm := float32(math.Sqrt(25 + 36 + 49 + 64 + 1e-6))
	for i := 0; i < 4; i++ {
		want := input[4+i] / norm
		if math.Abs(float64(got[4+i]-want)) > 1e-4 {
			t.Fatalf("index %d: got %v want %v", 4+i, got[4+i], want)
		}
	}
}

func floatsToBytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func bytesToFloats(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func TestPlanGemmF16(t *testing.T) {
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

	const nIn, nOut, batch = 35, 4, 3
	weights := make([]byte, nOut*nIn*2)
	values := make([]float32, nOut*nIn)
	for i := range values {
		values[i] = float32((i%13)-6) * 0.25
		half := tensor.Float32ToFloat16(values[i])
		binary.LittleEndian.PutUint16(weights[i*2:], half)
	}
	// Decode the stored half values so the reference uses the exact weights
	// uploaded to the device.
	for i := 0; i < nOut*nIn; i++ {
		values[i] = tensor.Float16ToFloat32(binary.LittleEndian.Uint16(weights[i*2:]))
	}
	x := make([]float32, nIn*batch)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.5
	}
	want := make([]float32, nOut*batch)
	for row := 0; row < batch; row++ {
		for col := 0; col < nOut; col++ {
			var sum float32
			for i := 0; i < nIn; i++ {
				sum += x[row*nIn+i] * values[col*nIn+i]
			}
			want[row*nOut+col] = sum
		}
	}

	wBuf, err := dev.Alloc(uint64(len(weights)))
	if err != nil {
		t.Fatal(err)
	}
	defer wBuf.Close()
	xBuf, err := dev.Alloc(uint64(len(x) * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer xBuf.Close()
	outBuf, err := dev.Alloc(uint64(len(want) * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Close()
	if err := dev.Upload(wBuf, weights); err != nil {
		t.Fatal(err)
	}
	if err := dev.Upload(xBuf, floatsToBytes(x)); err != nil {
		t.Fatal(err)
	}
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if err := plan.AddGemm(wBuf, xBuf, outBuf, nIn, nOut, batch, tensor.F16); err != nil {
		t.Fatal(err)
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
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-3*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestPlanGemvBatchQ8_0(t *testing.T) {
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

	const nIn, nOut, batch = 64, 3, 4
	weights := make([]byte, nOut*(nIn/32)*34)
	for row := 0; row < nOut; row++ {
		for block := 0; block < nIn/32; block++ {
			base := row*(nIn/32)*34 + block*34
			binary.LittleEndian.PutUint16(weights[base:], tensor.Float32ToFloat16(0.25))
			for i := 0; i < 32; i++ {
				weights[base+2+i] = byte(int8((row+1)*(i%7) - 3))
			}
		}
	}
	x := make([]float32, nIn*batch)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.25
	}
	matrix := tensor.Matrix{Data: weights, Rows: nOut, Cols: nIn, DType: tensor.Q8_0}
	want := make([]float32, nOut*batch)
	for row := 0; row < batch; row++ {
		if err := matrix.MatVec(want[row*nOut:(row+1)*nOut], x[row*nIn:(row+1)*nIn]); err != nil {
			t.Fatal(err)
		}
	}

	wBuf, _ := dev.Alloc(uint64(len(weights)))
	defer wBuf.Close()
	xBuf, _ := dev.Alloc(uint64(len(x) * 4))
	defer xBuf.Close()
	outBuf, _ := dev.Alloc(uint64(len(want) * 4))
	defer outBuf.Close()
	if err := dev.Upload(wBuf, weights); err != nil {
		t.Fatal(err)
	}
	if err := dev.Upload(xBuf, floatsToBytes(x)); err != nil {
		t.Fatal(err)
	}
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	if err := plan.AddGemvBatch(wBuf, xBuf, outBuf, nIn, nOut, batch, tensor.Q8_0); err != nil {
		t.Fatal(err)
	}
	if err := plan.AddGemvBatch(wBuf, xBuf, outBuf, nIn, nOut, 17, tensor.Q8_0); err == nil {
		t.Fatal("batch 17 was accepted")
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
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}
