//go:build hip && linux

// Package hip binds the yuuka HIP C ABI. It is built only with the hip tag so
// that the default Go build does not require the native shared library.
package hip

/*
#cgo CFLAGS: -I${SRCDIR}/bridge
#cgo LDFLAGS: -L${SRCDIR}/../../build -lyuuka_hip -Wl,-rpath,${SRCDIR}/../../build
#include <stdlib.h>
#include "yuuka_hip.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/yuuka-ai/yuuka/tensor"
)

// DeviceInfo describes the opened HIP device.
type DeviceInfo struct {
	Name         string
	GCNArch      string
	HIPVersion   string
	TotalMemory  uint64
	ComputeUnits uint32
	ClockMHz     uint32
	PCIDomain    int32
	PCIBus       int32
	PCIDevice    int32
}

// Device is an opened HIP device.
type Device struct {
	ptr    *C.yuuka_device
	info   DeviceInfo
	closed bool
}

// Open opens a HIP device by index.
func Open(index int) (*Device, error) {
	var ptr *C.yuuka_device
	if err := status(C.yuuka_device_open(C.int32_t(index), &ptr), "device_open"); err != nil {
		return nil, err
	}
	var info C.yuuka_device_info
	if err := status(C.yuuka_device_info_get(ptr, &info), "device_info"); err != nil {
		C.yuuka_device_close(ptr)
		return nil, err
	}
	d := &Device{
		ptr: ptr,
		info: DeviceInfo{
			Name:         C.GoString(&info.name[0]),
			GCNArch:      C.GoString(&info.gcn_arch[0]),
			HIPVersion:   C.GoString(&info.hip_version[0]),
			TotalMemory:  uint64(info.total_memory),
			ComputeUnits: uint32(info.compute_units),
			ClockMHz:     uint32(info.clock_mhz),
			PCIDomain:    int32(info.pci_domain),
			PCIBus:       int32(info.pci_bus),
			PCIDevice:    int32(info.pci_device),
		},
	}
	return d, nil
}

// Info returns the cached device properties.
func (d *Device) Info() DeviceInfo { return d.info }

// Close releases the device handle. It is idempotent.
func (d *Device) Close() error {
	if d == nil || d.closed {
		return nil
	}
	C.yuuka_device_close(d.ptr)
	d.ptr = nil
	d.closed = true
	return nil
}

// ABIVersion returns the shared library ABI version.
func ABIVersion() uint32 { return uint32(C.yuuka_abi_version()) }

// Buffer is a device allocation owned by the caller.
type Buffer struct {
	ptr    *C.yuuka_buffer
	bytes  uint64
	device *Device
	closed bool
}

// Alloc allocates device memory.
func (d *Device) Alloc(bytes uint64) (*Buffer, error) {
	if d == nil || d.closed {
		return nil, errors.New("hip: device is closed")
	}
	var ptr *C.yuuka_buffer
	if err := status(C.yuuka_buffer_alloc(d.ptr, C.uint64_t(bytes), &ptr), "buffer_alloc"); err != nil {
		return nil, err
	}
	return &Buffer{ptr: ptr, bytes: bytes, device: d}, nil
}

// Bytes returns the allocation size.
func (b *Buffer) Bytes() uint64 { return b.bytes }

// Pointer returns the raw device pointer. The caller must not use it after
// Close.
func (b *Buffer) Pointer() unsafe.Pointer {
	if b == nil || b.closed || b.ptr == nil {
		return nil
	}
	return C.yuuka_buffer_ptr(b.ptr)
}

// Close releases the allocation exactly once.
func (b *Buffer) Close() error {
	if b == nil || b.closed {
		return nil
	}
	C.yuuka_buffer_free(b.ptr)
	b.ptr = nil
	b.closed = true
	return nil
}

// Upload copies host bytes into a device buffer synchronously.
func (d *Device) Upload(dst *Buffer, data []byte) error {
	if err := d.validate(dst, uint64(len(data))); err != nil {
		return err
	}
	if len(data) == 0 {
		return errors.New("hip: upload is empty")
	}
	status := C.yuuka_memcpy_h2d(dst.ptr, unsafe.Pointer(&data[0]), C.uint64_t(len(data)))
	runtime.KeepAlive(data)
	return statusError(status, "memcpy_h2d")
}

// Download copies device bytes into host memory synchronously.
func (d *Device) Download(dst []byte, src *Buffer) error {
	if err := d.validate(src, uint64(len(dst))); err != nil {
		return err
	}
	if len(dst) == 0 {
		return errors.New("hip: download is empty")
	}
	status := C.yuuka_memcpy_d2h(unsafe.Pointer(&dst[0]), src.ptr, C.uint64_t(len(dst)))
	runtime.KeepAlive(dst)
	return statusError(status, "memcpy_d2h")
}

// Memset fills a device buffer on a stream.
func (d *Device) Memset(stream *Stream, dst *Buffer, value int, bytes uint64) error {
	if d == nil || d.closed {
		return errors.New("hip: device is closed")
	}
	if stream == nil || stream.closed {
		return errors.New("hip: stream is closed")
	}
	if err := d.validate(dst, bytes); err != nil {
		return err
	}
	return status(C.yuuka_memset(stream.ptr, dst.ptr, C.int(value), C.uint64_t(bytes)), "memset")
}

// DownloadFloats copies device bytes into a float32 slice.
func (d *Device) DownloadFloats(dst []float32, src *Buffer) error {
	if len(dst) == 0 {
		return errors.New("hip: download is empty")
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*4)
	return d.Download(bytes, src)
}

// UploadInt32 copies an int32 slice into a device buffer.
func (d *Device) UploadInt32(dst *Buffer, src []int32) error {
	if len(src) == 0 {
		return errors.New("hip: upload is empty")
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	return d.Upload(dst, bytes)
}

// UploadFloats copies a float32 slice into a device buffer.
func (d *Device) UploadFloats(dst *Buffer, src []float32) error {
	if len(src) == 0 {
		return errors.New("hip: upload is empty")
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	return d.Upload(dst, bytes)
}

func (d *Device) validate(buf *Buffer, bytes uint64) error {
	if d == nil || d.closed {
		return errors.New("hip: device is closed")
	}
	if buf == nil || buf.closed {
		return errors.New("hip: buffer is closed")
	}
	if buf.device != d {
		return errors.New("hip: buffer belongs to a different device")
	}
	if bytes > buf.bytes {
		return fmt.Errorf("hip: transfer of %d bytes exceeds buffer of %d", bytes, buf.bytes)
	}
	return nil
}

// Stream is an explicit HIP stream.
type Stream struct {
	ptr    *C.yuuka_stream
	closed bool
}

func (d *Device) NewStream() (*Stream, error) {
	if d == nil || d.closed {
		return nil, errors.New("hip: device is closed")
	}
	var ptr *C.yuuka_stream
	if err := status(C.yuuka_stream_create(d.ptr, &ptr), "stream_create"); err != nil {
		return nil, err
	}
	return &Stream{ptr: ptr}, nil
}

func (s *Stream) Synchronize() error {
	if s == nil || s.closed {
		return errors.New("hip: stream is closed")
	}
	return status(C.yuuka_stream_synchronize(s.ptr), "stream_synchronize")
}

func (s *Stream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	C.yuuka_stream_destroy(s.ptr)
	s.ptr = nil
	s.closed = true
	return nil
}

// Event signals completion of submitted work.
type Event struct {
	ptr    *C.yuuka_event
	closed bool
}

func (d *Device) NewEvent() (*Event, error) {
	if d == nil || d.closed {
		return nil, errors.New("hip: device is closed")
	}
	var ptr *C.yuuka_event
	if err := status(C.yuuka_event_create(d.ptr, &ptr), "event_create"); err != nil {
		return nil, err
	}
	return &Event{ptr: ptr}, nil
}

func (e *Event) Wait() error {
	if e == nil || e.closed {
		return errors.New("hip: event is closed")
	}
	return status(C.yuuka_event_wait(e.ptr), "event_wait")
}

func (e *Event) Close() error {
	if e == nil || e.closed {
		return nil
	}
	C.yuuka_event_destroy(e.ptr)
	e.ptr = nil
	e.closed = true
	return nil
}

func status(st C.yuuka_status, what string) error {
	return statusError(st, what)
}

func statusError(st C.yuuka_status, what string) error {
	if st == C.YUUKA_OK {
		return nil
	}
	return fmt.Errorf("hip: %s: %s", what, C.GoString(C.yuuka_last_error()))
}

// Plan batches generic device operations into one submission.
type Plan struct {
	ptr    *C.yuuka_plan
	device *Device
	tail   bool
	ops    int
}

func (d *Device) NewPlan() (*Plan, error) {
	if d == nil || d.closed {
		return nil, errors.New("hip: device is closed")
	}
	var ptr *C.yuuka_plan
	if err := status(C.yuuka_plan_create(d.ptr, &ptr), "plan_create"); err != nil {
		return nil, err
	}
	return &Plan{ptr: ptr, device: d}, nil
}

// MarkTail marks subsequent operations as output-tail operations. Execute with
// runTail=false skips them.
func (p *Plan) MarkTail() { p.tail = true }

func (p *Plan) add(op C.yuuka_op) error {
	if p == nil || p.ptr == nil {
		return errors.New("hip: plan is closed")
	}
	if p.tail {
		op.flags |= C.YUUKA_OP_FLAG_TAIL
	}
	if err := status(C.yuuka_plan_add(p.ptr, &op), "plan_add"); err != nil {
		return err
	}
	p.ops++
	return nil
}

func (p *Plan) check(buffers ...*Buffer) error {
	for _, buf := range buffers {
		if buf == nil || buf.closed || buf.ptr == nil {
			return errors.New("hip: plan references a closed buffer")
		}
		if buf.device != p.device {
			return errors.New("hip: plan buffer belongs to a different device")
		}
	}
	return nil
}

// checkGemvDType rejects dtypes without a device kernel and widths that do not
// satisfy the quantization block layout. It mirrors the C ABI validation so the
// Go planner fails before submission.
func checkGemvDType(dtype tensor.DType, nIn int64) error {
	if !dtype.Supported() {
		return fmt.Errorf("hip: GEMV dtype %s is not supported", dtype)
	}
	if _, err := dtype.RowBytes(nIn); err != nil {
		return fmt.Errorf("hip: GEMV shape: %w", err)
	}
	return nil
}

// AddGemv appends out = W * x for a packed weight matrix.
func (p *Plan) AddGemv(w, x, out *Buffer, nIn, nOut int64, dtype tensor.DType) error {
	return p.AddGemvAt(w, 0, x, 0, out, 0, nIn, nOut, dtype)
}

// AddGemvAt appends a GEMV with byte offsets applied to each buffer.
func (p *Plan) AddGemvAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut int64, dtype tensor.DType) error {
	if nIn <= 0 || nOut <= 0 {
		return errors.New("hip: GEMV dimensions must be positive")
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if err := p.check(w, x, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMV}
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddGemvBatch appends out = W * X for a packed decode weight matrix and a
// batch of up to 16 rows. x is [batch, nIn] and out is [batch, nOut].
func (p *Plan) AddGemvBatch(w, x, out *Buffer, nIn, nOut, batch int64, dtype tensor.DType) error {
	return p.AddGemvBatchAt(w, 0, x, 0, out, 0, nIn, nOut, batch, dtype)
}

// AddGemvBatchAt appends a batched GEMV with byte offsets applied to each buffer.
func (p *Plan) AddGemvBatchAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType) error {
	if batch <= 0 || batch > 16 {
		return fmt.Errorf("hip: batched GEMV batch %d is outside 1..16", batch)
	}
	if nIn <= 0 || nOut <= 0 {
		return errors.New("hip: batched GEMV dimensions must be positive")
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if err := p.check(w, x, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMV_BATCH}
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddGemm appends out = W * X for a packed weight matrix and a batch of
// contiguous input rows. x is [batch, nIn] and out is [batch, nOut].
func (p *Plan) AddGemm(w, x, out *Buffer, nIn, nOut, batch int64, dtype tensor.DType) error {
	return p.AddGemmAt(w, 0, x, 0, out, 0, nIn, nOut, batch, dtype)
}

// AddGemmAt appends a GEMM with byte offsets applied to each buffer.
func (p *Plan) AddGemmAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType) error {
	if dtype != tensor.F32 && dtype != tensor.F16 {
		return fmt.Errorf("hip: GEMM dtype %s is not supported", dtype)
	}
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: GEMM dimensions must be positive")
	}
	if err := p.check(w, x, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMM}
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

func (p *Plan) AddRMSNorm(src, weight, dst *Buffer, n int64, eps float32) error {
	return p.AddRMSNormAt(src, 0, weight, dst, 0, n, eps)
}

func (p *Plan) AddRMSNormAt(src *Buffer, srcOff uint64, weight *Buffer, dst *Buffer, dstOff uint64, n int64, eps float32) error {
	if err := p.check(src, weight, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RMSNORM}
	op.buf[0] = src.ptr
	op.buf[1] = weight.ptr
	op.buf[2] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[2] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(n)
	op.f[0] = C.float(eps)
	return p.add(op)
}

func (p *Plan) AddL2Norm(src, dst *Buffer, heads, size int64, eps, scale float32) error {
	return p.AddL2NormAt(src, 0, dst, 0, heads, size, eps, scale)
}

func (p *Plan) AddL2NormAt(src *Buffer, srcOff uint64, dst *Buffer, dstOff uint64, heads, size int64, eps, scale float32) error {
	if err := p.check(src, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_L2NORM}
	op.buf[0] = src.ptr
	op.buf[2] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[2] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(heads)
	op.n[1] = C.uint64_t(size)
	op.n[2] = C.uint64_t(size)
	op.f[0] = C.float(eps)
	op.f[1] = C.float(scale)
	return p.add(op)
}

func (p *Plan) AddGatedRMSNorm(src, weight, gate, dst *Buffer, heads, size int64, eps float32) error {
	if err := p.check(src, weight, gate, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GATED_RMSNORM}
	op.buf[0] = src.ptr
	op.buf[1] = weight.ptr
	op.buf[2] = gate.ptr
	op.buf[3] = dst.ptr
	op.n[0] = C.uint64_t(size)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(size)
	op.f[0] = C.float(eps)
	return p.add(op)
}

func (p *Plan) AddRMSNormHeadsAt(src *Buffer, srcOff uint64, weight *Buffer, dst *Buffer, dstOff uint64, heads int, size, stride int64, eps float32) error {
	if err := p.check(src, weight, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RMSNORM_HEADS}
	op.buf[0] = src.ptr
	op.buf[1] = weight.ptr
	op.buf[2] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[2] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(size)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(stride)
	op.f[0] = C.float(eps)
	return p.add(op)
}

func (p *Plan) AddRoPE(data *Buffer, heads, stride, headDim, rotDim int, base float32) error {
	if err := p.check(data); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ROPE}
	op.buf[0] = data.ptr
	op.n[0] = C.uint64_t(heads)
	op.n[1] = C.uint64_t(stride)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(rotDim)
	op.f[0] = C.float(base)
	return p.add(op)
}

func (p *Plan) AddAttention(q, kCache, vCache, out, scores *Buffer, kvHeads, heads, headDim, kvDim, capacity int, scale float32) error {
	if err := p.check(q, kCache, vCache, out, scores); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ATTENTION}
	op.buf[0] = q.ptr
	op.buf[1] = kCache.ptr
	op.buf[2] = vCache.ptr
	op.buf[3] = out.ptr
	op.buf[4] = scores.ptr
	op.n[0] = C.uint64_t(kvHeads)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(kvDim)
	op.u[0] = C.uint32_t(capacity)
	op.f[0] = C.float(scale)
	return p.add(op)
}

func (p *Plan) AddConv1D(state, weights, qkv, out *Buffer, channels, kernel int) error {
	if err := p.check(state, weights, qkv, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_CONV1D}
	op.buf[0] = state.ptr
	op.buf[1] = weights.ptr
	op.buf[2] = qkv.ptr
	op.buf[3] = out.ptr
	op.n[0] = C.uint64_t(channels)
	op.n[1] = C.uint64_t(kernel)
	return p.add(op)
}

func (p *Plan) AddGDN(state, q, k, v, alpha, beta, dt, a, out *Buffer, groups, heads, stateSize int) error {
	return p.AddGDNAt(state, q, 0, k, 0, v, 0, alpha, beta, dt, a, out, groups, heads, stateSize)
}

func (p *Plan) AddGDNAt(state, q *Buffer, qOff uint64, k *Buffer, kOff uint64, v *Buffer, vOff uint64, alpha, beta, dt, a, out *Buffer, groups, heads, stateSize int) error {
	if err := p.check(state, q, k, v, alpha, beta, dt, a, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GDN}
	op.off[1] = C.uint64_t(qOff)
	op.off[2] = C.uint64_t(kOff)
	op.off[3] = C.uint64_t(vOff)
	op.buf[0] = state.ptr
	op.buf[1] = q.ptr
	op.buf[2] = k.ptr
	op.buf[3] = v.ptr
	op.buf[4] = alpha.ptr
	op.buf[5] = beta.ptr
	op.buf[6] = dt.ptr
	op.buf[7] = a.ptr
	op.buf[8] = out.ptr
	op.n[0] = C.uint64_t(groups)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(stateSize)
	return p.add(op)
}

func (p *Plan) AddSiluMul(gate, up, out *Buffer, n int64) error {
	if err := p.check(gate, up, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_SILU_MUL}
	op.buf[0] = gate.ptr
	op.buf[1] = up.ptr
	op.buf[2] = out.ptr
	op.n[0] = C.uint64_t(n)
	return p.add(op)
}

func (p *Plan) Add(x, delta *Buffer, n int64) error {
	if err := p.check(x, delta); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ADD}
	op.buf[0] = x.ptr
	op.buf[1] = delta.ptr
	op.n[0] = C.uint64_t(n)
	return p.add(op)
}

func (p *Plan) AddCopy(dst, src *Buffer, n int64) error {
	if err := p.check(dst, src); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_COPY}
	op.buf[0] = dst.ptr
	op.buf[1] = src.ptr
	op.n[0] = C.uint64_t(n)
	return p.add(op)
}

func (p *Plan) AddScale(x *Buffer, n int64, scale float32) error {
	if err := p.check(x); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_SCALE}
	op.buf[0] = x.ptr
	op.n[0] = C.uint64_t(n)
	op.f[0] = C.float(scale)
	return p.add(op)
}

func (p *Plan) AddEmbedding(weights, out *Buffer, cols, rowBytes int64, dtype tensor.DType) error {
	if err := p.check(weights, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_EMBEDDING}
	op.buf[0] = weights.ptr
	op.buf[1] = out.ptr
	op.n[0] = C.uint64_t(cols)
	op.n[1] = C.uint64_t(rowBytes)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddKVWrite copies the current K/V vectors into caches at the dynamic
// position supplied to Execute.
func (p *Plan) AddKVWrite(k, v, kCache, vCache *Buffer, kvDim int64) error {
	return p.AddKVWriteAt(k, 0, v, 0, kCache, vCache, kvDim)
}

func (p *Plan) AddKVWriteAt(k *Buffer, kOff uint64, v *Buffer, vOff uint64, kCache, vCache *Buffer, kvDim int64) error {
	if err := p.check(k, v, kCache, vCache); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_KV_WRITE}
	op.buf[0] = k.ptr
	op.buf[1] = v.ptr
	op.buf[2] = kCache.ptr
	op.buf[3] = vCache.ptr
	op.off[0] = C.uint64_t(kOff)
	op.off[1] = C.uint64_t(vOff)
	op.n[0] = C.uint64_t(kvDim)
	return p.add(op)
}

// GemmPath selects the prefill GEMM implementation recorded in the plan.
type GemmPath uint32

const (
	// GemmAuto uses the compiled default for the dtype.
	GemmAuto GemmPath = 0
	// GemmBLAS is the rocBLAS FP16 baseline and requires an FP16 x buffer.
	GemmBLAS GemmPath = 1
	// GemmWMMA is the custom rocWMMA kernel with FP32 activations.
	GemmWMMA GemmPath = 2
)

// AddGemmPathAt appends a GEMM with an explicit implementation path and byte
// offsets. Path GemmBLAS requires dtype F16 and an FP16 input buffer. Path
// GemmWMMA accepts FP32 input and every supported weight dtype.
func (p *Plan) AddGemmPathAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType, path GemmPath) error {
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: GEMM dimensions must be positive")
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if path == GemmBLAS && dtype != tensor.F16 {
		return fmt.Errorf("hip: rocBLAS GEMM requires F16 weights, got %s", dtype)
	}
	if path > GemmWMMA {
		return fmt.Errorf("hip: unknown GEMM path %d", path)
	}
	if err := p.check(w, x, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMM}
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(dtype)
	op.u[1] = C.uint32_t(path)
	return p.add(op)
}

// AddRMSNormBatchAt appends a row-wise RMSNorm over rows rows of cols values.
func (p *Plan) AddRMSNormBatchAt(src *Buffer, srcOff uint64, weight *Buffer, dst *Buffer, dstOff uint64, rows, cols, srcStride, dstStride int64, eps float32) error {
	if rows <= 0 || cols <= 0 || srcStride < cols || dstStride < cols {
		return errors.New("hip: invalid batch RMSNorm dimensions")
	}
	if err := p.check(src, weight, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RMSNORM_BATCH}
	op.buf[0] = src.ptr
	op.buf[1] = weight.ptr
	op.buf[2] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[2] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(cols)
	op.n[1] = C.uint64_t(rows)
	op.n[2] = C.uint64_t(srcStride)
	op.n[3] = C.uint64_t(dstStride)
	op.f[0] = C.float(eps)
	return p.add(op)
}

// AddRMSNormHeadsBatchAt appends per-head RMSNorm over a [rows, heads, stride]
// layout. The weight vector is head-sized.
func (p *Plan) AddRMSNormHeadsBatchAt(src *Buffer, srcOff uint64, weight *Buffer, dst *Buffer, dstOff uint64, rows, heads, size, stride int64, eps float32) error {
	if rows <= 0 || heads <= 0 || size <= 0 || stride < size {
		return errors.New("hip: invalid batch head RMSNorm dimensions")
	}
	if err := p.check(src, weight, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RMSNORM_HEADS_BATCH}
	op.buf[0] = src.ptr
	op.buf[1] = weight.ptr
	op.buf[2] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[2] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(size)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(stride)
	op.n[3] = C.uint64_t(rows)
	op.f[0] = C.float(eps)
	return p.add(op)
}

// AddRoPEBatchAt appends RoPE to a [rows, heads, stride] chunk starting at the
// dynamic position supplied to Execute.
func (p *Plan) AddRoPEBatchAt(data *Buffer, dataOff uint64, rows, heads, stride, headDim, rotDim int, base float32, rowStride int64) error {
	if rows <= 0 || heads <= 0 || rotDim <= 0 || rotDim%2 != 0 || rotDim > headDim || stride < headDim || rowStride < int64(heads*stride) {
		return errors.New("hip: invalid batch RoPE dimensions")
	}
	if err := p.check(data); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ROPE_BATCH}
	op.buf[0] = data.ptr
	op.off[0] = C.uint64_t(dataOff)
	op.n[0] = C.uint64_t(heads)
	op.n[1] = C.uint64_t(stride)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(rotDim)
	op.n[4] = C.uint64_t(rows)
	op.n[5] = C.uint64_t(rowStride)
	op.f[0] = C.float(base)
	return p.add(op)
}

// AddKVWriteBatchAt copies rows K/V rows into the caches starting at the
// dynamic position supplied to Execute. capacity bounds the cache.
func (p *Plan) AddKVWriteBatchAt(k *Buffer, kOff uint64, v *Buffer, vOff uint64, kCache, vCache *Buffer, kvDim, rows, capacity int64) error {
	if kvDim <= 0 || rows <= 0 || capacity < rows {
		return errors.New("hip: invalid batch KV write dimensions")
	}
	if err := p.check(k, v, kCache, vCache); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_KV_WRITE_BATCH}
	op.buf[0] = k.ptr
	op.buf[1] = v.ptr
	op.buf[2] = kCache.ptr
	op.buf[3] = vCache.ptr
	op.off[0] = C.uint64_t(kOff)
	op.off[1] = C.uint64_t(vOff)
	op.n[0] = C.uint64_t(kvDim)
	op.n[1] = C.uint64_t(rows)
	op.u[0] = C.uint32_t(capacity)
	return p.add(op)
}

// AddAttentionPrefillAt appends causal tiled attention for a prompt chunk.
func (p *Plan) AddAttentionPrefillAt(q *Buffer, qOff uint64, kCache, vCache, out *Buffer, outOff uint64, kvHeads, heads, headDim, kvDim, rows, capacity int, scale float32) error {
	if rows <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 || kvDim != kvHeads*headDim || capacity < rows || headDim > 256 {
		return errors.New("hip: invalid prefill attention dimensions")
	}
	if err := p.check(q, kCache, vCache, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ATTENTION_PREFILL}
	op.buf[0] = q.ptr
	op.buf[1] = kCache.ptr
	op.buf[2] = vCache.ptr
	op.buf[3] = out.ptr
	op.off[0] = C.uint64_t(qOff)
	op.off[3] = C.uint64_t(outOff)
	op.n[0] = C.uint64_t(kvHeads)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(kvDim)
	op.n[4] = C.uint64_t(rows)
	op.u[0] = C.uint32_t(capacity)
	op.f[0] = C.float(scale)
	return p.add(op)
}

// RecurrentChunkArgs groups the per-layer tensors used by a gated delta-net
// prompt chunk.
type RecurrentChunkArgs struct {
	ConvState   *Buffer
	State       *Buffer
	QKV         *Buffer
	QKVOff      uint64
	Z           *Buffer
	ZOff        uint64
	Alpha       *Buffer
	Beta        *Buffer
	ConvWeights *Buffer
	DT          *Buffer
	A           *Buffer
	Norm        *Buffer
	Conv        *Buffer
	Out         *Buffer
	OutOff      uint64
	Rows        int64
	Groups      int64
	Heads       int64
	StateSize   int64
	ConvDim     int64
	Inner       int64
	Kernel      int64
	Eps         float32
	QScale      float32
}

// AddRecurrentChunkAt appends the fused convolution, gated delta-net state
// update, L2 normalization, and gated RMSNorm for one chunk. Rows must be
// positive and StateSize must be a power of two no larger than 1024.
func (p *Plan) AddRecurrentChunkAt(a RecurrentChunkArgs) error {
	if a.Rows <= 0 || a.Groups <= 0 || a.Heads <= 0 || a.StateSize <= 0 || a.Kernel < 2 ||
		a.Inner != a.Heads*a.StateSize || a.ConvDim != 2*a.Groups*a.StateSize+a.Inner ||
		a.StateSize&(a.StateSize-1) != 0 || a.StateSize > 1024 {
		return errors.New("hip: invalid recurrent chunk dimensions")
	}
	if a.ConvState == nil || a.State == nil || a.QKV == nil || a.Z == nil || a.Alpha == nil ||
		a.Beta == nil || a.ConvWeights == nil || a.DT == nil || a.A == nil || a.Norm == nil ||
		a.Conv == nil || a.Out == nil {
		return errors.New("hip: recurrent chunk buffer is nil")
	}
	if err := p.check(a.ConvState, a.State, a.QKV, a.Z, a.Alpha, a.Beta, a.ConvWeights, a.DT, a.A, a.Norm, a.Conv, a.Out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RECURRENT_CHUNK}
	op.buf[0] = a.ConvState.ptr
	op.buf[1] = a.State.ptr
	op.buf[2] = a.QKV.ptr
	op.buf[3] = a.Z.ptr
	op.buf[4] = a.Alpha.ptr
	op.buf[5] = a.Beta.ptr
	op.buf[6] = a.ConvWeights.ptr
	op.buf[7] = a.DT.ptr
	op.buf[8] = a.A.ptr
	op.buf[9] = a.Norm.ptr
	op.buf[10] = a.Out.ptr
	op.buf[11] = a.Conv.ptr
	op.off[2] = C.uint64_t(a.QKVOff)
	op.off[3] = C.uint64_t(a.ZOff)
	op.off[10] = C.uint64_t(a.OutOff)
	op.n[0] = C.uint64_t(a.Rows)
	op.n[1] = C.uint64_t(a.Groups)
	op.n[2] = C.uint64_t(a.Heads)
	op.n[3] = C.uint64_t(a.StateSize)
	op.n[4] = C.uint64_t(a.ConvDim)
	op.n[5] = C.uint64_t(a.Inner)
	op.u[0] = C.uint32_t(a.Kernel)
	op.f[0] = C.float(a.Eps)
	op.f[1] = C.float(a.QScale)
	return p.add(op)
}

// AddEmbeddingBatchAt appends token-table row gathers for a prompt chunk.
func (p *Plan) AddEmbeddingBatchAt(weights, out, tokens *Buffer, outOff uint64, rows, cols, rowBytes, vocab int64, dtype tensor.DType) error {
	if rows <= 0 || cols <= 0 || rowBytes <= 0 || vocab <= 0 {
		return errors.New("hip: invalid batch embedding dimensions")
	}
	if want, err := dtype.RowBytes(cols); err != nil || int64(want) != rowBytes {
		return fmt.Errorf("hip: embedding row layout does not match dtype %s", dtype)
	}
	if err := p.check(weights, out, tokens); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_EMBEDDING_BATCH}
	op.buf[0] = weights.ptr
	op.buf[1] = out.ptr
	op.buf[2] = tokens.ptr
	op.off[1] = C.uint64_t(outOff)
	op.n[0] = C.uint64_t(cols)
	op.n[1] = C.uint64_t(rows)
	op.n[2] = C.uint64_t(rowBytes)
	op.n[3] = C.uint64_t(vocab)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddCastF16At appends an FP32-to-FP16 conversion of n contiguous elements.
func (p *Plan) AddCastF16At(src *Buffer, srcOff uint64, dst *Buffer, dstOff uint64, n int64) error {
	if n <= 0 {
		return errors.New("hip: cast element count must be positive")
	}
	if err := p.check(src, dst); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_CAST_F16}
	op.buf[0] = src.ptr
	op.buf[1] = dst.ptr
	op.off[0] = C.uint64_t(srcOff)
	op.off[1] = C.uint64_t(dstOff)
	op.n[0] = C.uint64_t(n)
	return p.add(op)
}

// Execute submits the plan. token and position are dynamic values consumed by
// embedding and position-dependent kernels. runTail selects output-tail ops.
func (p *Plan) Execute(stream *Stream, token int32, position uint32, runTail bool, event *Event) error {
	if p == nil || p.ptr == nil {
		return errors.New("hip: plan is closed")
	}
	if stream == nil || stream.closed {
		return errors.New("hip: stream is closed")
	}
	tail := C.int(0)
	if runTail {
		tail = 1
	}
	var ev *C.yuuka_event
	if event != nil {
		ev = event.ptr
	}
	return status(C.yuuka_plan_execute(p.ptr, stream.ptr, C.int32_t(token), C.uint32_t(position), tail, ev), "plan_execute")
}

// OpTime is one profiled operation in plan order.
type OpTime struct {
	Code  uint32
	Nanos uint64
}

// ExecuteProfiled runs the plan with HIP events around every operation and
// leaks no events. It synchronizes once before returning, so it is a
// diagnostic path rather than a hot-path submission.
func (p *Plan) ExecuteProfiled(stream *Stream, token int32, position uint32, runTail bool, event *Event) ([]OpTime, error) {
	if p == nil || p.ptr == nil {
		return nil, errors.New("hip: plan is closed")
	}
	if stream == nil || stream.closed {
		return nil, errors.New("hip: stream is closed")
	}
	tail := C.int(0)
	if runTail {
		tail = 1
	}
	var ev *C.yuuka_event
	if event != nil {
		ev = event.ptr
	}
	codes := make([]C.uint32_t, p.ops)
	times := make([]C.uint64_t, p.ops)
	var codesPtr *C.uint32_t
	var timesPtr *C.uint64_t
	if len(codes) != 0 {
		codesPtr = &codes[0]
		timesPtr = &times[0]
	}
	count := C.uint64_t(0)
	err := status(C.yuuka_plan_execute_profiled(
		p.ptr, stream.ptr, C.int32_t(token), C.uint32_t(position), tail,
		codesPtr, timesPtr, C.uint64_t(len(codes)), &count, ev), "plan_execute_profiled")
	runtime.KeepAlive(codes)
	runtime.KeepAlive(times)
	if err != nil {
		return nil, err
	}
	out := make([]OpTime, int(count))
	for i := range out {
		out[i] = OpTime{Code: uint32(codes[i]), Nanos: uint64(times[i])}
	}
	return out, nil
}

// Close releases the plan. It is idempotent.
func (p *Plan) Close() error {
	if p == nil || p.ptr == nil {
		return nil
	}
	C.yuuka_plan_destroy(p.ptr)
	p.ptr = nil
	return nil
}

// Ops returns the number of operations in the plan.
func (p *Plan) Ops() int { return p.ops }
