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

// AddGemv appends out = W * x for a packed weight matrix.
func (p *Plan) AddGemv(w, x, out *Buffer, nIn, nOut int64, dtype tensor.DType) error {
	return p.AddGemvAt(w, 0, x, 0, out, 0, nIn, nOut, dtype)
}

// AddGemvAt appends a GEMV with byte offsets applied to each buffer.
func (p *Plan) AddGemvAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut int64, dtype tensor.DType) error {
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
