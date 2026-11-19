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

// ErrUnsupported reports a capability limit in a plan operation. Callers can
// distinguish it from malformed dimensions or buffer ranges with errors.Is.
var ErrUnsupported = errors.New("hip: unsupported")

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

// RequiredABI is the shared-library ABI this package was compiled against.
const RequiredABI = 10

// Open opens a HIP device by index after checking the library ABI version.
func Open(index int) (*Device, error) {
	if got := ABIVersion(); got != RequiredABI {
		return nil, fmt.Errorf("hip: shared library ABI %d, want %d", got, RequiredABI)
	}
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

// ID returns a process-local identity for the allocation. Diagnostic tooling
// uses it to match a planned operation slot (OpInfo.Bufs) to the buffer it
// reads; the value must not be dereferenced and becomes meaningless once the
// buffer is freed.
func (b *Buffer) ID() uintptr {
	if b == nil || b.ptr == nil {
		return 0
	}
	return uintptr(unsafe.Pointer(b.ptr))
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

// CopyBuffer submits an asynchronous device-to-device copy.
func (d *Device) CopyBuffer(stream *Stream, dst *Buffer, dstOff uint64, src *Buffer, srcOff uint64, bytes uint64, event *Event) error {
	if d == nil || d.closed {
		return errors.New("hip: device is closed")
	}
	if stream == nil || stream.closed {
		return errors.New("hip: stream is closed")
	}
	if err := d.validate(dst, dstOff+bytes); err != nil {
		return err
	}
	if err := d.validate(src, srcOff+bytes); err != nil {
		return err
	}
	var ev *C.yuuka_event
	if event != nil {
		ev = event.ptr
	}
	return status(C.yuuka_buffer_copy(stream.ptr, dst.ptr, C.uint64_t(dstOff),
		src.ptr, C.uint64_t(srcOff), C.uint64_t(bytes), ev), "buffer_copy")
}

// SetDynamic uploads the token and position consumed by a captured graph.
func (d *Device) SetDynamic(stream *Stream, dynamic *Buffer, token int32, position uint32) error {
	if d == nil || d.closed {
		return errors.New("hip: device is closed")
	}
	if stream == nil || stream.closed {
		return errors.New("hip: stream is closed")
	}
	if dynamic == nil || dynamic.closed {
		return errors.New("hip: dynamic buffer is closed")
	}
	if dynamic.bytes < 8 {
		return errors.New("hip: dynamic buffer must hold two int32 values")
	}
	return status(C.yuuka_dynamic_set(stream.ptr, dynamic.ptr,
		C.int32_t(token), C.uint32_t(position)), "dynamic_set")
}

// DownloadFloats copies device bytes into a float32 slice.
func (d *Device) DownloadFloats(dst []float32, src *Buffer) error {
	if len(dst) == 0 {
		return errors.New("hip: download is empty")
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*4)
	return d.Download(bytes, src)
}

// DownloadFloatsAt copies len(dst) float32 values starting at a byte offset.
func (d *Device) DownloadFloatsAt(dst []float32, src *Buffer, offset uint64) error {
	if len(dst) == 0 {
		return errors.New("hip: download is empty")
	}
	if src == nil || src.closed {
		return errors.New("hip: download source is closed")
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*4)
	return status(C.yuuka_memcpy_d2h_at(unsafe.Pointer(&bytes[0]), src.ptr,
		C.uint64_t(offset), C.uint64_t(len(bytes))), "memcpy_d2h_at")
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
	message := C.GoString(C.yuuka_last_error())
	if st == C.YUUKA_ERR_UNSUPPORTED {
		return fmt.Errorf("%w: hip: %s: %s", ErrUnsupported, what, message)
	}
	return fmt.Errorf("hip: %s: %s", what, message)
}

// Plan batches generic device operations into one submission. It keeps a
// Go-side shadow of the operations it added so diagnostics can name them.
type Plan struct {
	ptr    *C.yuuka_plan
	device *Device
	tail   bool
	ops    []OpInfo
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
	info := OpInfo{
		Index: len(p.ops),
		Code:  uint32(op.code),
		Flags: uint32(op.flags),
		Roles: opRoles(uint32(op.code)),
	}
	for i := range info.N {
		info.N[i] = int64(op.n[i])
	}
	for i := range info.U {
		info.U[i] = uint32(op.u[i])
	}
	for i := range info.Bufs {
		if op.buf[i] != nil {
			info.Bufs[i] = uintptr(unsafe.Pointer(op.buf[i]))
		}
	}
	p.ops = append(p.ops, info)
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
	if !dtype.HIPSupported() {
		return fmt.Errorf("hip: GEMV dtype %s is not supported", dtype)
	}
	if _, err := dtype.RowBytes(nIn); err != nil {
		return fmt.Errorf("hip: GEMV shape: %w", err)
	}
	return nil
}

// checkMoEDType is the expert-tensor form of checkGemvDType. The MoE kernels
// decode the IQ2/IQ3 codebooks as well, so this accepts a wider dtype set while
// keeping the same row-width check; every other kernel family keeps rejecting
// those dtypes.
func checkMoEDType(dtype tensor.DType, nIn int64) error {
	if !dtype.HIPMoESupported() {
		return fmt.Errorf("hip: MoE dtype %s is not supported", dtype)
	}
	if _, err := dtype.RowBytes(nIn); err != nil {
		return fmt.Errorf("hip: MoE shape: %w", err)
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

// AddConvL2Norm appends the fused short convolution and L2 norm of a gated
// delta-net layer: out receives the SiLU of the convolution for every channel,
// and then its first qkChannels values are L2 normalized per group of stateSize,
// with qScale applied to the first qChannels of them. The convolution ring state
// in convState is advanced for every channel. channels and qkChannels must be
// multiples of 256 and stateSize must be 128; other shapes are rejected by the
// bridge with YUUKA_ERR_UNSUPPORTED.
func (p *Plan) AddConvL2Norm(convState, weights, qkv, out *Buffer, channels, kernel, qkChannels, qChannels, stateSize int, eps, qScale float32) error {
	if err := p.check(convState, weights, qkv, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_CONV_L2NORM}
	op.buf[0] = convState.ptr
	op.buf[1] = weights.ptr
	op.buf[2] = qkv.ptr
	op.buf[3] = out.ptr
	op.n[0] = C.uint64_t(channels)
	op.n[1] = C.uint64_t(kernel)
	op.n[2] = C.uint64_t(qkChannels)
	op.n[3] = C.uint64_t(qChannels)
	op.n[4] = C.uint64_t(stateSize)
	op.f[0] = C.float(eps)
	op.f[1] = C.float(qScale)
	return p.add(op)
}

// AddGemvResidualAt appends the streaming matvec with the
// residual epilogue: out = W*x + residual. out may alias residual, which is how
// the decode plan ends a sub-layer in one launch instead of a matvec plus an ADD.
// The shape and dtype rules are those of AddLMHeadAt, and rows must be 1..16.
func (p *Plan) AddGemvResidualAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, res *Buffer, resOff uint64, nIn, nOut, rows int64, dtype tensor.DType) error {
	if err := p.check(w, x, out, res); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMV_RESIDUAL}
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.buf[3] = res.ptr
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.off[3] = C.uint64_t(resOff)
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(rows)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// Epilogues of AddGemvPairAt.
const (
	// PairWriteBoth writes both projection rows.
	PairWriteBoth = 0
	// PairSwiglu writes silu(A*x) * (B*x) into the first output and requires equal
	// output widths.
	PairSwiglu = 1
)

// AddGemvPairAt appends two single-row matvecs over one activation row as a
// single launch. The two matrices may differ in packed dtype and
// width, which is what lets the plan merge projections that are independent but
// share their input row. With PairSwiglu the second output is unused and nil is
// allowed; with PairWriteBoth both outputs are written.
func (p *Plan) AddGemvPairAt(a *Buffer, aOff uint64, b *Buffer, bOff uint64, x *Buffer, xOff uint64, outA *Buffer, outAOff uint64, outB *Buffer, outBOff uint64, nIn, nOutA, nOutB int64, dtypeA, dtypeB tensor.DType, epilogue int) error {
	if err := p.check(a, b, x, outA); err != nil {
		return err
	}
	if outB == nil {
		if epilogue != PairSwiglu {
			return errors.New("hip: a gemv pair that writes both rows needs both outputs")
		}
	} else if err := p.check(outB); err != nil {
		return err
	}
	if epilogue != PairWriteBoth && epilogue != PairSwiglu {
		return fmt.Errorf("hip: unknown gemv pair epilogue %d", epilogue)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMV_PAIR}
	op.buf[0] = a.ptr
	op.buf[1] = b.ptr
	op.buf[2] = x.ptr
	op.buf[3] = outA.ptr
	if outB != nil {
		op.buf[4] = outB.ptr
	}
	op.off[0] = C.uint64_t(aOff)
	op.off[1] = C.uint64_t(bOff)
	op.off[2] = C.uint64_t(xOff)
	op.off[3] = C.uint64_t(outAOff)
	if outB != nil {
		op.off[4] = C.uint64_t(outBOff)
	}
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOutA)
	op.n[2] = C.uint64_t(nOutB)
	op.u[0] = C.uint32_t(dtypeA)
	op.u[1] = C.uint32_t(dtypeB)
	op.u[2] = C.uint32_t(epilogue)
	return p.add(op)
}

// AddRecurrentTokenAt appends one token of a gated delta-net layer with the
// head's state held in registers. out receives the gated RMSNorm of the head,
// not the raw delta-net result: the operator fuses the decay, both S^T products,
// the rank-1 update, and the normalization, because the normalization reduces
// over the same head's values. state is [heads,stateSize,stateSize] FP32 and is
// updated in place; q and k are indexed by group, v, out, and gate by head.
// stateSize must be 128.
func (p *Plan) AddRecurrentTokenAt(state, q *Buffer, qOff uint64, k *Buffer, kOff uint64, v *Buffer, vOff uint64, alpha, beta, dt, a, out, normWeight, gate *Buffer, groups, heads, stateSize int, eps float32) error {
	if err := p.check(state, q, k, v, alpha, beta, dt, a, out, normWeight, gate); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_RECURRENT_TOKEN}
	op.buf[0] = state.ptr
	op.buf[1] = q.ptr
	op.buf[2] = k.ptr
	op.buf[3] = v.ptr
	op.buf[4] = alpha.ptr
	op.buf[5] = beta.ptr
	op.buf[6] = dt.ptr
	op.buf[7] = a.ptr
	op.buf[8] = out.ptr
	op.buf[9] = normWeight.ptr
	op.buf[10] = gate.ptr
	op.off[1] = C.uint64_t(qOff)
	op.off[2] = C.uint64_t(kOff)
	op.off[3] = C.uint64_t(vOff)
	op.n[0] = C.uint64_t(groups)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(stateSize)
	op.f[0] = C.float(eps)
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
	return p.AddCopyAt(dst, 0, src, 0, n)
}

// AddCopyAt appends an FP32 copy at byte offsets. Ranges in the same buffer
// must be disjoint or identical.
func (p *Plan) AddCopyAt(dst *Buffer, dstOff uint64, src *Buffer, srcOff uint64, n int64) error {
	if err := p.check(dst, src); err != nil {
		return err
	}
	if n <= 0 || dstOff%4 != 0 || srcOff%4 != 0 {
		return errors.New("hip: invalid FP32 copy count or offset")
	}
	if dst.ptr == src.ptr && dstOff != srcOff && uint64(n) > (max(dstOff, srcOff)-min(dstOff, srcOff))/4 {
		return errors.New("hip: overlapping FP32 copy ranges")
	}
	op := C.yuuka_op{code: C.YUUKA_OP_COPY}
	op.buf[0] = dst.ptr
	op.buf[1] = src.ptr
	op.off[0] = C.uint64_t(dstOff)
	op.off[1] = C.uint64_t(srcOff)
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
	if cols <= 0 || rowBytes <= 0 {
		return fmt.Errorf("hip: invalid embedding dimensions cols=%d row_bytes=%d", cols, rowBytes)
	}
	if !dtype.HIPSupported() {
		return fmt.Errorf("hip: embedding dtype %s is not supported", dtype)
	}
	want, err := dtype.RowBytes(cols)
	if err != nil {
		return fmt.Errorf("hip: embedding row width %d: %w", cols, err)
	}
	if int64(want) != rowBytes {
		return fmt.Errorf("hip: embedding row bytes %d, want %d for %s", rowBytes, want, dtype)
	}
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
	// GemmWMMA is the custom rocWMMA kernel with FP32 activations.
	GemmWMMA GemmPath = 2
	// GemmMMQ is the int8 route: Q4_K/Q5_K/Q6_K weights, Q8_1 activations, and
	// exact int32 accumulation with FP32 weight scales. It needs the Q8_1
	// activation and scale buffers, so it is only appended with AddGemmMmqAt.
	GemmMMQ GemmPath = 3

	// gemmPathBLAS is the rocBLAS route. It is not part of the public path
	// selection: it needs an FP16 output scratch in addition to FP16 weights and
	// activations, so it is only appended through AddGemmBlasAt.
	gemmPathBLAS GemmPath = 1
)

// AddGemmPathAt appends a GEMM with an explicit implementation path and byte
// offsets. Only GemmAuto and GemmWMMA are accepted here; the rocBLAS route is
// appended with AddGemmBlasAt because it needs an FP16 output scratch, and the
// int8 route is appended with AddGemmMmqAt because it needs a Q8_1 activation
// pair.
func (p *Plan) AddGemmPathAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType, path GemmPath) error {
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: GEMM dimensions must be positive")
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if path != GemmAuto && path != GemmWMMA {
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

// AddGemmBlasAt appends a rocBLAS GEMM. w holds FP16 weights, x16 holds
// batch*nIn FP16 activations, and out16 is an FP16 scratch of nOut*batch
// elements that receives the product before it is cast into out. rocBLAS
// accumulates in FP32 and rounds each output once, so only the FP16 form of the
// result is lost.
func (p *Plan) AddGemmBlasAt(w *Buffer, wOff uint64, x16, out16, out *Buffer, outOff uint64, nIn, nOut, batch int64) error {
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: GEMM dimensions must be positive")
	}
	if err := p.check(w, x16, out16, out); err != nil {
		return err
	}
	want, err := tensor.F16.RowBytes(nIn)
	if err != nil {
		return fmt.Errorf("hip: rocBLAS GEMM shape: %w", err)
	}
	if wOff > w.bytes || int64(want)*nOut > int64(w.bytes-wOff) {
		return fmt.Errorf("hip: rocBLAS GEMM weight buffer holds %d bytes, want %d",
			w.bytes-wOff, int64(want)*nOut)
	}
	if batch*nIn*2 > int64(x16.bytes) {
		return fmt.Errorf("hip: rocBLAS GEMM activation buffer holds %d bytes, want %d",
			x16.bytes, batch*nIn*2)
	}
	if nOut*batch*2 > int64(out16.bytes) {
		return fmt.Errorf("hip: rocBLAS GEMM scratch buffer holds %d bytes, want %d",
			out16.bytes, nOut*batch*2)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMM}
	op.off[0] = C.uint64_t(wOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = x16.ptr
	op.buf[2] = out.ptr
	op.buf[3] = out16.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(tensor.F16)
	op.u[1] = C.uint32_t(gemmPathBLAS)
	return p.add(op)
}

// AddQuantizeQ8_1At appends the Q8_1 activation quantization that feeds the
// int8 prefill GEMM. x holds batch*nIn FP32 values, qs receives batch*nIn int8
// values, and ds receives batch*(nIn/32) float2 block scale and block sum
// entries. nIn must be a multiple of 128.
func (p *Plan) AddQuantizeQ8_1At(x *Buffer, xOff uint64, qs, ds *Buffer, nIn, batch int64) error {
	if nIn <= 0 || batch <= 0 {
		return errors.New("hip: Q8_1 dimensions must be positive")
	}
	if nIn%128 != 0 {
		return fmt.Errorf("hip: Q8_1 width %d is not a multiple of 128", nIn)
	}
	if err := p.check(x, qs, ds); err != nil {
		return err
	}
	if xOff > x.bytes || uint64(nIn*batch)*4 > x.bytes-xOff {
		return fmt.Errorf("hip: Q8_1 source holds %d bytes, want %d",
			x.bytes-xOff, nIn*batch*4)
	}
	if nIn*batch > int64(qs.bytes) {
		return fmt.Errorf("hip: Q8_1 destination holds %d bytes, want %d",
			qs.bytes, nIn*batch)
	}
	if batch*(nIn/32)*8 > int64(ds.bytes) {
		return fmt.Errorf("hip: Q8_1 scale buffer holds %d bytes, want %d",
			ds.bytes, batch*(nIn/32)*8)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_QUANT_Q8_1}
	op.off[0] = C.uint64_t(xOff)
	op.buf[0] = x.ptr
	op.buf[1] = qs.ptr
	op.buf[2] = ds.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(batch)
	return p.add(op)
}

// AddSiluMulQuantQ8_1At writes the Q8_1 digits and scale/sum pairs for
// SiLU(gate)*up. The four offsets are byte offsets into the respective buffers.
func (p *Plan) AddSiluMulQuantQ8_1At(gate *Buffer, gateOff uint64, up *Buffer, upOff uint64,
	qs *Buffer, qsOff uint64, ds *Buffer, dsOff uint64, width, rows int64) error {
	if width <= 0 || rows <= 0 {
		return errors.New("hip: SwiGLU Q8_1 dimensions must be positive")
	}
	if err := p.check(gate, up, qs, ds); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_SILU_MUL_QUANT_Q8_1}
	op.buf[0], op.off[0] = gate.ptr, C.uint64_t(gateOff)
	op.buf[1], op.off[1] = up.ptr, C.uint64_t(upOff)
	op.buf[2], op.off[2] = qs.ptr, C.uint64_t(qsOff)
	op.buf[3], op.off[3] = ds.ptr, C.uint64_t(dsOff)
	op.n[0], op.n[1] = C.uint64_t(width), C.uint64_t(rows)
	return p.add(op)
}

// AddGemmMmqAt appends the int8 prefill GEMM: out = W * x with packed
// Q4_K/Q5_K/Q6_K weights, Q8_1 activations in qs, their float2 block scales in
// ds, and an FP32 result. nIn must be a multiple of 256.
func (p *Plan) AddGemmMmqAt(w *Buffer, wOff uint64, qs, ds *Buffer, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType) error {
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: GEMM dimensions must be positive")
	}
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
	default:
		return fmt.Errorf("hip: int8 GEMM supports K-quants only, got %s", dtype)
	}
	if nIn%256 != 0 {
		return fmt.Errorf("hip: int8 GEMM input width %d is not a multiple of 256", nIn)
	}
	if err := p.check(w, qs, ds, out); err != nil {
		return err
	}
	rowBytes, err := dtype.RowBytes(nIn)
	if err != nil {
		return fmt.Errorf("hip: int8 GEMM shape: %w", err)
	}
	if wOff > w.bytes || int64(rowBytes)*nOut > int64(w.bytes-wOff) {
		return fmt.Errorf("hip: int8 GEMM weight buffer holds %d bytes, want %d",
			w.bytes-wOff, int64(rowBytes)*nOut)
	}
	if batch*nIn > int64(qs.bytes) {
		return fmt.Errorf("hip: int8 GEMM activation buffer holds %d bytes, want %d",
			qs.bytes, batch*nIn)
	}
	if batch*(nIn/32)*8 > int64(ds.bytes) {
		return fmt.Errorf("hip: int8 GEMM scale buffer holds %d bytes, want %d",
			ds.bytes, batch*(nIn/32)*8)
	}
	if outOff > out.bytes || batch*nOut*4 > int64(out.bytes-outOff) {
		return fmt.Errorf("hip: int8 GEMM output buffer holds %d bytes, want %d",
			out.bytes-outOff, batch*nOut*4)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMM}
	op.off[0] = C.uint64_t(wOff)
	op.off[2] = C.uint64_t(outOff)
	op.buf[0] = w.ptr
	op.buf[1] = qs.ptr
	op.buf[2] = out.ptr
	op.buf[3] = ds.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(dtype)
	op.u[1] = C.uint32_t(GemmMMQ)
	return p.add(op)
}

// AddGemmSwigluAt appends one fused gate/up GEMM. Both weight buffers must
// have the same dtype and [n_out, n_in] shape, x is FP32, and out receives
// silu(gate*x) * (up*x) in FP32. The kernel does not materialize the two GEMM
// intermediates.
func (p *Plan) AddGemmSwigluAt(gate *Buffer, gateOff uint64, up *Buffer, upOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, batch int64, dtype tensor.DType) error {
	if nIn <= 0 || nOut <= 0 || batch <= 0 {
		return errors.New("hip: fused GEMM dimensions must be positive")
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if err := p.check(gate, up, x, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMM_SWIGLU}
	op.buf[0] = gate.ptr
	op.buf[1] = up.ptr
	op.buf[2] = x.ptr
	op.buf[3] = out.ptr
	op.off[0] = C.uint64_t(gateOff)
	op.off[1] = C.uint64_t(upOff)
	op.off[2] = C.uint64_t(xOff)
	op.off[3] = C.uint64_t(outOff)
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(batch)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// MaxLMHeadRows is the row-block cap of the streaming LM head kernel. The
// prefill tail uses one row, and speculative verification uses at most one row
// per draft token, which is the same cap.
const MaxLMHeadRows = 16

// AddLMHeadAt appends the streaming prefill LM head: out[rows, nOut] =
// W * x[rows, nIn] with packed weights that may be any dtype the GEMV family
// supports. Each weight element is read once and reused across rows, so this is
// the prefill form of the output projection; the decode path keeps AddGemv.
// nIn must be a multiple of 256 and rows must be in 1..MaxLMHeadRows. The
// weight and activation offsets must stay 16-byte aligned because the decoders
// read whole 128-bit groups of packed values and activations.
func (p *Plan) AddLMHeadAt(w *Buffer, wOff uint64, x *Buffer, xOff uint64, out *Buffer, outOff uint64, nIn, nOut, rows int64, dtype tensor.DType) error {
	if nIn <= 0 || nOut <= 0 {
		return errors.New("hip: LM head dimensions must be positive")
	}
	if rows <= 0 || rows > MaxLMHeadRows {
		return fmt.Errorf("hip: LM head row block %d is outside 1..%d", rows, MaxLMHeadRows)
	}
	if nIn%256 != 0 {
		return fmt.Errorf("hip: LM head input width %d is not a multiple of 256", nIn)
	}
	if wOff%16 != 0 || xOff%16 != 0 {
		return fmt.Errorf("hip: LM head offset w=%d x=%d is not 16-byte aligned",
			wOff, xOff)
	}
	if err := checkGemvDType(dtype, nIn); err != nil {
		return err
	}
	if err := p.check(w, x, out); err != nil {
		return err
	}
	rowBytes, err := dtype.RowBytes(nIn)
	if err != nil {
		return fmt.Errorf("hip: LM head shape: %w", err)
	}
	if wOff > w.bytes || int64(rowBytes)*nOut > int64(w.bytes-wOff) {
		return fmt.Errorf("hip: LM head weight buffer holds %d bytes, want %d",
			w.bytes-wOff, int64(rowBytes)*nOut)
	}
	if xOff > x.bytes || rows*nIn*4 > int64(x.bytes-xOff) {
		return fmt.Errorf("hip: LM head activation buffer holds %d bytes, want %d",
			x.bytes-xOff, rows*nIn*4)
	}
	if outOff > out.bytes || rows*nOut*4 > int64(out.bytes-outOff) {
		return fmt.Errorf("hip: LM head output buffer holds %d bytes, want %d",
			out.bytes-outOff, rows*nOut*4)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_LM_HEAD}
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	op.buf[2] = out.ptr
	op.off[0] = C.uint64_t(wOff)
	op.off[1] = C.uint64_t(xOff)
	op.off[2] = C.uint64_t(outOff)
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.n[2] = C.uint64_t(rows)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddGemvQ8_1At appends an int8 decode matvec: out = W * x with x already
// quantized to Q8_1 by AddQuantizeQ8_1At, which writes qs (the int8 digits) and
// ds (the float2 block scales). Only one row is served: speculative verification
// keeps the FP32 streaming kernel. Q4_K, Q5_K, Q6_K, and Q8_0 weights are
// supported, nIn must be a multiple of 256, and any other shape fails here
// rather than falling back silently.
func (p *Plan) AddGemvQ8_1At(w *Buffer, wOff uint64, qs, ds, out *Buffer, nIn, nOut int64, dtype tensor.DType) error {
	return p.addGemvQ8_1(w, wOff, qs, ds, out, nIn, nOut, dtype, false)
}

// AddGemvQ8_1FusedAt appends the same int8 decode matvec with the activation
// quantized inside the matvec block: x is the FP32 activation row, the block
// stages its digits in shared memory, and no separate quantization operation is
// submitted for the group. It computes the same product as AddGemvQ8_1At with the
// same quantizer contract, so the two are interchangeable per shape.
func (p *Plan) AddGemvQ8_1FusedAt(w *Buffer, wOff uint64, x, out *Buffer, nIn, nOut int64, dtype tensor.DType) error {
	return p.addGemvQ8_1(w, wOff, x, nil, out, nIn, nOut, dtype, true)
}

func (p *Plan) addGemvQ8_1(w *Buffer, wOff uint64, x, ds, out *Buffer, nIn, nOut int64, dtype tensor.DType, fused bool) error {
	if nIn <= 0 || nOut <= 0 {
		return errors.New("hip: int8 GEMV dimensions must be positive")
	}
	if nIn%256 != 0 {
		return fmt.Errorf("hip: int8 GEMV width %d is not a multiple of 256", nIn)
	}
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K, tensor.Q8_0:
	default:
		return fmt.Errorf("%w: hip: int8 GEMV dtype %s", ErrUnsupported, dtype)
	}
	rowBytes, err := dtype.RowBytes(nIn)
	if err != nil {
		return fmt.Errorf("hip: int8 GEMV shape: %w", err)
	}
	buffers := []*Buffer{w, x, out}
	if !fused {
		buffers = []*Buffer{w, x, ds, out}
	}
	if err := p.check(buffers...); err != nil {
		return err
	}
	if wOff%16 != 0 {
		return fmt.Errorf("hip: int8 GEMV weight offset %d is not 16-byte aligned", wOff)
	}
	if wOff > w.bytes || int64(rowBytes)*nOut > int64(w.bytes-wOff) {
		return fmt.Errorf("hip: int8 GEMV weight buffer holds %d bytes, want %d",
			w.bytes-wOff, int64(rowBytes)*nOut)
	}
	if fused {
		if int64(x.bytes) < nIn*4 {
			return fmt.Errorf("hip: int8 GEMV activation buffer holds %d bytes, want %d",
				x.bytes, nIn*4)
		}
	} else {
		if int64(x.bytes) < nIn {
			return fmt.Errorf("hip: int8 GEMV digit buffer holds %d bytes, want %d",
				x.bytes, nIn)
		}
		if int64(ds.bytes) < (nIn/32)*8 {
			return fmt.Errorf("hip: int8 GEMV scale buffer holds %d bytes, want %d",
				ds.bytes, (nIn/32)*8)
		}
	}
	if int64(out.bytes) < nOut*4 {
		return fmt.Errorf("hip: int8 GEMV output buffer holds %d bytes, want %d",
			out.bytes, nOut*4)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_GEMV_Q8_1}
	op.off[0] = C.uint64_t(wOff)
	op.buf[0] = w.ptr
	op.buf[1] = x.ptr
	if !fused {
		op.buf[2] = ds.ptr
	}
	op.buf[3] = out.ptr
	op.n[0] = C.uint64_t(nIn)
	op.n[1] = C.uint64_t(nOut)
	op.u[0] = C.uint32_t(dtype)
	if fused {
		op.u[1] = 1
	}
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
	return p.AddRoPEBatchOffsetAt(data, dataOff, rows, heads, stride, headDim, rotDim, base, rowStride, 0)
}

// AddRoPEBatchOffsetAt rotates the first row at the execution position plus
// displacement tokens. The remaining rows use consecutive positions.
func (p *Plan) AddRoPEBatchOffsetAt(data *Buffer, dataOff uint64, rows, heads, stride, headDim, rotDim int, base float32, rowStride int64, displacement uint32) error {
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
	op.u[0] = C.uint32_t(displacement)
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
	if rows <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 || kvDim != kvHeads*headDim || capacity < rows {
		return errors.New("hip: invalid prefill attention dimensions")
	}
	if headDim > 256 {
		return fmt.Errorf("hip: prefill attention head dimension %d exceeds 256: %w", headDim, ErrUnsupported)
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

// KVRowBytes returns the stored byte size of one KV row for dtype.
func KVRowBytes(dtype tensor.DType, kvDim int64) (uint64, error) {
	if kvDim <= 0 {
		return 0, fmt.Errorf("hip: invalid KV row width %d", kvDim)
	}
	switch dtype {
	case tensor.F32:
		return uint64(kvDim) * 4, nil
	case tensor.F16:
		return uint64(kvDim) * 2, nil
	case tensor.Q8_0:
		return uint64((kvDim+31)/32) * 34, nil
	default:
		return 0, fmt.Errorf("hip: unsupported KV dtype %s", dtype)
	}
}

func checkKVDType(dtype tensor.DType, kvDim int64) error {
	_, err := KVRowBytes(dtype, kvDim)
	return err
}

// AddKVWritePagedAt appends one K/V row write through a block table. capacity
// is the number of request positions and blockSize is the paged geometry. The
// pool is F32.
func (p *Plan) AddKVWritePagedAt(k, v, kPool, vPool, table *Buffer, kvDim, capacity int64, blockSize int) error {
	return p.AddKVWritePagedFormatAt(k, v, kPool, vPool, table, kvDim, capacity, blockSize, tensor.F32)
}

// AddKVWritePagedFormatAt appends one K/V row write with an explicit pool dtype.
func (p *Plan) AddKVWritePagedFormatAt(k, v, kPool, vPool, table *Buffer, kvDim, capacity int64, blockSize int, dtype tensor.DType) error {
	return p.addKVWritePaged(k, v, kPool, vPool, table, kvDim, 1, capacity, blockSize, dtype, C.YUUKA_OP_KV_WRITE_PAGED)
}

// AddKVWriteBatchPagedAt appends rows K/V writes through a block table. The
// pool is F32.
func (p *Plan) AddKVWriteBatchPagedAt(k *Buffer, kOff uint64, v *Buffer, vOff uint64, kPool, vPool, table *Buffer, kvDim, rows, capacity int64, blockSize int) error {
	return p.AddKVWriteBatchPagedFormatAt(k, kOff, v, vOff, kPool, vPool, table, kvDim, rows, capacity, blockSize, tensor.F32)
}

// AddKVWriteBatchPagedFormatAt appends rows K/V writes with an explicit pool
// dtype.
func (p *Plan) AddKVWriteBatchPagedFormatAt(k *Buffer, kOff uint64, v *Buffer, vOff uint64, kPool, vPool, table *Buffer, kvDim, rows, capacity int64, blockSize int, dtype tensor.DType) error {
	if kOff != 0 || vOff != 0 {
		return errors.New("hip: paged batch KV write does not take source offsets")
	}
	return p.addKVWritePaged(k, v, kPool, vPool, table, kvDim, rows, capacity, blockSize, dtype, C.YUUKA_OP_KV_WRITE_BATCH_PAGED)
}

func (p *Plan) addKVWritePaged(k, v, kPool, vPool, table *Buffer, kvDim, rows, capacity int64, blockSize int, dtype tensor.DType, code C.uint32_t) error {
	if kvDim <= 0 || rows <= 0 || capacity < rows || blockSize <= 0 {
		return errors.New("hip: invalid paged KV write dimensions")
	}
	if int64(blockSize) > capacity {
		return errors.New("hip: paged KV block size exceeds capacity")
	}
	if err := checkKVDType(dtype, kvDim); err != nil {
		return err
	}
	if err := p.check(k, v, kPool, vPool, table); err != nil {
		return err
	}
	op := C.yuuka_op{code: code}
	op.buf[0] = k.ptr
	op.buf[1] = v.ptr
	op.buf[2] = kPool.ptr
	op.buf[3] = vPool.ptr
	op.buf[4] = table.ptr
	op.n[0] = C.uint64_t(kvDim)
	if code == C.YUUKA_OP_KV_WRITE_BATCH_PAGED {
		op.n[1] = C.uint64_t(rows)
	}
	op.u[0] = C.uint32_t(blockSize)
	op.u[1] = C.uint32_t(capacity)
	op.u[2] = C.uint32_t(dtype)
	return p.add(op)
}

// AddAttentionPagedAt appends single-token attention through a block table.
// scoreStride is the per-head stride of the scores workspace (usually the
// maximum context); capacity bounds live positions. The workspace also holds the
// split-key partials, so it must be at least heads*ntiles*(headDim+2) floats with
// ntiles = ceil(scoreStride / tile); see decodeAttentionScratch in engine/gpu.
// The pool is F32.
func (p *Plan) AddAttentionPagedAt(q, kPool, vPool, out, scores, table *Buffer, kvHeads, heads, headDim, kvDim, scoreStride, capacity, blockSize int, scale float32) error {
	return p.AddAttentionPagedFormatAt(q, kPool, vPool, out, scores, table, kvHeads, heads, headDim, kvDim, scoreStride, capacity, blockSize, scale, tensor.F32)
}

// AddAttentionPagedFormatAt appends single-token attention with an explicit
// pool dtype.
func (p *Plan) AddAttentionPagedFormatAt(q, kPool, vPool, out, scores, table *Buffer, kvHeads, heads, headDim, kvDim, scoreStride, capacity, blockSize int, scale float32, dtype tensor.DType) error {
	return p.addAttentionPaged(q, kPool, vPool, out, scores, table, kvHeads, heads,
		headDim, kvDim, scoreStride, capacity, blockSize, scale, dtype, false)
}

// AddAttentionPagedDenseFormatAt is AddAttentionPagedFormatAt for models whose
// query row is heads*head_dim with no output gate.
func (p *Plan) AddAttentionPagedDenseFormatAt(q, kPool, vPool, out, scores, table *Buffer, kvHeads, heads, headDim, kvDim, scoreStride, capacity, blockSize int, scale float32, dtype tensor.DType) error {
	return p.addAttentionPaged(q, kPool, vPool, out, scores, table, kvHeads, heads,
		headDim, kvDim, scoreStride, capacity, blockSize, scale, dtype, true)
}

func (p *Plan) addAttentionPaged(q, kPool, vPool, out, scores, table *Buffer, kvHeads, heads, headDim, kvDim, scoreStride, capacity, blockSize int, scale float32, dtype tensor.DType, dense bool) error {
	if heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 ||
		kvDim != kvHeads*headDim || capacity <= 0 ||
		blockSize <= 0 || blockSize > capacity || scoreStride < capacity {
		return errors.New("hip: invalid paged attention dimensions")
	}
	if headDim > 256 {
		return fmt.Errorf("hip: paged attention head dimension %d exceeds 256: %w", headDim, ErrUnsupported)
	}
	if err := checkKVDType(dtype, int64(kvDim)); err != nil {
		return err
	}
	if err := p.check(q, kPool, vPool, out, scores, table); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ATTENTION_PAGED}
	op.buf[0] = q.ptr
	op.buf[1] = kPool.ptr
	op.buf[2] = vPool.ptr
	op.buf[3] = out.ptr
	op.buf[4] = scores.ptr
	op.buf[5] = table.ptr
	op.n[0] = C.uint64_t(kvHeads)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(kvDim)
	op.n[4] = C.uint64_t(scoreStride)
	op.u[0] = C.uint32_t(blockSize)
	op.u[1] = C.uint32_t(capacity)
	op.u[2] = C.uint32_t(dtype)
	if dense {
		op.u[3] = 1
	}
	op.f[0] = C.float(scale)
	return p.add(op)
}

// AddMoERoute appends top-k expert selection and assignment binning. logits is
// [rows, experts]; ids and weights are [rows, used]. counts is [experts] and
// the three assignment buffers are [experts, rows*used].
func (p *Plan) AddMoERoute(logits, ids, weights, counts, assignRows, assignSlots, assignWeights *Buffer, rows, experts, used int64) error {
	if rows <= 0 || experts <= 0 || used <= 0 || used > experts {
		return errors.New("hip: invalid MoE route dimensions")
	}
	if err := p.check(logits, ids, weights, counts, assignRows, assignSlots, assignWeights); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_ROUTE}
	op.buf[0] = logits.ptr
	op.buf[1] = ids.ptr
	op.buf[2] = weights.ptr
	op.buf[3] = counts.ptr
	op.buf[4] = assignRows.ptr
	op.buf[5] = assignSlots.ptr
	op.buf[6] = assignWeights.ptr
	op.n[0] = C.uint64_t(rows)
	op.n[1] = C.uint64_t(experts)
	op.n[2] = C.uint64_t(used)
	return p.add(op)
}

// AddMoEGateUpGrouped appends grouped gate/up expert projections. norm is
// [rows, hidden], counts and assignments come from AddMoERoute, and gateOut and
// upOut are [rows*used, ffn].
func (p *Plan) AddMoEGateUpGrouped(norm, gateExps, upExps, counts, assignRows, assignSlots, gateOut, upOut *Buffer, hidden, ffn, rows, experts, used int64, dtype tensor.DType) error {
	if hidden <= 0 || ffn <= 0 || rows <= 0 || experts <= 0 || used <= 0 || used > experts {
		return errors.New("hip: invalid MoE gate/up dimensions")
	}
	if err := checkMoEDType(dtype, hidden); err != nil {
		return err
	}
	if err := p.check(norm, gateExps, upExps, counts, assignRows, assignSlots, gateOut, upOut); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_GATE_UP}
	op.buf[0] = norm.ptr
	op.buf[1] = gateExps.ptr
	op.buf[2] = upExps.ptr
	op.buf[3] = counts.ptr
	op.buf[4] = assignRows.ptr
	op.buf[5] = assignSlots.ptr
	op.buf[6] = gateOut.ptr
	op.buf[7] = upOut.ptr
	op.n[0] = C.uint64_t(hidden)
	op.n[1] = C.uint64_t(ffn)
	op.n[2] = C.uint64_t(rows)
	op.n[3] = C.uint64_t(experts)
	op.n[4] = C.uint64_t(used)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddMoEDownGrouped appends grouped expert down projections. act is
// [rows*used, ffn] and out is [rows*used, hidden].
func (p *Plan) AddMoEDownGrouped(act, downExps, counts, assignRows, assignSlots, out *Buffer, ffn, hidden, rows, experts, used int64, dtype tensor.DType) error {
	if ffn <= 0 || hidden <= 0 || rows <= 0 || experts <= 0 || used <= 0 || used > experts {
		return errors.New("hip: invalid MoE down dimensions")
	}
	if err := checkMoEDType(dtype, ffn); err != nil {
		return err
	}
	if err := p.check(act, downExps, counts, assignRows, assignSlots, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_DOWN}
	op.buf[0] = act.ptr
	op.buf[1] = downExps.ptr
	op.buf[2] = counts.ptr
	op.buf[3] = assignRows.ptr
	op.buf[4] = assignSlots.ptr
	op.buf[5] = out.ptr
	op.n[0] = C.uint64_t(ffn)
	op.n[1] = C.uint64_t(hidden)
	op.n[2] = C.uint64_t(rows)
	op.n[3] = C.uint64_t(experts)
	op.n[4] = C.uint64_t(used)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddMoEReduceAdd appends weighted accumulation of per-slot expert outputs into
// the residual stream x. expertOut is [rows*used, hidden] and weights is
// [rows, used].
func (p *Plan) AddMoEReduceAdd(x, expertOut, weights *Buffer, hidden, rows, used int64) error {
	if hidden <= 0 || rows <= 0 || used <= 0 {
		return errors.New("hip: invalid MoE reduce dimensions")
	}
	if err := p.check(x, expertOut, weights); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_REDUCE_ADD}
	op.buf[0] = x.ptr
	op.buf[1] = expertOut.ptr
	op.buf[2] = weights.ptr
	op.n[0] = C.uint64_t(hidden)
	op.n[1] = C.uint64_t(rows)
	op.n[2] = C.uint64_t(used)
	return p.add(op)
}

// AddMoESharedAdd appends the sigmoid-gated shared expert accumulation:
// x[row,i] += sigmoid(dot(gate, norm[row])) * src[row,i]. gate is the FP32
// [hidden] shared-expert gate vector.
func (p *Plan) AddMoESharedAdd(x, src, gate, norm *Buffer, hidden, rows int64) error {
	if hidden <= 0 || rows <= 0 {
		return errors.New("hip: invalid MoE shared add dimensions")
	}
	if err := p.check(x, src, gate, norm); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_SHARED_ADD}
	op.buf[0] = x.ptr
	op.buf[1] = src.ptr
	op.buf[2] = gate.ptr
	op.buf[3] = norm.ptr
	op.n[0] = C.uint64_t(hidden)
	op.n[1] = C.uint64_t(rows)
	return p.add(op)
}

// AddMoEGateUpSelected appends the selected-expert gate and up projections:
// one block per (expert-FFN row, routing slot). ids is int32[rows*used] from
// AddMoERoute and act receives silu(gate)*up as [rows*used, ffn]. Both expert
// matrices must share dtype, which the kernel decodes with one instantiation.
func (p *Plan) AddMoEGateUpSelected(norm, ids, gateExps, upExps, act *Buffer, hidden, ffn, rows, used, experts int64, dtype tensor.DType) error {
	if hidden <= 0 || ffn <= 0 || rows <= 0 || experts <= 0 || used <= 0 || used > experts {
		return errors.New("hip: invalid MoE selected gate/up dimensions")
	}
	if err := checkMoEDType(dtype, hidden); err != nil {
		return err
	}
	if err := p.check(norm, ids, gateExps, upExps, act); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_GATE_UP_SELECTED}
	op.buf[0] = norm.ptr
	op.buf[1] = ids.ptr
	op.buf[2] = gateExps.ptr
	op.buf[3] = upExps.ptr
	op.buf[4] = act.ptr
	op.n[0] = C.uint64_t(hidden)
	op.n[1] = C.uint64_t(ffn)
	op.n[2] = C.uint64_t(rows)
	op.n[3] = C.uint64_t(used)
	op.n[4] = C.uint64_t(experts)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddMoEDownSelected appends the selected-expert down projection. act is
// [rows*used, ffn] and out receives [rows*used, hidden].
func (p *Plan) AddMoEDownSelected(act, ids, downExps, out *Buffer, ffn, hidden, rows, used, experts int64, dtype tensor.DType) error {
	if ffn <= 0 || hidden <= 0 || rows <= 0 || experts <= 0 || used <= 0 || used > experts {
		return errors.New("hip: invalid MoE selected down dimensions")
	}
	if err := checkMoEDType(dtype, ffn); err != nil {
		return err
	}
	if err := p.check(act, ids, downExps, out); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_MOE_DOWN_SELECTED}
	op.buf[0] = act.ptr
	op.buf[1] = ids.ptr
	op.buf[2] = downExps.ptr
	op.buf[3] = out.ptr
	op.n[0] = C.uint64_t(ffn)
	op.n[1] = C.uint64_t(hidden)
	op.n[2] = C.uint64_t(rows)
	op.n[3] = C.uint64_t(used)
	op.n[4] = C.uint64_t(experts)
	op.u[0] = C.uint32_t(dtype)
	return p.add(op)
}

// AddAttentionPrefillPagedAt appends causal tiled attention over a prompt chunk
// through a block table. The pool is F32.
func (p *Plan) AddAttentionPrefillPagedAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32) error {
	return p.AddAttentionPrefillPagedFormatAt(q, qOff, kPool, vPool, out, outOff, table, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, tensor.F32)
}

// AddAttentionPrefillPagedBlasFormatAt selects the experimental FP16 KV
// matrix-multiplication attention path. It uses the ordinary plan validator,
// with buf[5] carrying the scratch and the BLAS flag requesting rocBLAS.
func (p *Plan) AddAttentionPrefillPagedBlasFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table, scratch *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32) error {
	return p.AddAttentionPrefillPagedBlasOffsetFormatAt(q, qOff, kPool, vPool, out, outOff, table, scratch,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, 0)
}

// AddAttentionPrefillPagedBlasOffsetFormatAt selects the experimental FP16 KV
// attention path starting at the execution position plus displacement tokens.
func (p *Plan) AddAttentionPrefillPagedBlasOffsetFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table, scratch *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, displacement uint32) error {
	if rows <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 ||
		heads%kvHeads != 0 || kvDim != kvHeads*headDim || capacity < rows ||
		blockSize <= 0 || blockSize > capacity || headDim > 256 {
		return errors.New("hip: invalid BLAS attention shape")
	}
	if err := p.check(q, kPool, vPool, out, table, scratch); err != nil {
		return err
	}
	qElems := uint64(heads) * uint64(rows) * uint64(headDim)
	kvElems := uint64(kvHeads) * uint64(capacity) * uint64(headDim)
	scoreElems := uint64(heads) * uint64(rows) * uint64(capacity)
	want := 6*qElems + 4*kvElems + 6*scoreElems
	if scratch.bytes < want {
		return fmt.Errorf("hip: BLAS attention scratch %d < %d", scratch.bytes, want)
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ATTENTION_PREFILL_PAGED, flags: C.YUUKA_OP_FLAG_PREFILL_BLAS}
	op.buf[0], op.buf[1], op.buf[2], op.buf[3], op.buf[4], op.buf[5] =
		q.ptr, kPool.ptr, vPool.ptr, out.ptr, table.ptr, scratch.ptr
	op.off[0], op.off[3] = C.uint64_t(qOff), C.uint64_t(outOff)
	op.n[0], op.n[1], op.n[2], op.n[3], op.n[4] =
		C.uint64_t(kvHeads), C.uint64_t(heads), C.uint64_t(headDim), C.uint64_t(kvDim), C.uint64_t(rows)
	op.n[5] = C.uint64_t(displacement)
	op.u[0], op.u[1], op.u[2] = C.uint32_t(blockSize), C.uint32_t(capacity), C.uint32_t(tensor.F16)
	op.f[0] = C.float(scale)
	return p.add(op)
}

// AddAttentionPrefillPagedFormatAt appends causal tiled attention with an
// explicit pool dtype.
func (p *Plan) AddAttentionPrefillPagedFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType) error {
	return p.addAttentionPrefillPaged(q, qOff, kPool, vPool, out, outOff, table,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, dtype, false)
}

// AddAttentionPrefillPagedDenseFormatAt is
// AddAttentionPrefillPagedFormatAt for models whose query row is
// heads*head_dim with no output gate.
func (p *Plan) AddAttentionPrefillPagedDenseFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType) error {
	return p.addAttentionPrefillPaged(q, qOff, kPool, vPool, out, outOff, table,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, dtype, true)
}

// AddAttentionPrefillPagedTiledFormatAt is AddAttentionPrefillPagedFormatAt
// with the FP16 MMA kernel forced off, so a test can compare the two kernels on
// the same plan input.
func (p *Plan) AddAttentionPrefillPagedTiledFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType) error {
	return p.AddAttentionPrefillPagedOffsetFormatAt(q, qOff, kPool, vPool, out, outOff, table,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, dtype, 0, true)
}

// AddAttentionPrefillPagedOffsetFormatAt appends gated attention starting at
// execution position plus displacement. It preserves dispatch for the parent
// chunk of displacement+rows and can force the scalar tiled implementation.
func (p *Plan) AddAttentionPrefillPagedOffsetFormatAt(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType, displacement uint32, tiled bool) error {
	return p.addAttentionPrefillPagedFlag(q, qOff, kPool, vPool, out, outOff, table, nil,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, dtype, displacement, false, tiled)
}

// AddAttentionPrefillPagedSplitFormatAt appends gated F16 attention with
// scratch for up to four FP32 key ranges. The native launcher selects the
// split count using the position supplied to Execute.
func (p *Plan) AddAttentionPrefillPagedSplitFormatAt(q, kPool, vPool, out, table, partial *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32) error {
	if partial == nil {
		return errors.New("hip: split attention scratch is nil")
	}
	return p.addAttentionPrefillPagedFlag(q, 0, kPool, vPool, out, 0, table, partial,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, tensor.F16, 0, false, false)
}

func (p *Plan) addAttentionPrefillPaged(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType, dense bool) error {
	return p.addAttentionPrefillPagedFlag(q, qOff, kPool, vPool, out, outOff, table, nil,
		kvHeads, heads, headDim, kvDim, rows, capacity, blockSize, scale, dtype, 0, dense, false)
}

func (p *Plan) addAttentionPrefillPagedFlag(q *Buffer, qOff uint64, kPool, vPool, out *Buffer, outOff uint64, table, partial *Buffer, kvHeads, heads, headDim, kvDim, rows, capacity, blockSize int, scale float32, dtype tensor.DType, displacement uint32, dense, tiled bool) error {
	if rows <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 ||
		kvDim != kvHeads*headDim || capacity < rows ||
		blockSize <= 0 || blockSize > capacity {
		return errors.New("hip: invalid paged prefill attention dimensions")
	}
	if headDim > 256 {
		return fmt.Errorf("hip: paged prefill attention head dimension %d exceeds 256: %w", headDim, ErrUnsupported)
	}
	if err := checkKVDType(dtype, int64(kvDim)); err != nil {
		return err
	}
	if err := p.check(q, kPool, vPool, out, table); err != nil {
		return err
	}
	op := C.yuuka_op{code: C.YUUKA_OP_ATTENTION_PREFILL_PAGED}
	op.buf[0] = q.ptr
	op.buf[1] = kPool.ptr
	op.buf[2] = vPool.ptr
	op.buf[3] = out.ptr
	op.buf[4] = table.ptr
	if partial != nil {
		if err := p.check(partial); err != nil {
			return err
		}
		op.buf[6] = partial.ptr
	}
	op.off[0] = C.uint64_t(qOff)
	op.off[3] = C.uint64_t(outOff)
	op.n[0] = C.uint64_t(kvHeads)
	op.n[1] = C.uint64_t(heads)
	op.n[2] = C.uint64_t(headDim)
	op.n[3] = C.uint64_t(kvDim)
	op.n[4] = C.uint64_t(rows)
	op.n[5] = C.uint64_t(displacement)
	op.u[0] = C.uint32_t(blockSize)
	op.u[1] = C.uint32_t(capacity)
	op.u[2] = C.uint32_t(dtype)
	if dense {
		op.u[3] = 1
	}
	if tiled {
		op.flags |= C.YUUKA_OP_FLAG_PREFILL_ATTENTION_TILED
	}
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
	// Scratch holds two chunk x chunk matrices per 32-token chunk and head for
	// the chunkwise path. It is mutually exclusive with Prepared.
	Scratch *Buffer
	// Prepared holds FP32 Q, K, decay and beta during a 128-state recurrence.
	// With no scratch buffers, preparation runs inside the per-token core.
	Prepared *Buffer

	Rows      int64
	Groups    int64
	Heads     int64
	StateSize int64
	ConvDim   int64
	Inner     int64
	Kernel    int64
	Eps       float32
	QScale    float32
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
	if a.Scratch != nil {
		// Two chunk x chunk matrices per 32-token chunk of every head.
		chunks := (a.Rows + 31) / 32
		if got := int64(a.Scratch.Bytes() / 4); got < chunks*2*32*32*a.Heads {
			return fmt.Errorf("hip: recurrent chunk scratch holds %d elements, want %d",
				got, chunks*2*32*32*a.Heads)
		}
	}
	if a.Prepared != nil {
		if err := p.check(a.Prepared); err != nil {
			return err
		}
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
	if a.Scratch != nil {
		op.buf[12] = a.Scratch.ptr
	}
	if a.Prepared != nil {
		op.buf[13] = a.Prepared.ptr
	}
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
	if !dtype.HIPSupported() {
		return fmt.Errorf("hip: batch embedding dtype %s is not supported", dtype)
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
	codes := make([]C.uint32_t, len(p.ops))
	times := make([]C.uint64_t, len(p.ops))
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

// Graph is a captured plan execution with device-resident dynamic parameters.
// It is owned by the caller and must be closed before the plan. Replay is valid
// only while every buffer address captured into the graph is alive.
type Graph struct {
	ptr    *C.yuuka_graph
	closed bool
}

// CaptureGraph captures and instantiates the plan's non-tail or tail ops. Ops
// without dynamic-parameter kernels make capture fail with ErrUnsupported.
func (p *Plan) CaptureGraph(stream *Stream, dynamic *Buffer, runTail bool) (*Graph, error) {
	if p == nil || p.ptr == nil {
		return nil, errors.New("hip: plan is closed")
	}
	if stream == nil || stream.closed {
		return nil, errors.New("hip: stream is closed")
	}
	if dynamic == nil || dynamic.closed {
		return nil, errors.New("hip: dynamic buffer is closed")
	}
	tail := C.int(0)
	if runTail {
		tail = 1
	}
	var ptr *C.yuuka_graph
	if err := status(C.yuuka_graph_create(p.ptr, stream.ptr, dynamic.ptr, tail, &ptr),
		"graph_create"); err != nil {
		return nil, err
	}
	return &Graph{ptr: ptr}, nil
}

// Replay submits a captured graph and optionally records a completion event.
func (g *Graph) Replay(stream *Stream, event *Event) error {
	if g == nil || g.ptr == nil || g.closed {
		return errors.New("hip: graph is closed")
	}
	if stream == nil || stream.closed {
		return errors.New("hip: stream is closed")
	}
	var ev *C.yuuka_event
	if event != nil {
		ev = event.ptr
	}
	return status(C.yuuka_graph_replay(g.ptr, stream.ptr, ev), "graph_replay")
}

// Close destroys the captured graph. It is idempotent.
func (g *Graph) Close() error {
	if g == nil || g.closed {
		return nil
	}
	C.yuuka_graph_destroy(g.ptr)
	g.ptr = nil
	g.closed = true
	return nil
}

// Ops returns a copy of the plan's operation shadow in submission order. The
// slice has one entry per operation the plan holds, including tail operations
// that an Execute call can skip; ExecuteProfiled returns times only for the
// operations it executed, so a call with runTail=false lines its results up
// with a prefix of this slice. The copy keeps the caller from mutating the
// plan's record; it is a diagnostic path, not a hot-path accessor.
func (p *Plan) Ops() []OpInfo {
	if p == nil {
		return nil
	}
	out := make([]OpInfo, len(p.ops))
	copy(out, p.ops)
	return out
}

// OpCount returns the number of operations in the plan.
func (p *Plan) OpCount() int { return len(p.ops) }
