//go:build hip && linux

package gpu

import (
	"fmt"
	"sync"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

type weight struct {
	buf   *hip.Buffer
	dtype tensor.DType
}

type activations struct {
	x       *hip.Buffer
	norm    *hip.Buffer
	q       *hip.Buffer
	k       *hip.Buffer
	v       *hip.Buffer
	attn    *hip.Buffer
	tmp     *hip.Buffer
	qkv     *hip.Buffer
	z       *hip.Buffer
	beta    *hip.Buffer
	alpha   *hip.Buffer
	conv    *hip.Buffer
	gdn     *hip.Buffer
	gdnNorm *hip.Buffer
	ffnGate *hip.Buffer
	ffnUp   *hip.Buffer
	logits  *hip.Buffer
	scores  *hip.Buffer
}

type prefillActivations struct {
	x       *hip.Buffer
	norm    *hip.Buffer
	q       *hip.Buffer
	k       *hip.Buffer
	v       *hip.Buffer
	attn    *hip.Buffer
	tmp     *hip.Buffer
	qkv     *hip.Buffer
	z       *hip.Buffer
	beta    *hip.Buffer
	alpha   *hip.Buffer
	conv    *hip.Buffer
	gdnNorm *hip.Buffer
	ffnGate *hip.Buffer
	ffnUp   *hip.Buffer
	xF16    *hip.Buffer
	tokens  *hip.Buffer
}

// Options configures a shared GPU model and its KV pool.
type Options struct {
	Blocks        int
	BlockSize     int
	PrefillChunk  int
	EnableGraph   bool
	GraphMinSteps int
	// KVType selects the full-attention KV pool storage. Zero is F32. F16 and
	// Q8_0 are opt-in Phase 5 formats.
	KVType tensor.DType
}

const maxPrefillChunk = 4096

// Model owns device weights, shared decode/prefill activations, the KV block
// pools, and the block allocator. Sessions bind request block tables to the
// pool. Model execution is serialized by stepMu, so sessions must be stepped
// one at a time; the serving scheduler guarantees that ordering.
type Model struct {
	device *hip.Device
	stream *hip.Stream
	event  *hip.Event
	config *qwen35.Config
	alloc  *memory.Allocator

	weights map[string]weight
	act     activations
	prefill prefillActivations

	poolK     []*hip.Buffer
	poolV     []*hip.Buffer
	blockSize int
	blocks    int
	capacity  int
	kvType    tensor.DType
	kvRow     uint64

	prefillRows int
	prefillGen  int64
	prefillAll  []*hip.Buffer

	specLogits *hip.Buffer

	stepMu sync.Mutex
	all    []*hip.Buffer
	closed bool
}

// Open uploads weights and allocates the device pool shared by all sessions.
func Open(reader *gguf.Reader, model *qwen35.Model, opts Options) (*Model, error) {
	if model == nil || reader == nil {
		return nil, fmt.Errorf("gpu: model and reader are required")
	}
	if opts.Blocks <= 0 || opts.BlockSize <= 0 {
		return nil, fmt.Errorf("gpu: KV pool needs positive blocks and block size")
	}
	if err := validateModelSupport(model); err != nil {
		return nil, err
	}
	kvRow, err := hip.KVRowBytes(opts.KVType, int64(model.Config.KVHeadCount*model.Config.HeadDim))
	if err != nil {
		return nil, err
	}
	alloc, err := memory.NewAllocator(opts.Blocks, opts.BlockSize)
	if err != nil {
		return nil, err
	}
	device, err := hip.Open(0)
	if err != nil {
		return nil, err
	}
	stream, err := device.NewStream()
	if err != nil {
		_ = device.Close()
		return nil, err
	}
	event, err := device.NewEvent()
	if err != nil {
		_ = stream.Close()
		_ = device.Close()
		return nil, err
	}
	m := &Model{
		device:    device,
		stream:    stream,
		event:     event,
		config:    model.Config,
		alloc:     alloc,
		weights:   make(map[string]weight, len(reader.Tensors())),
		blockSize: opts.BlockSize,
		blocks:    opts.Blocks,
		capacity:  opts.Blocks * opts.BlockSize,
		kvType:    opts.KVType,
		kvRow:     kvRow,
		poolK:     make([]*hip.Buffer, model.Config.BlockCount),
		poolV:     make([]*hip.Buffer, model.Config.BlockCount),
	}
	if err := m.setup(reader, model); err != nil {
		_ = m.Close()
		return nil, err
	}
	chunk := opts.PrefillChunk
	if chunk <= 0 {
		chunk = defaultPrefillChunk
	}
	if chunk > maxPrefillChunk {
		chunk = maxPrefillChunk
	}
	if chunk > m.capacity {
		chunk = m.capacity
	}
	if err := m.configurePrefill(chunk); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

const (
	defaultPrefillChunk = 512
	maxSpecDraft        = 16
)

func (m *Model) setup(reader *gguf.Reader, model *qwen35.Model) error {
	for _, info := range reader.Tensors() {
		data, err := info.Data()
		if err != nil {
			return err
		}
		buf, err := m.allocDevice(uint64(len(data)))
		if err != nil {
			return fmt.Errorf("allocate %s: %w", info.Name, err)
		}
		if err := m.device.Upload(buf, data); err != nil {
			return fmt.Errorf("upload %s: %w", info.Name, err)
		}
		m.weights[info.Name] = weight{buf: buf, dtype: info.DType}
	}
	if err := m.allocActivations(); err != nil {
		return err
	}
	if err := m.allocPools(); err != nil {
		return err
	}
	spec, err := m.allocDevice(uint64(maxSpecDraft) * uint64(m.config.VocabSize) * 4)
	if err != nil {
		return fmt.Errorf("allocate speculative logits: %w", err)
	}
	m.specLogits = spec
	return m.stream.Synchronize()
}

func (m *Model) allocDevice(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	buf, err := m.device.Alloc(bytes)
	if err != nil {
		return nil, err
	}
	m.all = append(m.all, buf)
	return buf, nil
}

// allocTransient allocates a session-owned buffer that is not tracked in the
// model lifetime list. The session is responsible for closing it.
func (m *Model) allocTransient(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	return m.device.Alloc(bytes)
}

// Allocations returns the number of model-lifetime device allocations. Session
// buffers are not included; this is used by tests to detect growth.
func (m *Model) Allocations() int { return len(m.all) }

func (m *Model) allocActivations() error {
	cfg := m.config
	sizes := []struct {
		dst **hip.Buffer
		n   uint64
	}{
		{&m.act.x, uint64(cfg.HiddenSize)},
		{&m.act.norm, uint64(cfg.HiddenSize)},
		{&m.act.q, uint64(cfg.HeadCount * cfg.HeadDim * 2)},
		{&m.act.k, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&m.act.v, uint64(cfg.KVHeadCount * cfg.HeadDim)},
		{&m.act.attn, uint64(cfg.HeadCount * cfg.HeadDim)},
		{&m.act.tmp, uint64(cfg.HiddenSize)},
		{&m.act.qkv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&m.act.z, uint64(cfg.SSMInnerSize)},
		{&m.act.beta, uint64(cfg.SSMTimeStepRank)},
		{&m.act.alpha, uint64(cfg.SSMTimeStepRank)},
		{&m.act.conv, uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&m.act.gdn, uint64(cfg.SSMInnerSize)},
		{&m.act.gdnNorm, uint64(cfg.SSMInnerSize)},
		{&m.act.ffnGate, uint64(cfg.FFNSize)},
		{&m.act.ffnUp, uint64(cfg.FFNSize)},
		{&m.act.logits, uint64(cfg.VocabSize)},
		{&m.act.scores, uint64(cfg.HeadCount * m.capacity)},
	}
	for _, entry := range sizes {
		buf, err := m.allocDevice(entry.n * 4)
		if err != nil {
			return err
		}
		*entry.dst = buf
	}
	return nil
}

// allocPools allocates one K and V block pool per full-attention layer. The
// pools use allocator block IDs directly.
func (m *Model) allocPools() error {
	cfg := m.config
	bytes := uint64(m.blocks) * uint64(m.blockSize) * m.kvRow
	for i := range cfg.BlockCount {
		if cfg.IsRecurrent(i) {
			continue
		}
		k, err := m.allocDevice(bytes)
		if err != nil {
			return err
		}
		v, err := m.allocDevice(bytes)
		if err != nil {
			return err
		}
		if err := m.device.Memset(m.stream, k, 0, bytes); err != nil {
			return err
		}
		if err := m.device.Memset(m.stream, v, 0, bytes); err != nil {
			return err
		}
		m.poolK[i] = k
		m.poolV[i] = v
	}
	return nil
}

// SetPrefillChunk selects the prompt chunk size. Existing sessions detect the
// generation change and rebuild their cached chunk plans.
func (m *Model) SetPrefillChunk(rows int) error {
	if m == nil || m.closed {
		return fmt.Errorf("gpu: model is closed")
	}
	if rows <= 0 {
		return fmt.Errorf("gpu: prefill chunk %d must be positive", rows)
	}
	if rows > maxPrefillChunk {
		rows = maxPrefillChunk
	}
	if rows > m.capacity {
		rows = m.capacity
	}
	m.stepMu.Lock()
	defer m.stepMu.Unlock()
	return m.configurePrefill(rows)
}

func (m *Model) configurePrefill(rows int) error {
	if rows <= 0 {
		return fmt.Errorf("gpu: prefill chunk %d must be positive", rows)
	}
	if rows == m.prefillRows && m.prefill.x != nil {
		return nil
	}
	for i := len(m.prefillAll) - 1; i >= 0; i-- {
		_ = m.prefillAll[i].Close()
	}
	m.prefillAll = nil
	cfg := m.config
	p := &m.prefill
	maxWidth := cfg.HiddenSize
	for _, width := range []int{cfg.FFNSize, cfg.HeadCount * cfg.HeadDim, cfg.SSMInnerSize} {
		if width > maxWidth {
			maxWidth = width
		}
	}
	rowElems := func(width int) uint64 { return uint64(rows) * uint64(width) * 4 }
	entries := []struct {
		dst   **hip.Buffer
		bytes uint64
	}{
		{&p.x, rowElems(cfg.HiddenSize)},
		{&p.norm, rowElems(cfg.HiddenSize)},
		{&p.q, rowElems(cfg.HeadCount * cfg.HeadDim * 2)},
		{&p.k, rowElems(cfg.KVHeadCount * cfg.HeadDim)},
		{&p.v, rowElems(cfg.KVHeadCount * cfg.HeadDim)},
		{&p.attn, rowElems(cfg.HeadCount * cfg.HeadDim)},
		{&p.tmp, rowElems(cfg.HiddenSize)},
		{&p.qkv, rowElems(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&p.z, rowElems(cfg.SSMInnerSize)},
		{&p.beta, rowElems(cfg.SSMTimeStepRank)},
		{&p.alpha, rowElems(cfg.SSMTimeStepRank)},
		{&p.conv, rowElems(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)},
		{&p.gdnNorm, rowElems(cfg.SSMInnerSize)},
		{&p.ffnGate, rowElems(cfg.FFNSize)},
		{&p.ffnUp, rowElems(cfg.FFNSize)},
		{&p.xF16, uint64(rows) * uint64(maxWidth) * 2},
		{&p.tokens, uint64(rows) * 4},
	}
	for _, entry := range entries {
		buf, err := m.allocTransient(entry.bytes)
		if err != nil {
			return fmt.Errorf("allocate prefill chunk buffer: %w", err)
		}
		*entry.dst = buf
		m.prefillAll = append(m.prefillAll, buf)
	}
	m.prefillRows = rows
	m.prefillGen++
	return nil
}

func (m *Model) weight(name string) (weight, error) {
	w, ok := m.weights[name]
	if !ok {
		return weight{}, fmt.Errorf("missing device weight %q", name)
	}
	return w, nil
}

// CopyBlock copies one complete KV block for every full-attention layer. It is
// synchronous with the model stream and is used for private prefix blocks.
func (m *Model) CopyBlock(dst, src int) error {
	if dst < 0 || dst >= m.blocks || src < 0 || src >= m.blocks {
		return fmt.Errorf("gpu: KV block copy %d <- %d is out of range", dst, src)
	}
	m.stepMu.Lock()
	defer m.stepMu.Unlock()
	rowBytes := uint64(m.blockSize) * m.kvRow
	dstOff := uint64(dst) * rowBytes
	srcOff := uint64(src) * rowBytes
	copies := 0
	for i := range m.poolK {
		if m.poolK[i] == nil {
			continue
		}
		if err := m.device.CopyBuffer(m.stream, m.poolK[i], dstOff, m.poolK[i], srcOff, rowBytes, nil); err != nil {
			return err
		}
		if err := m.device.CopyBuffer(m.stream, m.poolV[i], dstOff, m.poolV[i], srcOff, rowBytes, nil); err != nil {
			return err
		}
		copies++
	}
	if copies == 0 {
		return nil
	}
	return m.stream.Synchronize()
}

// Allocator returns the shared KV block allocator.
func (m *Model) Allocator() *memory.Allocator { return m.alloc }

// BlockSize returns the KV block size.
func (m *Model) BlockSize() int { return m.blockSize }

// VocabSize returns the model vocabulary size.
func (m *Model) VocabSize() int { return m.config.VocabSize }

// KVType returns the full-attention KV pool storage type.
func (m *Model) KVType() tensor.DType { return m.kvType }

// Capacity returns the maximum number of positions in the pool.
func (m *Model) Capacity() int { return m.capacity }

// PrefillRows returns the current prompt chunk size.
func (m *Model) PrefillRows() int { return m.prefillRows }

// Close releases the stream, device, and every allocation. Sessions must be
// closed first.
func (m *Model) Close() error {
	if m == nil || m.closed {
		return nil
	}
	m.closed = true
	for i := len(m.prefillAll) - 1; i >= 0; i-- {
		_ = m.prefillAll[i].Close()
	}
	m.prefillAll = nil
	for i := len(m.all) - 1; i >= 0; i-- {
		_ = m.all[i].Close()
	}
	m.all = nil
	if m.event != nil {
		_ = m.event.Close()
	}
	if m.stream != nil {
		_ = m.stream.Close()
	}
	if m.device != nil {
		_ = m.device.Close()
	}
	return nil
}
