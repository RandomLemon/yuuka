//go:build hip && linux

package gpu

import (
	"fmt"
	"sync"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/dense"
	"github.com/yuuka-ai/yuuka/tensor"
)

type moeActivations struct {
	maxRows int

	x       *hip.Buffer
	norm    *hip.Buffer
	q       *hip.Buffer
	k       *hip.Buffer
	v       *hip.Buffer
	attn    *hip.Buffer
	tmp     *hip.Buffer
	logits  *hip.Buffer
	scores  *hip.Buffer
	router  *hip.Buffer
	ids     *hip.Buffer
	weights *hip.Buffer
	counts  *hip.Buffer

	assignRows    *hip.Buffer
	assignSlots   *hip.Buffer
	assignWeights *hip.Buffer

	gateOut   *hip.Buffer
	upOut     *hip.Buffer
	actOut    *hip.Buffer
	expertOut *hip.Buffer

	xF16   *hip.Buffer
	tokens *hip.Buffer
}

// MoEModel owns device weights and grouped-expert scratch for a Qwen3 MoE
// model. It is separate from Model so the qwen35 hybrid path is unchanged.
type MoEModel struct {
	device *hip.Device
	stream *hip.Stream
	event  *hip.Event
	config *dense.Config
	alloc  *memory.Allocator

	weights map[string]weight
	act     moeActivations

	poolK     []*hip.Buffer
	poolV     []*hip.Buffer
	blockSize int
	blocks    int
	capacity  int
	kvType    tensor.DType
	kvRow     uint64

	stepMu sync.Mutex
	all    []*hip.Buffer
	closed bool
}

// OpenMoE uploads a qwen3moe model and allocates the shared KV pool and MoE
// scratch for prompts up to Options.PrefillChunk.
func OpenMoE(reader *gguf.Reader, model *dense.Model, opts Options) (*MoEModel, error) {
	if model == nil || reader == nil {
		return nil, fmt.Errorf("gpu: MoE model and reader are required")
	}
	if !model.Config.IsMoE() {
		return nil, fmt.Errorf("gpu: model architecture %q is not a MoE architecture", model.Config.Architecture)
	}
	if opts.Blocks <= 0 || opts.BlockSize <= 0 {
		return nil, fmt.Errorf("gpu: KV pool needs positive blocks and block size")
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
	m := &MoEModel{
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
	rows := opts.PrefillChunk
	if rows <= 0 {
		rows = defaultPrefillChunk
	}
	if rows > maxPrefillChunk {
		rows = maxPrefillChunk
	}
	if rows > m.capacity {
		rows = m.capacity
	}
	if err := m.setup(reader, rows); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

func (m *MoEModel) setup(reader *gguf.Reader, rows int) error {
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
	if err := m.allocActivations(rows); err != nil {
		return err
	}
	if err := m.allocPools(); err != nil {
		return err
	}
	return m.stream.Synchronize()
}

func (m *MoEModel) allocDevice(bytes uint64) (*hip.Buffer, error) {
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

func (m *MoEModel) allocTransient(bytes uint64) (*hip.Buffer, error) {
	if bytes == 0 {
		return nil, fmt.Errorf("zero-byte allocation")
	}
	return m.device.Alloc(bytes)
}

func (m *MoEModel) allocActivations(rows int) error {
	cfg := m.config
	a := &m.act
	a.maxRows = rows
	maxWidth := cfg.HiddenSize
	for _, width := range []int{cfg.ExpertFFNSize, cfg.ExpertCount} {
		if width > maxWidth {
			maxWidth = width
		}
	}
	rowBytes := func(width int) uint64 { return uint64(rows) * uint64(width) * 4 }
	assignRows := uint64(cfg.ExpertCount) * uint64(rows) * uint64(cfg.ExpertUsedCount) * 4
	entries := []struct {
		dst   **hip.Buffer
		bytes uint64
	}{
		{&a.x, rowBytes(cfg.HiddenSize)},
		{&a.norm, rowBytes(cfg.HiddenSize)},
		{&a.q, rowBytes(cfg.HeadCount * cfg.HeadDim)},
		{&a.k, rowBytes(cfg.KVHeadCount * cfg.HeadDim)},
		{&a.v, rowBytes(cfg.KVHeadCount * cfg.HeadDim)},
		{&a.attn, rowBytes(cfg.HeadCount * cfg.HeadDim)},
		{&a.tmp, rowBytes(cfg.HiddenSize)},
		{&a.logits, uint64(cfg.VocabSize) * 4},
		{&a.scores, uint64(cfg.HeadCount) * uint64(m.capacity) * 4},
		{&a.router, rowBytes(cfg.ExpertCount)},
		{&a.ids, uint64(rows) * uint64(cfg.ExpertUsedCount) * 4},
		{&a.weights, uint64(rows) * uint64(cfg.ExpertUsedCount) * 4},
		{&a.counts, uint64(cfg.ExpertCount) * 4},
		{&a.assignRows, assignRows},
		{&a.assignSlots, assignRows},
		{&a.assignWeights, assignRows},
		{&a.gateOut, uint64(rows) * uint64(cfg.ExpertUsedCount) * uint64(cfg.ExpertFFNSize) * 4},
		{&a.upOut, uint64(rows) * uint64(cfg.ExpertUsedCount) * uint64(cfg.ExpertFFNSize) * 4},
		{&a.actOut, uint64(rows) * uint64(cfg.ExpertUsedCount) * uint64(cfg.ExpertFFNSize) * 4},
		{&a.expertOut, uint64(rows) * uint64(cfg.ExpertUsedCount) * uint64(cfg.HiddenSize) * 4},
		{&a.xF16, uint64(rows) * uint64(maxWidth) * 2},
		{&a.tokens, uint64(rows) * 4},
	}
	for _, entry := range entries {
		buf, err := m.allocDevice(entry.bytes)
		if err != nil {
			return fmt.Errorf("allocate MoE activation: %w", err)
		}
		*entry.dst = buf
	}
	return nil
}

func (m *MoEModel) allocPools() error {
	cfg := m.config
	bytes := uint64(m.blocks) * uint64(m.blockSize) * m.kvRow
	for i := range cfg.BlockCount {
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

func (m *MoEModel) weight(name string) (weight, error) {
	w, ok := m.weights[name]
	if !ok {
		return weight{}, fmt.Errorf("missing device weight %q", name)
	}
	return w, nil
}

// CopyBlock copies a complete formatted KV block for every layer.
func (m *MoEModel) CopyBlock(dst, src int) error {
	if dst < 0 || dst >= m.blocks || src < 0 || src >= m.blocks {
		return fmt.Errorf("gpu: KV block copy %d <- %d is out of range", dst, src)
	}
	m.stepMu.Lock()
	defer m.stepMu.Unlock()
	rowBytes := uint64(m.blockSize) * m.kvRow
	dstOff := uint64(dst) * rowBytes
	srcOff := uint64(src) * rowBytes
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
	}
	return m.stream.Synchronize()
}

// Allocator returns the shared block allocator.
func (m *MoEModel) Allocator() *memory.Allocator { return m.alloc }

// BlockSize returns the KV block size.
func (m *MoEModel) BlockSize() int { return m.blockSize }

// VocabSize returns the vocabulary size.
func (m *MoEModel) VocabSize() int { return m.config.VocabSize }

// Capacity returns the maximum pool positions.
func (m *MoEModel) Capacity() int { return m.capacity }

// KVType returns the KV pool storage type.
func (m *MoEModel) KVType() tensor.DType { return m.kvType }

// Close releases every model-lifetime allocation. Sessions must close first.
func (m *MoEModel) Close() error {
	if m == nil || m.closed {
		return nil
	}
	m.closed = true
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
