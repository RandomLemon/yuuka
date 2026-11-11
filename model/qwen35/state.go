package qwen35

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/memory"
)

type recurrentLayer struct {
	conv []float32
	s    []float32
}

type scratch struct {
	x       []float32
	norm    []float32
	q       []float32
	k       []float32
	v       []float32
	attn    []float32
	tmp     []float32
	qkv     []float32
	z       []float32
	beta    []float32
	alpha   []float32
	conv    []float32
	gdn     []float32
	ffnGate []float32
	ffnUp   []float32
	scores  []float32
	sk      []float32
	d       []float32
	logits  []float32
}

// State holds the per-request block table, recurrent convolution/state tensors,
// and reusable scratch buffers. Full-attention KV lives in a shared KVPool
// indexed by block ID. It is not safe for concurrent use.
type State struct {
	Pos      int
	Capacity int
	table    *memory.Table
	pool     *KVPool
	rec      []*recurrentLayer
	scratch  scratch
	owned    *memory.Allocator
}

// NewState allocates a private KV pool and block table for capacity tokens.
// capacity must not exceed the model context length.
func NewState(cfg *Config, capacity int) (*State, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("context capacity must be positive")
	}
	if capacity > cfg.ContextLength {
		return nil, fmt.Errorf("context capacity %d exceeds model limit %d",
			capacity, cfg.ContextLength)
	}
	blockSize := 16
	if capacity < blockSize {
		blockSize = capacity
	}
	blocks := (capacity + blockSize - 1) / blockSize
	pool, err := NewKVPool(cfg, blocks, blockSize)
	if err != nil {
		return nil, err
	}
	alloc, err := memory.NewAllocator(blocks, blockSize)
	if err != nil {
		return nil, err
	}
	table := memory.NewTable(alloc)
	if err := table.Cover(capacity); err != nil {
		return nil, err
	}
	state, err := NewPagedState(cfg, pool, table, capacity)
	if err != nil {
		table.Release()
		return nil, err
	}
	state.owned = alloc
	return state, nil
}

// NewPagedState binds one request to a shared KV pool and block table. The
// table must already map at least capacity positions and the caller retains
// ownership of every block reference in it. Close releases the references.
func NewPagedState(cfg *Config, pool *KVPool, table *memory.Table, capacity int) (*State, error) {
	if cfg == nil || pool == nil || table == nil {
		return nil, fmt.Errorf("qwen35: config, KV pool, and block table are required")
	}
	if capacity <= 0 || capacity > cfg.ContextLength {
		return nil, fmt.Errorf("context capacity %d is invalid", capacity)
	}
	if table.Bytes() < capacity {
		return nil, fmt.Errorf("block table maps %d positions, need %d", table.Bytes(), capacity)
	}
	if table.Allocator().BlockSize() != pool.BlockSize() {
		return nil, fmt.Errorf("block table and KV pool block sizes differ")
	}
	st := &State{
		Capacity: capacity,
		table:    table,
		pool:     pool,
		rec:      make([]*recurrentLayer, cfg.BlockCount),
	}
	for i := 0; i < cfg.BlockCount; i++ {
		if !cfg.IsRecurrent(i) {
			continue
		}
		convDim := 2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize
		st.rec[i] = &recurrentLayer{
			conv: make([]float32, (cfg.SSMConvKernel-1)*convDim),
			s:    make([]float32, cfg.SSMTimeStepRank*cfg.SSMStateSize*cfg.SSMStateSize),
		}
	}
	qDim := cfg.HeadCount * cfg.HeadDim * 2
	oDim := cfg.HeadCount * cfg.HeadDim
	inner := cfg.SSMInnerSize
	convDim := 2*cfg.SSMGroupCount*cfg.SSMStateSize + inner
	st.scratch = scratch{
		x:       make([]float32, cfg.HiddenSize),
		norm:    make([]float32, cfg.HiddenSize),
		q:       make([]float32, qDim),
		k:       make([]float32, cfg.KVHeadCount*cfg.HeadDim),
		v:       make([]float32, cfg.KVHeadCount*cfg.HeadDim),
		attn:    make([]float32, oDim),
		tmp:     make([]float32, cfg.HiddenSize),
		qkv:     make([]float32, convDim),
		z:       make([]float32, inner),
		beta:    make([]float32, cfg.SSMTimeStepRank),
		alpha:   make([]float32, cfg.SSMTimeStepRank),
		conv:    make([]float32, convDim),
		gdn:     make([]float32, inner),
		ffnGate: make([]float32, cfg.FFNSize),
		ffnUp:   make([]float32, cfg.FFNSize),
		scores:  make([]float32, capacity),
		sk:      make([]float32, cfg.SSMStateSize),
		d:       make([]float32, cfg.SSMStateSize),
		logits:  make([]float32, cfg.VocabSize),
	}
	return st, nil
}

// Reset clears recurrent state, returns the position to zero, and zeroes KV
// blocks owned privately by NewState. It must not be called on a state sharing
// its pool with other requests.
func (s *State) Reset() {
	s.Pos = 0
	if s.owned != nil {
		for _, id := range s.table.Blocks() {
			if s.owned.Refs(id) == 1 {
				s.pool.ClearBlock(id)
			}
		}
	}
	for _, rec := range s.rec {
		if rec != nil {
			clear(rec.conv)
			clear(rec.s)
		}
	}
}

// Remaining returns the number of token positions left in the request.
func (s *State) Remaining() int { return s.Capacity - s.Pos }

// Table returns the request block table.
func (s *State) Table() *memory.Table { return s.table }

// Pool returns the shared KV pool.
func (s *State) Pool() *KVPool { return s.pool }

// SetPos restores the absolute position after a prefix-cache hit.
func (s *State) SetPos(pos int) error {
	if pos < 0 || pos > s.Capacity {
		return fmt.Errorf("qwen35: position %d exceeds capacity %d", pos, s.Capacity)
	}
	s.Pos = pos
	return nil
}

// Snapshot serializes recurrent convolution and matrix state. Full-attention
// KV is described by the block table and is not copied.
func (s *State) Snapshot() []byte {
	elements := 0
	for _, rec := range s.rec {
		if rec != nil {
			elements += len(rec.conv) + len(rec.s)
		}
	}
	data := make([]byte, elements*4)
	offset := 0
	put := func(values []float32) {
		for _, value := range values {
			binary.LittleEndian.PutUint32(data[offset:], math.Float32bits(value))
			offset += 4
		}
	}
	for _, rec := range s.rec {
		if rec == nil {
			continue
		}
		put(rec.conv)
		put(rec.s)
	}
	return data
}

// Restore applies a snapshot produced by Snapshot at the same absolute
// position. It validates the byte length before mutating state.
func (s *State) Restore(data []byte) error {
	elements := 0
	for _, rec := range s.rec {
		if rec != nil {
			elements += len(rec.conv) + len(rec.s)
		}
	}
	if len(data) != elements*4 {
		return fmt.Errorf("qwen35: state snapshot is %d bytes, want %d", len(data), elements*4)
	}
	offset := 0
	get := func(values []float32) {
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4
		}
	}
	for _, rec := range s.rec {
		if rec == nil {
			continue
		}
		get(rec.conv)
		get(rec.s)
	}
	return nil
}
