package serving

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
)

// Qwen35Backend executes Qwen3.5 requests on the CPU reference path over a
// shared host KV pool. The scheduler serializes sequence execution.
type Qwen35Backend struct {
	model *qwen35.Model
	pool  *qwen35.KVPool
	alloc *memory.Allocator
	key   string
}

// NewQwen35Backend creates a host F32 KV pool with fixed-size blocks. key must
// identify the model and tokenizer configuration for prefix caching.
func NewQwen35Backend(model *qwen35.Model, key string, blocks, blockSize int) (*Qwen35Backend, error) {
	return NewQwen35BackendWithFormat(model, key, blocks, blockSize, qwen35.KVF32)
}

// NewQwen35BackendWithFormat creates a host KV pool in format. key must include
// the format for prefix-cache correctness.
func NewQwen35BackendWithFormat(model *qwen35.Model, key string, blocks, blockSize int, format qwen35.KVFormat) (*Qwen35Backend, error) {
	if model == nil {
		return nil, fmt.Errorf("serving: qwen35 model is required")
	}
	alloc, err := memory.NewAllocator(blocks, blockSize)
	if err != nil {
		return nil, err
	}
	pool, err := qwen35.NewKVPoolWithFormat(model.Config, blocks, blockSize, format)
	if err != nil {
		return nil, err
	}
	return &Qwen35Backend{model: model, pool: pool, alloc: alloc, key: key}, nil
}

// KVFormat returns the host pool storage format.
func (b *Qwen35Backend) KVFormat() qwen35.KVFormat { return b.pool.Format() }

// ModelKey returns the prefix-cache identity.
func (b *Qwen35Backend) ModelKey() string { return b.key }

// VocabSize returns the model vocabulary size.
func (b *Qwen35Backend) VocabSize() int { return b.model.Config.VocabSize }

// BlockSize returns the KV block size.
func (b *Qwen35Backend) BlockSize() int { return b.alloc.BlockSize() }

// Allocator returns the shared KV block allocator.
func (b *Qwen35Backend) Allocator() *memory.Allocator { return b.alloc }

// NewSession binds a block table at position and restores state.
func (b *Qwen35Backend) NewSession(table *memory.Table, capacity, position int, state []byte) (Sequence, error) {
	st, err := qwen35.NewPagedState(b.model.Config, b.pool, table, capacity)
	if err != nil {
		return nil, err
	}
	if err := st.SetPos(position); err != nil {
		return nil, err
	}
	if len(state) > 0 {
		if err := st.Restore(state); err != nil {
			return nil, err
		}
	}
	return &qwen35Sequence{model: b.model, state: st}, nil
}

// CopyBlock copies one complete host KV block.
func (b *Qwen35Backend) CopyBlock(dst, src int) error { return b.pool.CopyBlock(dst, src) }

// Close releases backend-owned resources. The shared model stays open.
func (b *Qwen35Backend) Close() error { return nil }

type qwen35Sequence struct {
	model *qwen35.Model
	state *qwen35.State
}

func (s *qwen35Sequence) Prefill(tokens []int32, final bool) ([]float32, error) {
	return s.model.Prefill(tokens, s.state, final)
}

func (s *qwen35Sequence) Decode(token int32) ([]float32, error) {
	return s.model.Step(token, s.state)
}

func (s *qwen35Sequence) Snapshot() ([]byte, error) {
	return s.state.Snapshot(), nil
}

func (s *qwen35Sequence) Close() error { return nil }

var (
	_ Backend  = (*Qwen35Backend)(nil)
	_ Sequence = (*qwen35Sequence)(nil)
)
