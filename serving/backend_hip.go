//go:build hip && linux

package serving

import (
	"github.com/yuuka-ai/yuuka/engine/gpu"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

// Qwen35GPUBackend executes requests on the HIP backend over a shared paged KV
// pool. Sessions are stepped serially by the scheduler.
type Qwen35GPUBackend struct {
	model *gpu.Model
	key   string
	graph bool
}

// NewQwen35GPUBackend opens the shared device model and KV pools.
func NewQwen35GPUBackend(reader *gguf.Reader, model *qwen35.Model, key string,
	blocks, blockSize, prefillChunk int, graph bool) (*Qwen35GPUBackend, error) {
	return NewQwen35GPUBackendWithKV(reader, model, key, blocks, blockSize,
		prefillChunk, graph, tensor.F32)
}

// NewQwen35GPUBackendWithKV is NewQwen35GPUBackend with an explicit KV pool
// storage type. key must include the format for prefix-cache correctness.
func NewQwen35GPUBackendWithKV(reader *gguf.Reader, model *qwen35.Model, key string,
	blocks, blockSize, prefillChunk int, graph bool, kvType tensor.DType) (*Qwen35GPUBackend, error) {
	m, err := gpu.Open(reader, model, gpu.Options{
		Blocks:        blocks,
		BlockSize:     blockSize,
		PrefillChunk:  prefillChunk,
		EnableGraph:   graph,
		GraphMinSteps: 1,
		KVType:        kvType,
	})
	if err != nil {
		return nil, err
	}
	return &Qwen35GPUBackend{model: m, key: key, graph: graph}, nil
}

// ModelKey returns the prefix-cache identity.
func (b *Qwen35GPUBackend) ModelKey() string { return b.key }

// VocabSize returns the model vocabulary size.
func (b *Qwen35GPUBackend) VocabSize() int { return b.model.VocabSize() }

// BlockSize returns the KV block size.
func (b *Qwen35GPUBackend) BlockSize() int { return b.model.BlockSize() }

// Allocator returns the shared block allocator.
func (b *Qwen35GPUBackend) Allocator() *memory.Allocator { return b.model.Allocator() }

// NewSession binds a block table and restores recurrent state.
func (b *Qwen35GPUBackend) NewSession(table *memory.Table, capacity, position int, state []byte) (Sequence, error) {
	s, err := b.model.NewSession(table, position, state, gpu.SessionOptions{
		Capacity:      capacity,
		EnableGraph:   b.graph,
		GraphMinSteps: 1,
	})
	if err != nil {
		return nil, err
	}
	return &gpuSequence{session: s}, nil
}

// CopyBlock copies a complete KV block across all full-attention layers.
func (b *Qwen35GPUBackend) CopyBlock(dst, src int) error { return b.model.CopyBlock(dst, src) }

// Close releases the device model and KV pools.
func (b *Qwen35GPUBackend) Close() error { return b.model.Close() }

// GPUModel exposes the shared model for diagnostics and tests.
func (b *Qwen35GPUBackend) GPUModel() *gpu.Model { return b.model }

type gpuSequence struct {
	session *gpu.Session
}

func (s *gpuSequence) Prefill(tokens []int32, final bool) ([]float32, error) {
	return s.session.Prefill(tokens, final)
}

func (s *gpuSequence) Decode(token int32) ([]float32, error) {
	return s.session.Decode(token)
}

func (s *gpuSequence) Snapshot() ([]byte, error) { return s.session.Snapshot() }

func (s *gpuSequence) Close() error { return s.session.Close() }

var (
	_ Backend  = (*Qwen35GPUBackend)(nil)
	_ Sequence = (*gpuSequence)(nil)
)
