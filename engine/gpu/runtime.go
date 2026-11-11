//go:build hip && linux

// Package gpu executes the Qwen3.5 hybrid decoder and Qwen3 MoE on the HIP
// backend. A shared model owns weights and KV pools; each request owns a block
// table and optional recurrent state. Runtime and MoERuntime are the
// single-request compatibility wrappers used by the CLI.
package gpu

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

// Runtime is a single-request HIP execution backend implementing
// engine.Runtime. It owns one block table and one session over a shared Model.
type Runtime struct {
	model       *Model
	session     *Session
	table       *memory.Table
	capacity    int
	prefillRows int
	closed      bool
}

// New uploads model weights and builds a single-request F32-KV session with a
// contiguous block table covering capacity positions.
func New(reader *gguf.Reader, model *qwen35.Model, capacity int) (*Runtime, error) {
	return NewWithKVType(reader, model, capacity, tensor.F32)
}

// NewWithKVType is New with an explicit KV pool storage type.
func NewWithKVType(reader *gguf.Reader, model *qwen35.Model, capacity int, kvType tensor.DType) (*Runtime, error) {
	if model == nil {
		return nil, fmt.Errorf("gpu: model is required")
	}
	if capacity <= 0 || capacity > model.Config.ContextLength {
		return nil, fmt.Errorf("gpu: invalid context capacity %d", capacity)
	}
	blockSize := 16
	if capacity < blockSize {
		blockSize = capacity
	}
	blocks := (capacity + blockSize - 1) / blockSize
	chunk := defaultPrefillChunk
	if chunk > capacity {
		chunk = capacity
	}
	m, err := Open(reader, model, Options{
		Blocks:        blocks,
		BlockSize:     blockSize,
		PrefillChunk:  chunk,
		EnableGraph:   false,
		GraphMinSteps: 1,
		KVType:        kvType,
	})
	if err != nil {
		return nil, err
	}
	table := memory.NewTable(m.alloc)
	if err := table.Cover(capacity); err != nil {
		table.Release()
		_ = m.Close()
		return nil, err
	}
	session, err := m.NewSession(table, 0, nil, SessionOptions{
		Capacity:      capacity,
		EnableGraph:   false,
		GraphMinSteps: 1,
		Trace:         true,
	})
	if err != nil {
		table.Release()
		_ = m.Close()
		return nil, err
	}
	return &Runtime{
		model:       m,
		session:     session,
		table:       table,
		capacity:    capacity,
		prefillRows: m.prefillRows,
	}, nil
}

// SetPrefillChunk selects the prompt chunk size. It clamps the request to the
// runtime capacity and maxPrefillChunk, reallocates chunk buffers, and rebuilds
// cached plans.
func (r *Runtime) SetPrefillChunk(rows int) error {
	if r == nil || r.closed {
		return fmt.Errorf("runtime is closed")
	}
	if rows <= 0 {
		return fmt.Errorf("prefill chunk %d must be positive", rows)
	}
	if rows > maxPrefillChunk {
		rows = maxPrefillChunk
	}
	if rows > r.capacity {
		rows = r.capacity
	}
	if err := r.model.SetPrefillChunk(rows); err != nil {
		return err
	}
	r.prefillRows = r.model.prefillRows
	return nil
}

// Reset clears recurrent state and returns the position to zero.
func (r *Runtime) Reset() {
	if r == nil || r.closed {
		return
	}
	if err := r.session.Reset(); err != nil {
		return
	}
}

// Forward processes a prompt and returns the logits for its last token.
func (r *Runtime) Forward(tokens []int32) ([]float32, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.Prefill(tokens, true)
}

// Step processes one token and returns its logits.
func (r *Runtime) Step(token int32) ([]float32, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.Decode(token)
}

// PrefillLogits returns logits for every position in tokens.
func (r *Runtime) PrefillLogits(tokens []int32) ([][]float32, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.PrefillLogits(tokens)
}

// SnapshotState captures recurrent state and the current position.
func (r *Runtime) SnapshotState() ([]byte, int, error) {
	if r == nil || r.closed {
		return nil, 0, fmt.Errorf("runtime is closed")
	}
	data, err := r.session.Snapshot()
	if err != nil {
		return nil, 0, err
	}
	return data, r.session.Position(), nil
}

// RestoreState restores recurrent state and rewinds to position.
func (r *Runtime) RestoreState(data []byte, position int) error {
	if r == nil || r.closed {
		return fmt.Errorf("runtime is closed")
	}
	if err := r.session.Restore(data); err != nil {
		return err
	}
	return r.session.SetPosition(position)
}

// ForwardTrace returns logits and per-layer hidden state for the last token.
func (r *Runtime) ForwardTrace(tokens []int32) ([]float32, [][]float32, error) {
	if r == nil || r.closed {
		return nil, nil, fmt.Errorf("runtime is closed")
	}
	return r.session.ForwardTrace(tokens)
}

// ProfileStep executes one token with per-operation HIP events.
func (r *Runtime) ProfileStep(token int32) ([]hip.OpTime, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.ProfileStep(token)
}

// Capacity returns the reserved token positions.
func (r *Runtime) Capacity() int { return r.capacity }

// VocabSize returns the model vocabulary size.
func (r *Runtime) VocabSize() int { return r.model.config.VocabSize }

// Close releases the session, block table, and model allocations. It is
// idempotent.
func (r *Runtime) Close() error {
	if r == nil || r.closed {
		return nil
	}
	r.closed = true
	if r.session != nil {
		_ = r.session.Close()
	}
	if r.table != nil {
		r.table.Release()
	}
	if r.model != nil {
		_ = r.model.Close()
	}
	return nil
}

var _ engine.Runtime = (*Runtime)(nil)
