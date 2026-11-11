//go:build hip && linux

package gpu

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/dense"
	"github.com/yuuka-ai/yuuka/tensor"
)

// MoERuntime is a single-request HIP backend for Qwen3 MoE models. It
// implements engine.Runtime.
type MoERuntime struct {
	model    *MoEModel
	table    *memory.Table
	session  *MoESession
	capacity int
	closed   bool
}

// NewMoERuntime uploads a MoE model and builds a single-request session.
func NewMoERuntime(reader *gguf.Reader, model *dense.Model, capacity int, kvType tensor.DType) (*MoERuntime, error) {
	if model == nil || !model.Config.IsMoE() {
		return nil, fmt.Errorf("gpu: a MoE model is required")
	}
	if capacity <= 0 || capacity > model.Config.Context {
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
	m, err := OpenMoE(reader, model, Options{
		Blocks:        blocks,
		BlockSize:     blockSize,
		PrefillChunk:  chunk,
		KVType:        kvType,
		GraphMinSteps: 1,
	})
	if err != nil {
		return nil, err
	}
	r := &MoERuntime{model: m, capacity: capacity}
	if err := r.reset(); err != nil {
		_ = m.Close()
		return nil, err
	}
	return r, nil
}

func (r *MoERuntime) reset() error {
	if r.session != nil {
		_ = r.session.Close()
		r.session = nil
	}
	if r.table != nil {
		r.table.Release()
		r.table = nil
	}
	table := memory.NewTable(r.model.alloc)
	if err := table.Cover(r.capacity); err != nil {
		table.Release()
		return err
	}
	session, err := r.model.NewMoESession(table, 0, r.capacity)
	if err != nil {
		table.Release()
		return err
	}
	r.table = table
	r.session = session
	return nil
}

// Reset releases the request table and starts from position zero.
func (r *MoERuntime) Reset() {
	if r == nil || r.closed {
		return
	}
	_ = r.reset()
}

// Forward processes prompt tokens and returns the last-token logits.
func (r *MoERuntime) Forward(tokens []int32) ([]float32, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.Prefill(tokens, true)
}

// Step processes one token and returns its logits.
func (r *MoERuntime) Step(token int32) ([]float32, error) {
	if r == nil || r.closed {
		return nil, fmt.Errorf("runtime is closed")
	}
	return r.session.Decode(token)
}

// Capacity returns the reserved positions.
func (r *MoERuntime) Capacity() int { return r.capacity }

// VocabSize returns the model vocabulary size.
func (r *MoERuntime) VocabSize() int { return r.model.VocabSize() }

// SnapshotState returns the empty MoE state and current position.
func (r *MoERuntime) SnapshotState() ([]byte, int, error) {
	if r == nil || r.closed {
		return nil, 0, fmt.Errorf("runtime is closed")
	}
	return nil, r.session.Position(), nil
}

// RestoreState rewinds to position. MoE has no recurrent state to restore.
func (r *MoERuntime) RestoreState(_ []byte, position int) error {
	if r == nil || r.closed {
		return fmt.Errorf("runtime is closed")
	}
	return r.session.SetPosition(position)
}

// ForwardTrace returns logits and per-layer hidden state for the last token.
func (r *MoERuntime) ForwardTrace(tokens []int32) ([]float32, [][]float32, error) {
	if r == nil || r.closed {
		return nil, nil, fmt.Errorf("runtime is closed")
	}
	return r.session.ForwardTrace(tokens)
}

// ProfileStep is unavailable for MoE plans; it returns an explicit error.
func (r *MoERuntime) ProfileStep(token int32) ([]hip.OpTime, error) {
	return nil, fmt.Errorf("gpu: MoE profiling is not implemented")
}

// Close releases the session, block table, and model allocations.
func (r *MoERuntime) Close() error {
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
