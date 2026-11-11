package dense

import "fmt"

// Runtime adapts a dense CPU model to the engine.Runtime interface.
type Runtime struct {
	model  *Model
	state  *State
	logits []float32
}

// NewRuntime allocates CPU state for capacity tokens.
func NewRuntime(model *Model, capacity int) (*Runtime, error) {
	state, err := NewState(model.Config, capacity)
	if err != nil {
		return nil, err
	}
	return &Runtime{
		model:  model,
		state:  state,
		logits: make([]float32, model.Config.VocabSize),
	}, nil
}

// Reset clears the KV cache and recurrent state.
func (r *Runtime) Reset() { r.state.Reset() }

// Forward processes tokens and returns the last-token logits.
func (r *Runtime) Forward(tokens []int32) ([]float32, error) {
	logits, err := r.model.Forward(tokens, r.state)
	if err != nil {
		return nil, err
	}
	copy(r.logits, logits)
	return r.logits, nil
}

// Step processes one token.
func (r *Runtime) Step(token int32) ([]float32, error) { return r.Forward([]int32{token}) }

// Capacity returns the token capacity.
func (r *Runtime) Capacity() int { return r.state.Capacity }

// VocabSize returns the vocabulary size.
func (r *Runtime) VocabSize() int { return r.model.Config.VocabSize }

// Close releases owned references. The model still borrows the GGUF mapping.
func (r *Runtime) Close() error {
	if r == nil {
		return nil
	}
	r.model = nil
	r.state = nil
	r.logits = nil
	return nil
}

// Check that the adapter has the expected method shape without importing the
// engine package and creating a cycle.
var _ interface {
	Reset()
	Forward([]int32) ([]float32, error)
	Step(int32) ([]float32, error)
	Capacity() int
	VocabSize() int
	Close() error
} = (*Runtime)(nil)

// ValidateCapacity is a helper for callers that only have a config.
func ValidateCapacity(cfg *Config, capacity int) error {
	if capacity <= 0 {
		return fmt.Errorf("context capacity must be positive")
	}
	if capacity > cfg.Context {
		return fmt.Errorf("context capacity %d exceeds model limit %d", capacity, cfg.Context)
	}
	return nil
}
