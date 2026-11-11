package dense

import "fmt"

type kvLayer struct {
	k []float32
	v []float32
}

type scratch struct {
	x      []float32
	norm   []float32
	q      []float32
	k      []float32
	v      []float32
	attn   []float32
	tmp    []float32
	gate   []float32
	up     []float32
	scores []float32
	logits []float32

	router        []float32
	expertGate    []float32
	expertUp      []float32
	expertOut     []float32
	expertIDs     []int32
	expertWeights []float32
}

// State holds the per-request KV cache and reusable scratch buffers. It is not
// safe for concurrent use.
type State struct {
	Pos      int
	Capacity int
	kv       []*kvLayer
	scratch  scratch
}

// NewState allocates caches for capacity tokens.
func NewState(cfg *Config, capacity int) (*State, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("context capacity must be positive")
	}
	if capacity > cfg.Context {
		return nil, fmt.Errorf("context capacity %d exceeds model limit %d", capacity, cfg.Context)
	}
	st := &State{
		Capacity: capacity,
		kv:       make([]*kvLayer, cfg.BlockCount),
	}
	kvDim := cfg.KVHeadCount * cfg.HeadDim
	for i := range st.kv {
		st.kv[i] = &kvLayer{
			k: make([]float32, capacity*kvDim),
			v: make([]float32, capacity*kvDim),
		}
	}
	qDim := cfg.HeadCount * cfg.HeadDim
	st.scratch = scratch{
		x:      make([]float32, cfg.HiddenSize),
		norm:   make([]float32, cfg.HiddenSize),
		q:      make([]float32, qDim),
		k:      make([]float32, kvDim),
		v:      make([]float32, kvDim),
		attn:   make([]float32, qDim),
		tmp:    make([]float32, cfg.HiddenSize),
		gate:   make([]float32, cfg.FFNSize),
		up:     make([]float32, cfg.FFNSize),
		scores: make([]float32, capacity),
		logits: make([]float32, cfg.VocabSize),
	}
	if cfg.IsMoE() {
		ffn := cfg.FFNSize
		if cfg.ExpertFFNSize > ffn {
			ffn = cfg.ExpertFFNSize
		}
		st.scratch.router = make([]float32, cfg.ExpertCount)
		st.scratch.expertGate = make([]float32, ffn)
		st.scratch.expertUp = make([]float32, ffn)
		st.scratch.expertOut = make([]float32, cfg.HiddenSize)
		st.scratch.expertIDs = make([]int32, cfg.ExpertUsedCount)
		st.scratch.expertWeights = make([]float32, cfg.ExpertUsedCount)
	}
	return st, nil
}

// Reset clears all cached state and returns the position to zero.
func (s *State) Reset() {
	s.Pos = 0
	for _, kv := range s.kv {
		clear(kv.k)
		clear(kv.v)
	}
}
