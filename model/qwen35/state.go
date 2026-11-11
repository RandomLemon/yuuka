package qwen35

import "fmt"

type kvLayer struct {
	k []float32
	v []float32
}

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

// State holds the per-request KV cache, recurrent convolution/state tensors,
// and reusable scratch buffers. It is not safe for concurrent use.
type State struct {
	Pos      int
	Capacity int
	kv       []*kvLayer
	rec      []*recurrentLayer
	scratch  scratch
}

// NewState allocates caches for capacity tokens. capacity must not exceed the
// model context length.
func NewState(cfg *Config, capacity int) (*State, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("context capacity must be positive")
	}
	if capacity > cfg.ContextLength {
		return nil, fmt.Errorf("context capacity %d exceeds model limit %d",
			capacity, cfg.ContextLength)
	}
	st := &State{
		Capacity: capacity,
		kv:       make([]*kvLayer, cfg.BlockCount),
		rec:      make([]*recurrentLayer, cfg.BlockCount),
	}
	kvDim := cfg.KVHeadCount * cfg.HeadDim
	for i := 0; i < cfg.BlockCount; i++ {
		if !cfg.IsRecurrent(i) {
			st.kv[i] = &kvLayer{
				k: make([]float32, capacity*kvDim),
				v: make([]float32, capacity*kvDim),
			}
		} else {
			convDim := 2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize
			st.rec[i] = &recurrentLayer{
				conv: make([]float32, (cfg.SSMConvKernel-1)*convDim),
				s:    make([]float32, cfg.SSMTimeStepRank*cfg.SSMStateSize*cfg.SSMStateSize),
			}
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

// Reset clears all cached state and returns the position to zero.
func (s *State) Reset() {
	s.Pos = 0
	for _, kv := range s.kv {
		if kv != nil {
			clear(kv.k)
			clear(kv.v)
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
