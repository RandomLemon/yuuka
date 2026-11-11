package engine

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// SamplerConfig controls token sampling. Temperature <= 0 selects greedy
// decoding. TopK <= 0 disables top-k and TopP >= 1 disables nucleus filtering.
type SamplerConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        int64
}

type candidate struct {
	id    int32
	logit float32
	prob  float64
}

// Sampler is a single-request sampler. It is not safe for concurrent use.
type Sampler struct {
	cfg        SamplerConfig
	rng        *rand.Rand
	candidates []candidate
}

// NewSampler validates the configuration and creates a deterministic sampler.
func NewSampler(cfg SamplerConfig, vocabSize int) (*Sampler, error) {
	if vocabSize <= 0 {
		return nil, fmt.Errorf("invalid vocabulary size %d", vocabSize)
	}
	if cfg.Temperature < 0 {
		return nil, fmt.Errorf("temperature must not be negative")
	}
	if cfg.TopK < 0 {
		return nil, fmt.Errorf("top-k must not be negative")
	}
	if cfg.TopP < 0 || cfg.TopP > 1 {
		return nil, fmt.Errorf("top-p must be in (0,1]")
	}
	if cfg.TopP == 0 {
		cfg.TopP = 1
	}
	return &Sampler{
		cfg:        cfg,
		rng:        rand.New(rand.NewPCG(uint64(cfg.Seed), uint64(cfg.Seed)^0x9e3779b97f4a7c15)),
		candidates: make([]candidate, 0, vocabSize),
	}, nil
}

// Sample selects one token from logits. The input slice is not modified.
func (s *Sampler) Sample(logits []float32) (int32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	if s.cfg.Temperature <= 0 || s.cfg.TopK == 1 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return int32(best), nil
	}

	s.candidates = s.candidates[:0]
	for i, logit := range logits {
		if logit == float32(math.Inf(-1)) || math.IsNaN(float64(logit)) {
			continue
		}
		s.candidates = append(s.candidates, candidate{id: int32(i), logit: logit})
	}
	if len(s.candidates) == 0 {
		return 0, fmt.Errorf("all logits are invalid")
	}
	sort.Slice(s.candidates, func(i, j int) bool {
		if s.candidates[i].logit != s.candidates[j].logit {
			return s.candidates[i].logit > s.candidates[j].logit
		}
		return s.candidates[i].id < s.candidates[j].id
	})
	if s.cfg.TopK > 0 && s.cfg.TopK < len(s.candidates) {
		s.candidates = s.candidates[:s.cfg.TopK]
	}

	maxLogit := s.candidates[0].logit
	var total float64
	for i := range s.candidates {
		prob := math.Exp(float64(s.candidates[i].logit-maxLogit) / float64(s.cfg.Temperature))
		s.candidates[i].prob = prob
		total += prob
	}
	if total == 0 || math.IsInf(total, 0) {
		return s.candidates[0].id, nil
	}
	if s.cfg.TopP < 1 {
		var cumulative float64
		keep := 0
		for i := range s.candidates {
			cumulative += s.candidates[i].prob / total
			keep = i + 1
			if cumulative >= float64(s.cfg.TopP) {
				break
			}
		}
		s.candidates = s.candidates[:keep]
		total = 0
		for i := range s.candidates {
			total += s.candidates[i].prob
		}
	}
	draw := s.rng.Float64() * total
	var cumulative float64
	for i := range s.candidates {
		cumulative += s.candidates[i].prob
		if draw < cumulative {
			return s.candidates[i].id, nil
		}
	}
	return s.candidates[len(s.candidates)-1].id, nil
}
