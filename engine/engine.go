// Package engine drives single-request prefill, decode, sampling, and stop
// handling for a loaded model.
package engine

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// ErrStop is returned by an OnToken callback to terminate generation cleanly.
var ErrStop = errors.New("generation stopped by callback")

// GenerateOptions controls one generation request.
type GenerateOptions struct {
	MaxTokens int
	// StopTokens are checked before a token is emitted. EOS is always a stop
	// token when the tokenizer defines one.
	StopTokens []int32
	// IgnoreEOS continues generation past the tokenizer EOS token. It is used
	// by benchmarks that need a fixed decode count.
	IgnoreEOS bool
	// OnToken receives each decoded text delta. It must not retain the slice
	// beyond the call.
	OnToken func(token int32, delta string) error
	// Speculative enables exact greedy speculative decoding. It requires a
	// runtime implementing SpeculativeRuntime and a greedy sampler.
	Speculative *SpeculativeOptions
}

// GenerateResult reports emitted tokens, text, timings, and the stop reason.
type GenerateResult struct {
	Tokens         []int32
	Text           string
	PromptTokens   int
	OutputTokens   int
	FinishReason   string
	PrefillTime    time.Duration
	DecodeTime     time.Duration
	TimeToFirst    time.Duration
	PromptTokensPS float64
	DecodeTokensPS float64
	// DraftTokens and AcceptedDraft report speculative verification results.
	DraftTokens   int
	AcceptedDraft int
}

// SpeculativeRuntime is the additional execution surface required for exact
// greedy speculative decoding. It exposes per-position logits and a rollback
// point for partially accepted draft blocks.
type SpeculativeRuntime interface {
	Runtime
	// PrefillLogits processes tokens and returns logits for every position.
	PrefillLogits(tokens []int32) ([][]float32, error)
	// SnapshotState captures model state and the current absolute position.
	SnapshotState() ([]byte, int, error)
	// RestoreState restores a snapshot and rewinds to position.
	RestoreState(data []byte, position int) error
}

// Runtime is the execution surface used by the generation loop. The CPU
// reference backend and the HIP backend both implement it.
type Runtime interface {
	Reset()
	Forward(tokens []int32) ([]float32, error)
	Step(token int32) ([]float32, error)
	Capacity() int
	VocabSize() int
	Close() error
}

type cpuRuntime struct {
	model *qwen35.Model
	state *qwen35.State
}

func (r *cpuRuntime) Reset() { r.state.Reset() }
func (r *cpuRuntime) Forward(tokens []int32) ([]float32, error) {
	return r.model.Forward(tokens, r.state)
}
func (r *cpuRuntime) Step(token int32) ([]float32, error) {
	return r.model.Step(token, r.state)
}
func (r *cpuRuntime) Capacity() int  { return r.state.Capacity }
func (r *cpuRuntime) VocabSize() int { return r.model.Config.VocabSize }
func (r *cpuRuntime) Close() error   { return nil }

func (r *cpuRuntime) PrefillLogits(tokens []int32) ([][]float32, error) {
	return r.model.PrefillLogits(tokens, r.state)
}

func (r *cpuRuntime) SnapshotState() ([]byte, int, error) {
	return r.state.Snapshot(), r.state.Pos, nil
}

func (r *cpuRuntime) RestoreState(data []byte, position int) error {
	if err := r.state.Restore(data); err != nil {
		return err
	}
	return r.state.SetPos(position)
}

// Engine owns a runtime, tokenizer, and sampler.
type Engine struct {
	Runtime   Runtime
	Tokenizer *tokenizer.Tokenizer
	Sampler   *Sampler
}

// New creates a CPU reference engine with an F32 KV cache. capacity must not
// exceed the model context length.
func New(model *qwen35.Model, tok *tokenizer.Tokenizer, capacity int, sampler *Sampler) (*Engine, error) {
	return NewWithKVFormat(model, tok, capacity, sampler, qwen35.KVF32)
}

// NewWithKVFormat creates a CPU reference engine with an explicit KV cache
// format.
func NewWithKVFormat(model *qwen35.Model, tok *tokenizer.Tokenizer, capacity int, sampler *Sampler, format qwen35.KVFormat) (*Engine, error) {
	state, err := qwen35.NewStateWithFormat(model.Config, capacity, format)
	if err != nil {
		return nil, err
	}
	runtime := &cpuRuntime{model: model, state: state}
	return NewWithRuntime(runtime, tok, sampler)
}

// Close releases the runtime. It is idempotent.
func (e *Engine) Close() error {
	if e == nil || e.Runtime == nil {
		return nil
	}
	return e.Runtime.Close()
}

// NewWithRuntime creates an engine over an explicit runtime.
func NewWithRuntime(runtime Runtime, tok *tokenizer.Tokenizer, sampler *Sampler) (*Engine, error) {
	if runtime == nil || tok == nil {
		return nil, fmt.Errorf("runtime and tokenizer are required")
	}
	if sampler == nil {
		var err error
		sampler, err = NewSampler(SamplerConfig{}, runtime.VocabSize())
		if err != nil {
			return nil, err
		}
	}
	return &Engine{Runtime: runtime, Tokenizer: tok, Sampler: sampler}, nil
}

// Generate runs one prompt through the model and returns the generated tokens.
func (e *Engine) Generate(ctx context.Context, prompt []int32, opts GenerateOptions) (*GenerateResult, error) {
	if opts.Speculative != nil {
		return e.generateSpeculative(ctx, prompt, opts)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("prompt is empty")
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("max tokens must be positive")
	}
	if len(prompt) > e.Runtime.Capacity() {
		return nil, fmt.Errorf("prompt length %d exceeds context capacity %d",
			len(prompt), e.Runtime.Capacity())
	}
	remaining := e.Runtime.Capacity() - len(prompt)
	if opts.MaxTokens > remaining {
		opts.MaxTokens = remaining
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("no context space remains for generated tokens")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	e.Runtime.Reset()
	result := &GenerateResult{PromptTokens: len(prompt)}
	started := time.Now()
	logits, err := e.Runtime.Forward(prompt)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	result.PrefillTime = time.Since(started)
	if len(prompt) > 0 {
		result.PromptTokensPS = float64(len(prompt)) / result.PrefillTime.Seconds()
	}

	stops := make(map[int32]struct{}, len(opts.StopTokens)+1)
	for _, id := range opts.StopTokens {
		stops[id] = struct{}{}
	}
	if eos := e.Tokenizer.EOSToken(); eos >= 0 && !opts.IgnoreEOS {
		stops[eos] = struct{}{}
	}

	stream := &textStream{}
	result.Tokens = make([]int32, 0, opts.MaxTokens)
	firstTokenAt := time.Time{}
	result.FinishReason = "length"
	for i := 0; i < opts.MaxTokens; i++ {
		if err := ctx.Err(); err != nil {
			result.FinishReason = "cancelled"
			break
		}
		token, err := e.Sampler.Sample(logits)
		if err != nil {
			return nil, err
		}
		if _, stop := stops[token]; stop {
			result.FinishReason = "stop"
			break
		}
		piece, err := e.Tokenizer.DecodeToken(token)
		if err != nil {
			return nil, err
		}
		delta := stream.Feed(piece)
		if firstTokenAt.IsZero() {
			firstTokenAt = time.Now()
			result.TimeToFirst = firstTokenAt.Sub(started)
		}
		result.Tokens = append(result.Tokens, token)
		if delta != "" && opts.OnToken != nil {
			if err := opts.OnToken(token, delta); err != nil {
				if errors.Is(err, ErrStop) {
					result.FinishReason = "stop"
					result.Text = stream.Text()
					result.OutputTokens = len(result.Tokens)
					return result, nil
				}
				return nil, err
			}
		}
		if i+1 == opts.MaxTokens {
			break
		}
		logits, err = e.Runtime.Step(token)
		if err != nil {
			return nil, fmt.Errorf("decode: %w", err)
		}
	}
	if delta := stream.Flush(); delta != "" && opts.OnToken != nil {
		if err := opts.OnToken(-1, delta); err != nil {
			if errors.Is(err, ErrStop) {
				result.FinishReason = "stop"
				result.Text = stream.Text()
				result.OutputTokens = len(result.Tokens)
				return result, nil
			}
			return nil, err
		}
	}
	result.Text = stream.Text()
	result.OutputTokens = len(result.Tokens)
	if result.OutputTokens > 1 && !firstTokenAt.IsZero() {
		result.DecodeTime = time.Since(firstTokenAt)
		if result.DecodeTime > 0 {
			result.DecodeTokensPS = float64(result.OutputTokens-1) / result.DecodeTime.Seconds()
		}
	}
	return result, nil
}
