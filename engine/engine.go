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
}

// Engine owns a model, a reusable request state, and a sampler.
type Engine struct {
	Model     *qwen35.Model
	Tokenizer *tokenizer.Tokenizer
	State     *qwen35.State
	Sampler   *Sampler
}

// New creates an engine for a model and tokenizer. capacity must not exceed
// the model context length.
func New(model *qwen35.Model, tok *tokenizer.Tokenizer, capacity int, sampler *Sampler) (*Engine, error) {
	state, err := qwen35.NewState(model.Config, capacity)
	if err != nil {
		return nil, err
	}
	if sampler == nil {
		sampler, err = NewSampler(SamplerConfig{}, model.Config.VocabSize)
		if err != nil {
			return nil, err
		}
	}
	return &Engine{Model: model, Tokenizer: tok, State: state, Sampler: sampler}, nil
}

// Generate runs one prompt through the model and returns the generated tokens.
func (e *Engine) Generate(ctx context.Context, prompt []int32, opts GenerateOptions) (*GenerateResult, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("prompt is empty")
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("max tokens must be positive")
	}
	if len(prompt) > e.State.Capacity {
		return nil, fmt.Errorf("prompt length %d exceeds context capacity %d",
			len(prompt), e.State.Capacity)
	}
	remaining := e.State.Capacity - len(prompt)
	if opts.MaxTokens > remaining {
		opts.MaxTokens = remaining
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("no context space remains for generated tokens")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	e.State.Reset()
	result := &GenerateResult{PromptTokens: len(prompt)}
	started := time.Now()
	logits, err := e.Model.Forward(prompt, e.State)
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
		piece, err := e.Tokenizer.Decode([]int32{token})
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
		logits, err = e.Model.Step(token, e.State)
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
