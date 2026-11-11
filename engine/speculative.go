package engine

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// SpeculativeOptions controls exact greedy speculative decoding. Draft tokens
// come from prompt-lookup n-grams over the request history; no separate draft
// model is required.
type SpeculativeOptions struct {
	// MaxDraft is the maximum number of candidate tokens per verification
	// block. Values above 16 are capped by the HIP runtime.
	MaxDraft int
	// MinMatch is the shortest history suffix used to seed a candidate.
	MinMatch int
}

func (o *SpeculativeOptions) normalize() {
	if o.MaxDraft <= 0 {
		o.MaxDraft = 4
	}
	if o.MaxDraft > 16 {
		o.MaxDraft = 16
	}
	if o.MinMatch <= 0 {
		o.MinMatch = 2
	}
	if o.MinMatch > 8 {
		o.MinMatch = 8
	}
}

// lookupDraft finds the latest earlier occurrence of a history suffix and
// returns the tokens that followed it. The longest suffix wins, and no token
// is proposed beyond the available history.
func lookupDraft(history []int32, maxDraft, minMatch int) []int32 {
	if maxDraft <= 0 || minMatch <= 0 || len(history) <= minMatch {
		return nil
	}
	maxN := min(8, len(history)-1)
	for n := maxN; n >= minMatch; n-- {
		suffix := history[len(history)-n:]
		for start := len(history) - n - 1; start >= 0; start-- {
			match := true
			for i := range n {
				if history[start+i] != suffix[i] {
					match = false
					break
				}
			}
			if !match {
				continue
			}
			follow := start + n
			take := min(maxDraft, len(history)-follow)
			if take <= 0 {
				continue
			}
			return append([]int32(nil), history[follow:follow+take]...)
		}
	}
	return nil
}

func (e *Engine) generateSpeculative(ctx context.Context, prompt []int32, opts GenerateOptions) (*GenerateResult, error) {
	rt, ok := e.Runtime.(SpeculativeRuntime)
	if !ok {
		return nil, fmt.Errorf("speculative decoding is not supported by this runtime")
	}
	if !e.Sampler.Greedy() {
		return nil, fmt.Errorf("speculative decoding requires greedy sampling")
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("prompt is empty")
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("max tokens must be positive")
	}
	if len(prompt) > rt.Capacity() {
		return nil, fmt.Errorf("prompt length %d exceeds context capacity %d",
			len(prompt), rt.Capacity())
	}
	remaining := rt.Capacity() - len(prompt)
	if opts.MaxTokens > remaining {
		opts.MaxTokens = remaining
	}
	if opts.MaxTokens <= 0 {
		return nil, fmt.Errorf("no context space remains for generated tokens")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	so := *opts.Speculative
	so.normalize()

	rt.Reset()
	result := &GenerateResult{PromptTokens: len(prompt)}
	started := time.Now()
	logits, err := rt.Forward(prompt)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	result.PrefillTime = time.Since(started)
	result.PromptTokensPS = float64(len(prompt)) / result.PrefillTime.Seconds()

	stops := make(map[int32]struct{}, len(opts.StopTokens)+1)
	for _, id := range opts.StopTokens {
		stops[id] = struct{}{}
	}
	if eos := e.Tokenizer.EOSToken(); eos >= 0 && !opts.IgnoreEOS {
		stops[eos] = struct{}{}
	}
	stream := &textStream{}
	result.Tokens = make([]int32, 0, opts.MaxTokens)
	history := append([]int32(nil), prompt...)
	firstTokenAt := time.Time{}
	result.FinishReason = "length"
	stopped := false

	emit := func(token int32) (bool, error) {
		if _, stop := stops[token]; stop {
			result.FinishReason = "stop"
			return true, nil
		}
		piece, err := e.Tokenizer.DecodeToken(token)
		if err != nil {
			return false, err
		}
		delta := stream.Feed(piece)
		if firstTokenAt.IsZero() {
			firstTokenAt = time.Now()
			result.TimeToFirst = firstTokenAt.Sub(started)
		}
		result.Tokens = append(result.Tokens, token)
		history = append(history, token)
		if delta != "" && opts.OnToken != nil {
			if err := opts.OnToken(token, delta); err != nil {
				if errors.Is(err, ErrStop) {
					result.FinishReason = "stop"
					return true, nil
				}
				return false, err
			}
		}
		return false, nil
	}

	for len(result.Tokens) < opts.MaxTokens {
		if err := ctx.Err(); err != nil {
			result.FinishReason = "cancelled"
			break
		}
		limit := min(so.MaxDraft, opts.MaxTokens-len(result.Tokens))
		limit = min(limit, rt.Capacity()-len(history))
		if limit <= 0 {
			break
		}
		draft := lookupDraft(history, limit, so.MinMatch)
		if len(draft) == 0 {
			token := argmaxLogit(logits)
			stop, err := emit(token)
			if err != nil {
				return nil, err
			}
			if stop {
				stopped = true
				break
			}
			if len(result.Tokens) >= opts.MaxTokens {
				break
			}
			logits, err = rt.Step(token)
			if err != nil {
				return nil, fmt.Errorf("decode: %w", err)
			}
			continue
		}
		snapshot, position, err := rt.SnapshotState()
		if err != nil {
			return nil, fmt.Errorf("speculative snapshot: %w", err)
		}
		candidates, err := rt.PrefillLogits(draft)
		if err != nil {
			return nil, fmt.Errorf("speculative verify: %w", err)
		}
		result.DraftTokens += len(draft)
		accepted := 0
		before := logits
		correction := int32(-1)
		for i := range draft {
			want := argmaxLogit(before)
			if want == draft[i] {
				accepted++
				before = candidates[i]
				continue
			}
			correction = want
			break
		}
		result.AcceptedDraft += accepted
		var emitted []int32
		var nextLogits []float32
		if accepted == len(draft) {
			emitted = draft
			nextLogits = candidates[len(candidates)-1]
		} else {
			if err := rt.RestoreState(snapshot, position); err != nil {
				return nil, fmt.Errorf("speculative rollback: %w", err)
			}
			replay := make([]int32, 0, accepted+1)
			replay = append(replay, draft[:accepted]...)
			replay = append(replay, correction)
			nextLogits, err = rt.Forward(replay)
			if err != nil {
				return nil, fmt.Errorf("speculative replay: %w", err)
			}
			emitted = replay
		}
		for _, token := range emitted {
			stop, err := emit(token)
			if err != nil {
				return nil, err
			}
			if stop {
				stopped = true
				break
			}
			if len(result.Tokens) >= opts.MaxTokens {
				break
			}
		}
		if stopped || len(result.Tokens) >= opts.MaxTokens {
			break
		}
		logits = nextLogits
	}
	if stopped {
		result.FinishReason = "stop"
	}
	if delta := stream.Flush(); delta != "" && opts.OnToken != nil {
		if err := opts.OnToken(-1, delta); err != nil {
			if errors.Is(err, ErrStop) {
				result.FinishReason = "stop"
			} else {
				return nil, err
			}
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

func argmaxLogit(logits []float32) int32 {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return int32(best)
}
