package serving

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// generationTask drives one request through prefill, sampling, stop handling,
// and text streaming. It implements scheduler.Task.
type generationTask struct {
	id       string
	prompt   []int32
	maxToken int
	seq      Sequence
	sampler  *engine.Sampler
	tok      *tokenizer.Tokenizer
	stops    map[int32]struct{}
	sink     func(token int32, delta string) error
	filter   *stopFilter
	stream   utf8Stream

	promptOffset int
	promptDone   bool
	finished     bool
	reason       string
	failure      error

	pending    int32
	hasPending bool
	initial    []float32

	generated []int32
	text      strings.Builder

	started  time.Time
	firstAt  time.Time
	lastAt   time.Time
	prefill  time.Duration
	decode   time.Duration
	sampling time.Duration

	promptState    []byte
	promptLogits   []byte
	snapshotOK     bool
	initialHit     bool
	initialPending bool
	prefixTokens   int
}

func newGenerationTask(
	id string,
	prompt []int32,
	maxTokens int,
	seq Sequence,
	sampler *engine.Sampler,
	tok *tokenizer.Tokenizer,
	stops []int32,
	stopStrings []string,
	ignoreEOS bool,
	initialLogits []float32,
	initialState []byte,
	initialPosition int,
	sink func(int32, string) error,
) *generationTask {
	stopSet := make(map[int32]struct{}, len(stops)+1)
	for _, token := range stops {
		stopSet[token] = struct{}{}
	}
	if eos := tok.EOSToken(); eos >= 0 && !ignoreEOS {
		stopSet[eos] = struct{}{}
	}
	task := &generationTask{
		id:           id,
		prompt:       prompt,
		maxToken:     maxTokens,
		seq:          seq,
		sampler:      sampler,
		tok:          tok,
		stops:        stopSet,
		sink:         sink,
		filter:       newStopFilter(stopStrings),
		started:      time.Now(),
		promptOffset: initialPosition,
		initial:      initialLogits,
		promptState:  initialState,
		initialHit:   initialPosition > 0,
		prefixTokens: initialPosition,
	}
	if initialPosition == len(prompt) {
		task.promptDone = true
		task.initialPending = true
		// A fully cached prompt already has valid state and logits; do not
		// rewrite the cache entry when this request finishes.
		task.promptLogits = serializeLogits(initialLogits)
	}
	if initialPosition > len(prompt) {
		task.failure = fmt.Errorf("prefix position %d exceeds prompt length %d", initialPosition, len(prompt))
		task.finished = true
		task.reason = "error"
	}
	return task
}

// PrefillOnce processes at most budget prompt tokens. The first token is sampled
// when the final chunk completes.
func (t *generationTask) PrefillOnce(ctx context.Context, budget int) (scheduler.StepStats, bool, error) {
	var stats scheduler.StepStats
	if err := ctx.Err(); err != nil {
		return stats, false, err
	}
	if t.failure != nil {
		return stats, false, t.failure
	}
	if t.finished {
		return stats, true, nil
	}
	if t.initialPending {
		// The complete prompt was served from the prefix cache.
		t.initialPending = false
		elapsed, err := t.sampleAndEmit(t.initial)
		t.sampling += elapsed
		stats.SamplingTime = elapsed
		if err != nil {
			t.failure = err
		}
		return stats, true, t.failure
	}
	if t.promptDone {
		return stats, true, nil
	}
	if budget <= 0 {
		return stats, false, fmt.Errorf("prefill budget must be positive")
	}
	remaining := len(t.prompt) - t.promptOffset
	n := min(budget, remaining)
	final := n == remaining
	start := time.Now()
	logits, err := t.seq.Prefill(t.prompt[t.promptOffset:t.promptOffset+n], final)
	elapsed := time.Since(start)
	t.prefill += elapsed
	stats.PrefillTime = elapsed
	stats.PromptTokens = n
	t.promptOffset += n
	if err != nil {
		t.failure = err
		return stats, false, err
	}
	if !final {
		return stats, false, nil
	}
	t.promptDone = true
	if snapshot, snapshotErr := t.seq.Snapshot(); snapshotErr == nil {
		t.promptState = snapshot
		t.snapshotOK = true
	}
	t.promptLogits = serializeLogits(logits)
	sampleElapsed, err := t.sampleAndEmit(logits)
	t.sampling += sampleElapsed
	stats.SamplingTime = sampleElapsed
	if err != nil {
		t.failure = err
		return stats, true, err
	}
	return stats, true, t.failure
}

// DecodeOnce feeds the pending token and samples the next one.
func (t *generationTask) DecodeOnce(ctx context.Context) (scheduler.StepStats, bool, error) {
	var stats scheduler.StepStats
	if err := ctx.Err(); err != nil {
		return stats, false, err
	}
	if t.failure != nil {
		return stats, false, t.failure
	}
	if t.finished {
		return stats, true, nil
	}
	if !t.hasPending {
		t.failure = fmt.Errorf("decode step has no pending token")
		return stats, false, t.failure
	}
	token := t.pending
	t.hasPending = false
	start := time.Now()
	logits, err := t.seq.Decode(token)
	elapsed := time.Since(start)
	t.decode += elapsed
	stats.DecodeTime = elapsed
	if err != nil {
		t.failure = err
		return stats, false, err
	}
	sampleElapsed, err := t.sampleAndEmit(logits)
	t.sampling += sampleElapsed
	stats.SamplingTime = sampleElapsed
	if err != nil {
		t.failure = err
		return stats, false, err
	}
	return stats, t.finished, t.failure
}

// sampleAndEmit selects a token, applies stop conditions, and streams the
// decoded text. It is called with the logits for the current position.
func (t *generationTask) sampleAndEmit(logits []float32) (time.Duration, error) {
	if len(logits) != t.tok.VocabSize() {
		return 0, fmt.Errorf("logits have %d entries, want %d", len(logits), t.tok.VocabSize())
	}
	start := time.Now()
	token, err := t.sampler.Sample(logits)
	elapsed := time.Since(start)
	if err != nil {
		return elapsed, err
	}
	if _, stop := t.stops[token]; stop {
		t.finish("stop")
		return elapsed, t.failure
	}
	if err := t.emit(token); err != nil {
		return elapsed, err
	}
	if t.firstAt.IsZero() {
		t.firstAt = time.Now()
	}
	t.lastAt = time.Now()
	if len(t.generated) >= t.maxToken {
		t.finish("length")
		return elapsed, t.failure
	}
	t.pending = token
	t.hasPending = true
	return elapsed, nil
}

// emit decodes one token and forwards complete text to the sink.
func (t *generationTask) emit(token int32) error {
	piece, err := t.tok.DecodeToken(token)
	if err != nil {
		return fmt.Errorf("decode token %d: %w", token, err)
	}
	t.generated = append(t.generated, token)
	delta := t.stream.Feed(piece)
	safe, stopped := t.filter.Feed(delta)
	if safe != "" {
		t.text.WriteString(safe)
		if t.sink != nil {
			if err := t.sink(token, safe); err != nil {
				return err
			}
		}
	}
	if stopped {
		t.finish("stop")
	}
	return nil
}

// finish flushes buffered text, marks the request terminal, and records why.
func (t *generationTask) finish(reason string) {
	if t.finished {
		return
	}
	t.finished = true
	t.reason = reason
	if tail := t.filter.Flush(); tail != "" {
		t.text.WriteString(tail)
		if t.sink != nil {
			if err := t.sink(-1, tail); err != nil && t.failure == nil {
				t.failure = err
			}
		}
	}
	if tail := t.stream.Flush(); tail != "" {
		t.text.WriteString(tail)
		if t.sink != nil {
			if err := t.sink(-1, tail); err != nil && t.failure == nil {
				t.failure = err
			}
		}
	}
}

// Outcome implements scheduler.Task.
func (t *generationTask) Outcome() scheduler.Outcome {
	reason := t.reason
	if reason == "" {
		if t.finished {
			reason = "stop"
		} else {
			reason = "cancelled"
		}
	}
	ttft := time.Duration(0)
	if !t.firstAt.IsZero() {
		ttft = t.firstAt.Sub(t.started)
	}
	tpot := time.Duration(0)
	if len(t.generated) > 1 && !t.firstAt.IsZero() && !t.lastAt.IsZero() {
		tpot = t.lastAt.Sub(t.firstAt) / time.Duration(len(t.generated)-1)
	}
	promptTokens := t.promptOffset
	if t.promptDone {
		promptTokens = len(t.prompt)
	}
	return scheduler.Outcome{
		FinishReason:    reason,
		PromptTokens:    promptTokens,
		GeneratedTokens: len(t.generated),
		TimeToFirst:     ttft,
		TPOT:            tpot,
	}
}

// Close releases the sequence. It is idempotent.
func (t *generationTask) Close() error {
	if t.seq == nil {
		return nil
	}
	err := t.seq.Close()
	t.seq = nil
	return err
}

// PrefixReady reports whether this request produced a complete prompt prefix
// that can be cached. The returned slices are owned by the task's cache entry
// after Insert.
func (t *generationTask) PrefixReady() ([]int32, []byte, []byte, bool) {
	if !t.promptDone || !t.snapshotOK {
		return nil, nil, nil, false
	}
	if len(t.prompt) == 0 || len(t.promptLogits) == 0 {
		return nil, nil, nil, false
	}
	return t.prompt, t.promptState, t.promptLogits, true
}

func serializeLogits(logits []float32) []byte {
	if len(logits) == 0 {
		return nil
	}
	data := make([]byte, len(logits)*4)
	for i, value := range logits {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}

func deserializeLogits(data []byte, vocab int) ([]float32, bool) {
	if len(data) != vocab*4 {
		return nil, false
	}
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return logits, true
}
