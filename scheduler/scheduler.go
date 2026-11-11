// Package scheduler implements admission, continuous batching, prefill/decode
// interleaving, cancellation, and request metrics for one model executor.
//
// The scheduler never touches device memory directly. It asks a Budget whether
// a submission is plausible, then builds a Task through the submission's Build
// callback. A Task executes one bounded prefill opportunity or one decode step
// per call, so the scheduler can recompose the batch between steps. Model
// execution is expected to be serialized by the backend.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrQueueFull reports that the pending queue reached its limit.
	ErrQueueFull = errors.New("scheduler: queue is full")
	// ErrCapacity reports that the request cannot fit the KV or context budget.
	ErrCapacity = errors.New("scheduler: insufficient capacity")
	// ErrClosed reports submission after shutdown began.
	ErrClosed = errors.New("scheduler: closed")
)

// Budget performs admission checks against the model's KV and context capacity.
// It is advisory: the backend must still fail explicitly if allocation fails
// when the task is built.
type Budget interface {
	CheckAdmission(promptTokens, outputTokens int) error
}

// StepStats reports work performed by one task call. Durations cover only the
// corresponding phase; SamplingTime is part of DecodeTime.
type StepStats struct {
	PromptTokens int
	OutputTokens int
	PrefillTime  time.Duration
	DecodeTime   time.Duration
	SamplingTime time.Duration
}

// Outcome is the terminal request summary returned by a Task.
type Outcome struct {
	FinishReason    string
	PromptTokens    int
	GeneratedTokens int
	TimeToFirst     time.Duration
	TPOT            time.Duration
}

// Task is one request's generation state machine. PrefillOnce processes at most
// budget prompt tokens; done reports that the prompt is complete and decoding
// can start. DecodeOnce runs one decode step; finished reports that the request
// reached a stop condition or its token limit. Both must observe ctx.
type Task interface {
	PrefillOnce(ctx context.Context, budget int) (StepStats, bool, error)
	DecodeOnce(ctx context.Context) (StepStats, bool, error)
	Outcome() Outcome
	Close() error
}

// Submission is one request offered to the scheduler. Build is called only
// after admission and must create the Task without performing model work.
type Submission struct {
	ID           string
	PromptTokens int
	MaxOutput    int
	Build        func(ctx context.Context) (Task, error)
}

// Config controls scheduling policy. Zero values select the documented
// defaults.
type Config struct {
	MaxActive       int
	MaxQueue        int
	MaxBatch        int
	PrefillChunk    int
	PrefillPerCycle int
	DecodePerCycle  int
	RequestTimeout  time.Duration
	QueueTimeout    time.Duration
	Budget          Budget
	OnStep          func(Step)
}

// Step records the explicit batch composition of one scheduler iteration.
type Step struct {
	At      time.Time
	Decode  []string
	Prefill []string
}

func (c *Config) applyDefaults() {
	if c.MaxActive <= 0 {
		c.MaxActive = 4
	}
	if c.MaxQueue <= 0 {
		c.MaxQueue = 64
	}
	if c.MaxBatch <= 0 {
		c.MaxBatch = c.MaxActive
	}
	if c.PrefillChunk <= 0 {
		c.PrefillChunk = 512
	}
	if c.PrefillPerCycle <= 0 {
		c.PrefillPerCycle = 1
	}
	if c.DecodePerCycle <= 0 {
		c.DecodePerCycle = c.MaxActive
	}
	if c.RequestTimeout <= 0 {
		c.RequestTimeout = 5 * time.Minute
	}
}

// Scheduler is a single-model continuous batching controller. Run must be
// called exactly once; Submit and Close may be called concurrently with it.
type Scheduler struct {
	cfg    Config
	seq    atomic.Uint64
	wake   chan struct{}
	done   chan struct{}
	close  chan struct{}
	closed chan struct{}

	mu       sync.Mutex
	queue    []*pending
	active   []*running
	rr       int
	stopping bool
	draining bool
	metrics  metricsState
}

type pending struct {
	sub           Submission
	handle        *Handle
	ctx           context.Context
	cancel        context.CancelFunc
	enqueued      time.Time
	queueTimedOut bool
}

type running struct {
	*pending
	task        Task
	admitted    time.Time
	prefillDone bool
}

// New validates the configuration and creates a scheduler. The scheduler does
// not start until Run is called.
func New(cfg Config) (*Scheduler, error) {
	cfg.applyDefaults()
	if cfg.MaxBatch > cfg.MaxActive {
		cfg.MaxBatch = cfg.MaxActive
	}
	if cfg.PrefillChunk <= 0 || cfg.MaxQueue <= 0 {
		return nil, fmt.Errorf("scheduler: invalid configuration")
	}
	s := &Scheduler{
		cfg:    cfg,
		wake:   make(chan struct{}, 1),
		done:   make(chan struct{}),
		close:  make(chan struct{}),
		closed: make(chan struct{}),
	}
	return s, nil
}

// Submit validates admission, attaches a timeout, and queues the request. A
// nil error means the request was accepted, not that it has finished. The
// returned handle reports admission and completion.
func (s *Scheduler) Submit(parent context.Context, sub Submission) (*Handle, error) {
	if parent == nil {
		parent = context.Background()
	}
	if err := parent.Err(); err != nil {
		return nil, err
	}
	if sub.Build == nil {
		return nil, fmt.Errorf("scheduler: submission has no task builder")
	}
	if sub.PromptTokens <= 0 {
		return nil, fmt.Errorf("scheduler: prompt must not be empty")
	}
	if sub.MaxOutput <= 0 {
		return nil, fmt.Errorf("scheduler: max output must be positive")
	}
	if s.cfg.Budget != nil {
		if err := s.cfg.Budget.CheckAdmission(sub.PromptTokens, sub.MaxOutput); err != nil {
			s.mu.Lock()
			s.metrics.capacityRejected++
			s.mu.Unlock()
			return nil, err
		}
	}
	if sub.ID == "" {
		sub.ID = fmt.Sprintf("req-%d", s.seq.Add(1))
	}
	ctx, cancel := context.WithTimeout(parent, s.cfg.RequestTimeout)

	s.mu.Lock()
	if s.stopping || s.draining {
		s.mu.Unlock()
		cancel()
		return nil, ErrClosed
	}
	if len(s.queue) >= s.cfg.MaxQueue {
		s.mu.Unlock()
		cancel()
		s.mu.Lock()
		s.metrics.queueFull++
		s.mu.Unlock()
		return nil, ErrQueueFull
	}
	handle := newHandle(sub.ID, ctx, cancel)
	item := &pending{
		sub:      sub,
		handle:   handle,
		ctx:      ctx,
		cancel:   cancel,
		enqueued: time.Now(),
	}
	s.queue = append(s.queue, item)
	s.metrics.submissions++
	s.metrics.queueDepth = len(s.queue)
	s.mu.Unlock()
	s.signal()
	return handle, nil
}

// Run drives the scheduling loop until Close or the parent context ends. It
// must be called once, typically from a dedicated goroutine.
func (s *Scheduler) Run(ctx context.Context) {
	defer close(s.done)
	for {
		if ctx.Err() != nil {
			s.stopAccepting()
		}
		if !s.iterate(ctx) {
			if s.finished() {
				return
			}
			s.wait(ctx)
		}
	}
}

// stopAccepting cancels queued and active work after the run context ends.
func (s *Scheduler) stopAccepting() {
	s.mu.Lock()
	if !s.stopping {
		s.stopping = true
		close(s.close)
		for _, item := range s.queue {
			item.cancel()
		}
		for _, item := range s.active {
			item.cancel()
		}
	}
	s.mu.Unlock()
	s.signal()
}

// Close stops admission, cancels queued and active requests, waits for the
// controller to drain, and releases every task. It is idempotent.
func (s *Scheduler) Close() error {
	s.mu.Lock()
	if !s.stopping {
		s.stopping = true
		close(s.close)
		for _, item := range s.queue {
			item.cancel()
		}
		for _, item := range s.active {
			item.cancel()
		}
	}
	s.mu.Unlock()
	s.signal()
	<-s.done
	return nil
}

// Shutdown stops admission and waits up to ctx for active requests to finish
// naturally. Requests still running at the deadline are cancelled. Pending
// requests are failed with ErrClosed.
func (s *Scheduler) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	if !s.stopping {
		s.draining = true
		for _, item := range s.queue {
			item.cancel()
		}
	}
	s.mu.Unlock()
	s.signal()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		s.mu.Lock()
		empty := len(s.active) == 0 && len(s.queue) == 0
		s.mu.Unlock()
		if empty {
			return s.Close()
		}
		select {
		case <-ctx.Done():
			_ = s.Close()
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// Handle is the caller-side request view. Tokens are delivered by the Task's
// sink, not by the handle.
type Handle struct {
	id     string
	ctx    context.Context
	cancel context.CancelFunc

	done      chan struct{}
	mu        sync.Mutex
	err       error
	reason    string
	queueWait time.Duration
}

func newHandle(id string, ctx context.Context, cancel context.CancelFunc) *Handle {
	return &Handle{id: id, ctx: ctx, cancel: cancel, done: make(chan struct{})}
}

// ID returns the stable request identifier.
func (h *Handle) ID() string { return h.id }

// Context returns the request context. It is cancelled by Cancel, timeout, or
// scheduler shutdown.
func (h *Handle) Context() context.Context { return h.ctx }

// Done closes when the request reaches a terminal state.
func (h *Handle) Done() <-chan struct{} { return h.done }

// Cancel requests cancellation. It is idempotent.
func (h *Handle) Cancel() {
	if h.cancel != nil {
		h.cancel()
	}
}

// Err returns the terminal error, if any.
func (h *Handle) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

// QueueWait returns the time between submission and admission.
func (h *Handle) QueueWait() time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.queueWait
}

func (h *Handle) setQueueWait(wait time.Duration) {
	h.mu.Lock()
	h.queueWait = wait
	h.mu.Unlock()
}

// FinishReason returns "stop", "length", "cancelled", "timeout", or "error".
func (h *Handle) FinishReason() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.reason
}

func (h *Handle) finish(reason string, err error) {
	h.mu.Lock()
	if h.reason != "" {
		h.mu.Unlock()
		return
	}
	h.reason = reason
	h.err = err
	h.mu.Unlock()
	if h.cancel != nil {
		h.cancel()
	}
	close(h.done)
}

func (s *Scheduler) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *Scheduler) finished() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return (s.stopping || s.draining) && len(s.queue) == 0 && len(s.active) == 0
}

// wait blocks until new work arrives, the next deadline expires, or shutdown.
func (s *Scheduler) wait(ctx context.Context) {
	delay := 50 * time.Millisecond
	now := time.Now()
	s.mu.Lock()
	for _, item := range s.queue {
		if deadline, ok := item.ctx.Deadline(); ok {
			if d := deadline.Sub(now); d < delay {
				delay = d
			}
		}
	}
	for _, item := range s.active {
		if deadline, ok := item.ctx.Deadline(); ok {
			if d := deadline.Sub(now); d < delay {
				delay = d
			}
		}
	}
	s.mu.Unlock()
	if delay < time.Millisecond {
		delay = time.Millisecond
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-s.wake:
	case <-s.close:
	case <-ctx.Done():
	case <-timer.C:
	}
}

// iterate runs one admission/decode/prefill cycle. It reports false when there
// was no work to perform.
func (s *Scheduler) iterate(ctx context.Context) bool {
	now := time.Now()

	// Expire queued requests and admit new ones under the lock.
	var admit []*running
	var expired []*pending
	s.mu.Lock()
	if s.stopping || s.draining {
		for len(s.queue) > 0 {
			item := s.queue[0]
			s.queue = s.queue[1:]
			expired = append(expired, item)
		}
	} else {
		keep := s.queue[:0]
		for _, item := range s.queue {
			if item.ctx.Err() != nil {
				expired = append(expired, item)
				continue
			}
			if s.cfg.QueueTimeout > 0 && now.Sub(item.enqueued) >= s.cfg.QueueTimeout {
				item.queueTimedOut = true
				expired = append(expired, item)
				continue
			}
			keep = append(keep, item)
		}
		s.queue = keep
		for len(s.queue) > 0 && len(s.active) < s.cfg.MaxActive {
			item := s.queue[0]
			s.queue = s.queue[1:]
			run := &running{pending: item, admitted: now}
			s.active = append(s.active, run)
			admit = append(admit, run)
			s.metrics.admitted++
			wait := now.Sub(item.enqueued)
			s.metrics.queueWaitTotal += wait
			item.handle.setQueueWait(wait)
		}
		s.metrics.queueDepth = len(s.queue)
		s.metrics.active = len(s.active)
		if s.metrics.active > s.metrics.maxActiveObserved {
			s.metrics.maxActiveObserved = s.metrics.active
		}
	}
	s.mu.Unlock()

	for _, item := range expired {
		reason, err := contextOutcome(item.ctx)
		if item.queueTimedOut {
			reason, err = "timeout", context.DeadlineExceeded
		}
		if s.isStopping() {
			reason = "cancelled"
			err = ErrClosed
		}
		s.mu.Lock()
		s.metrics.queueDepth = len(s.queue)
		if reason == "timeout" {
			s.metrics.timeouts++
		} else {
			s.metrics.cancelled++
		}
		s.mu.Unlock()
		item.handle.finish(reason, err)
	}

	for _, run := range admit {
		task, err := run.sub.Build(run.ctx)
		if err != nil {
			s.mu.Lock()
			s.removeActiveLocked(run)
			s.metrics.failures++
			s.mu.Unlock()
			reason, termErr := "error", fmt.Errorf("build task: %w", err)
			if errors.Is(err, ErrCapacity) {
				reason = "capacity"
				s.metrics.capacityRejected++
			}
			run.handle.finish(reason, termErr)
			continue
		}
		run.task = task
	}

	// Expire active requests whose context ended.
	s.mu.Lock()
	var canceled []*running
	for _, run := range s.active {
		if run.task != nil && run.ctx.Err() != nil {
			canceled = append(canceled, run)
		}
	}
	s.mu.Unlock()
	for _, run := range canceled {
		reason, err := contextOutcome(run.ctx)
		s.finishRun(run, reason, err, true)
	}

	s.mu.Lock()
	hasActive := len(s.active) > 0
	s.mu.Unlock()
	if !hasActive {
		return false
	}
	// Compose the decode batch first, then prefill opportunities. Model calls
	// happen without the scheduler lock.
	s.mu.Lock()
	order := s.rotationLocked()
	s.mu.Unlock()

	step := Step{At: time.Now()}
	decoded := 0
	for _, run := range order {
		if decoded >= s.cfg.DecodePerCycle || decoded >= s.cfg.MaxBatch {
			break
		}
		s.mu.Lock()
		ready := run.prefillDone && run.task != nil && s.containsLocked(run)
		s.mu.Unlock()
		if !ready {
			continue
		}
		if run.ctx.Err() != nil {
			s.finishRun(run, "cancelled", run.ctx.Err(), true)
			continue
		}
		stats, finished, err := run.task.DecodeOnce(run.ctx)
		s.recordStep(stats)
		s.mu.Lock()
		s.metrics.batchSize = decoded + 1
		s.mu.Unlock()
		decoded++
		step.Decode = append(step.Decode, run.sub.ID)
		if err != nil {
			if ctxErr := run.ctx.Err(); ctxErr != nil {
				reason, termErr := contextOutcome(run.ctx)
				s.finishRun(run, reason, termErr, true)
			} else {
				s.finishRun(run, "error", err, true)
			}
			continue
		}
		if finished {
			s.finishRun(run, run.task.Outcome().FinishReason, nil, false)
		}
	}

	prefilled := 0
	for _, run := range order {
		if prefilled >= s.cfg.PrefillPerCycle {
			break
		}
		s.mu.Lock()
		pendingPrefill := !run.prefillDone && run.task != nil && s.containsLocked(run)
		s.mu.Unlock()
		if !pendingPrefill {
			continue
		}
		if run.ctx.Err() != nil {
			s.finishRun(run, "cancelled", run.ctx.Err(), true)
			continue
		}
		stats, done, err := run.task.PrefillOnce(run.ctx, s.cfg.PrefillChunk)
		s.recordStep(stats)
		prefilled++
		step.Prefill = append(step.Prefill, run.sub.ID)
		if err != nil {
			if ctxErr := run.ctx.Err(); ctxErr != nil {
				reason, termErr := contextOutcome(run.ctx)
				s.finishRun(run, reason, termErr, true)
			} else {
				s.finishRun(run, "error", err, true)
			}
			continue
		}
		if done {
			s.mu.Lock()
			run.prefillDone = true
			s.mu.Unlock()
		}
	}
	if len(step.Decode) > 0 || len(step.Prefill) > 0 {
		s.mu.Lock()
		s.metrics.lastBatch = len(step.Decode) + len(step.Prefill)
		if s.metrics.lastBatch > s.metrics.maxBatchObserved {
			s.metrics.maxBatchObserved = s.metrics.lastBatch
		}
		s.mu.Unlock()
		if s.cfg.OnStep != nil {
			s.cfg.OnStep(step)
		}
	}
	s.mu.Lock()
	s.metrics.active = len(s.active)
	s.mu.Unlock()
	return decoded > 0 || prefilled > 0 || len(admit) > 0
}

// rotationLocked returns active runs starting at the round-robin cursor. The
// cursor advances so every request gets a decode opportunity.
func (s *Scheduler) rotationLocked() []*running {
	if len(s.active) == 0 {
		return nil
	}
	if s.rr >= len(s.active) {
		s.rr = 0
	}
	out := make([]*running, 0, len(s.active))
	out = append(out, s.active[s.rr:]...)
	out = append(out, s.active[:s.rr]...)
	s.rr++
	if s.rr >= len(s.active) {
		s.rr = 0
	}
	return out
}

func (s *Scheduler) containsLocked(target *running) bool {
	for _, run := range s.active {
		if run == target {
			return true
		}
	}
	return false
}

func (s *Scheduler) removeActiveLocked(target *running) {
	for i, run := range s.active {
		if run == target {
			s.active = append(s.active[:i], s.active[i+1:]...)
			if s.rr > i {
				s.rr--
			}
			break
		}
	}
}

// finishRun removes a request, closes its task, records its outcome, and
// releases its context.
func (s *Scheduler) finishRun(run *running, reason string, err error, cancel bool) {
	s.mu.Lock()
	if !s.containsLocked(run) {
		s.mu.Unlock()
		return
	}
	s.removeActiveLocked(run)
	s.metrics.active = len(s.active)
	canceled := reason == "cancelled" || reason == "timeout"
	if reason == "cancelled" {
		s.metrics.cancelled++
	}
	if reason == "timeout" {
		s.metrics.timeouts++
	}
	if err != nil && reason == "error" {
		s.metrics.failures++
	}
	outcome := run.task.Outcome()
	if !canceled && err == nil {
		switch reason {
		case "stop", "length":
			s.metrics.completed++
		}
	}
	if outcome.TimeToFirst > 0 {
		s.metrics.ttft.add(outcome.TimeToFirst)
	}
	if outcome.TPOT > 0 {
		s.metrics.tpot.add(outcome.TPOT)
	}
	s.metrics.promptTokens += uint64(outcome.PromptTokens)
	s.metrics.generatedTokens += uint64(outcome.GeneratedTokens)
	s.mu.Unlock()

	if cancel {
		run.cancel()
	}
	if run.task != nil {
		_ = run.task.Close()
	}
	run.handle.finish(reason, err)
	if run.cancel != nil {
		run.cancel()
	}
	s.signal()
}

func (s *Scheduler) recordStep(stats StepStats) {
	s.mu.Lock()
	s.metrics.prefillTotal += stats.PrefillTime
	s.metrics.decodeTotal += stats.DecodeTime
	s.metrics.samplingTotal += stats.SamplingTime
	s.metrics.steps++
	s.mu.Unlock()
}

func (s *Scheduler) isStopping() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopping
}

func contextOutcome(ctx context.Context) (string, error) {
	err := ctx.Err()
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout", err
	}
	if err == nil {
		err = ErrClosed
	}
	return "cancelled", err
}
