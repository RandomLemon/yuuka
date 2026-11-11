package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

type fakeTask struct {
	id        string
	prompt    int
	maxOutput int

	mu        sync.Mutex
	remaining int
	generated int
	closed    int
	calls     []string
	block     <-chan struct{}
	reason    string
	stepDelay time.Duration
}

func newFakeTask(id string, prompt, maxOutput int) *fakeTask {
	return &fakeTask{id: id, prompt: prompt, maxOutput: maxOutput, remaining: prompt, reason: "length"}
}

func (t *fakeTask) PrefillOnce(ctx context.Context, budget int) (StepStats, bool, error) {
	t.mu.Lock()
	block := t.block
	t.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return StepStats{}, false, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return StepStats{}, false, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	n := min(budget, t.remaining)
	t.remaining -= n
	t.calls = append(t.calls, fmt.Sprintf("prefill:%d", n))
	if t.stepDelay > 0 {
		time.Sleep(t.stepDelay)
	}
	return StepStats{PromptTokens: n, PrefillTime: time.Millisecond}, t.remaining == 0, nil
}

func (t *fakeTask) DecodeOnce(ctx context.Context) (StepStats, bool, error) {
	t.mu.Lock()
	block := t.block
	t.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return StepStats{}, false, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return StepStats{}, false, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.generated++
	t.calls = append(t.calls, "decode")
	if t.stepDelay > 0 {
		time.Sleep(t.stepDelay)
	}
	return StepStats{OutputTokens: 1, DecodeTime: time.Millisecond, SamplingTime: 500 * time.Microsecond},
		t.generated >= t.maxOutput, nil
}

func (t *fakeTask) Outcome() Outcome {
	t.mu.Lock()
	defer t.mu.Unlock()
	reason := t.reason
	if t.generated >= t.maxOutput {
		reason = "length"
	}
	return Outcome{
		FinishReason:    reason,
		PromptTokens:    t.prompt,
		GeneratedTokens: t.generated,
		TimeToFirst:     2 * time.Millisecond,
		TPOT:            time.Millisecond,
	}
}

func (t *fakeTask) Close() error {
	t.mu.Lock()
	t.closed++
	t.mu.Unlock()
	return nil
}

func (t *fakeTask) callsCopy() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.calls...)
}

func (t *fakeTask) closedCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.closed
}

type rejectingBudget struct{ err error }

func (b rejectingBudget) CheckAdmission(int, int) error { return b.err }

func newScheduler(t *testing.T, cfg Config) *Scheduler {
	t.Helper()
	sched, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return sched
}

func startScheduler(t *testing.T, sched *Scheduler) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		sched.Run(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		_ = sched.Close()
		<-done
	})
}

func runScheduler(t *testing.T, cfg Config) *Scheduler {
	t.Helper()
	sched := newScheduler(t, cfg)
	startScheduler(t, sched)
	return sched
}

func waitForActive(t *testing.T, sched *Scheduler, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for sched.Metrics().Active < want {
		if time.Now().After(deadline) {
			t.Fatalf("scheduler never reached %d active requests", want)
		}
		time.Sleep(time.Millisecond)
	}
}

func waitHandle(t *testing.T, h *Handle, timeout time.Duration) {
	t.Helper()
	select {
	case <-h.Done():
	case <-time.After(timeout):
		t.Fatalf("request %s did not finish", h.ID())
	}
}

func TestSubmitValidationAndBudget(t *testing.T) {
	sched := runScheduler(t, Config{MaxActive: 1, MaxQueue: 1})
	if _, err := sched.Submit(context.Background(), Submission{PromptTokens: 0, MaxOutput: 1, Build: func(context.Context) (Task, error) { return newFakeTask("x", 1, 1), nil }}); err == nil {
		t.Fatal("empty prompt must fail")
	}
	if _, err := sched.Submit(context.Background(), Submission{PromptTokens: 1, MaxOutput: 0, Build: func(context.Context) (Task, error) { return newFakeTask("x", 1, 1), nil }}); err == nil {
		t.Fatal("zero output must fail")
	}
	if _, err := sched.Submit(context.Background(), Submission{PromptTokens: 1, MaxOutput: 1}); err == nil {
		t.Fatal("missing builder must fail")
	}

	reject := runScheduler(t, Config{MaxActive: 1, MaxQueue: 1, Budget: rejectingBudget{err: ErrCapacity}})
	if _, err := reject.Submit(context.Background(), Submission{PromptTokens: 1, MaxOutput: 1, Build: func(context.Context) (Task, error) { return newFakeTask("x", 1, 1), nil }}); !errors.Is(err, ErrCapacity) {
		t.Fatalf("budget error = %v, want ErrCapacity", err)
	}
}

func TestQueueLimitCancellationAndCleanup(t *testing.T) {
	sched := runScheduler(t, Config{MaxActive: 1, MaxQueue: 1, RequestTimeout: time.Minute})
	block := make(chan struct{})
	first := newFakeTask("first", 2, 5)
	first.block = block
	h1, err := sched.Submit(context.Background(), Submission{
		ID: "first", PromptTokens: 2, MaxOutput: 5,
		Build: func(context.Context) (Task, error) { return first, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForActive(t, sched, 1)
	second := newFakeTask("second", 2, 5)
	h2, err := sched.Submit(context.Background(), Submission{
		ID: "second", PromptTokens: 2, MaxOutput: 5,
		Build: func(context.Context) (Task, error) { return second, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	third := newFakeTask("third", 2, 5)
	if _, err := sched.Submit(context.Background(), Submission{
		ID: "third", PromptTokens: 2, MaxOutput: 5,
		Build: func(context.Context) (Task, error) { return third, nil },
	}); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("third submit error = %v, want ErrQueueFull", err)
	}
	h2.Cancel()
	// A queued cancellation is observed between model steps, so release the
	// active task first.
	close(block)
	waitHandle(t, h2, 2*time.Second)
	if h2.FinishReason() != "cancelled" {
		t.Fatalf("second finish reason = %q", h2.FinishReason())
	}
	waitHandle(t, h1, 2*time.Second)
	if h1.FinishReason() != "length" {
		t.Fatalf("first finish reason = %q", h1.FinishReason())
	}
	if first.closedCount() != 1 {
		t.Fatalf("first task closed %d times", first.closedCount())
	}
	deadline := time.Now().Add(time.Second)
	for {
		metrics := sched.Metrics()
		if metrics.Active == 0 && metrics.QueueDepth == 0 {
			if metrics.Completed != 1 {
				t.Fatalf("completed = %d, want 1", metrics.Completed)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("scheduler did not drain: %+v", metrics)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestContinuousBatchingInterleaving(t *testing.T) {
	var mu sync.Mutex
	var steps []Step
	sched := newScheduler(t, Config{
		MaxActive:       2,
		MaxQueue:        4,
		MaxBatch:        2,
		PrefillChunk:    2,
		PrefillPerCycle: 1,
		DecodePerCycle:  2,
		RequestTimeout:  time.Minute,
		OnStep: func(step Step) {
			mu.Lock()
			steps = append(steps, step)
			mu.Unlock()
		},
	})
	tasks := map[string]*fakeTask{}
	handles := map[string]*Handle{}
	for _, id := range []string{"a", "b"} {
		task := newFakeTask(id, 4, 3)
		tasks[id] = task
		h, err := sched.Submit(context.Background(), Submission{
			ID: id, PromptTokens: 4, MaxOutput: 3,
			Build: func(context.Context) (Task, error) { return task, nil },
		})
		if err != nil {
			t.Fatal(err)
		}
		handles[id] = h
	}
	// Start only after both submissions are queued so the batching assertion
	// does not depend on goroutine startup timing.
	startScheduler(t, sched)
	for _, id := range []string{"a", "b"} {
		waitHandle(t, handles[id], 3*time.Second)
		if handles[id].FinishReason() != "length" {
			t.Fatalf("%s finish reason = %q", id, handles[id].FinishReason())
		}
		task := tasks[id]
		calls := task.callsCopy()
		prefills, decodes := 0, 0
		for _, call := range calls {
			switch call {
			case "decode":
				decodes++
			default:
				prefills++
			}
		}
		if prefills != 2 {
			t.Fatalf("%s prefill calls = %d (%v), want 2", id, prefills, calls)
		}
		if decodes != 3 {
			t.Fatalf("%s decode calls = %d (%v), want 3", id, decodes, calls)
		}
	}
	mu.Lock()
	observed := append([]Step(nil), steps...)
	mu.Unlock()
	if len(observed) == 0 {
		t.Fatal("no scheduling steps recorded")
	}
	sawDecode, sawPrefill := false, false
	sawBoth := map[string]bool{}
	for _, step := range observed {
		if len(step.Decode) > 0 {
			sawDecode = true
			for _, id := range step.Decode {
				sawBoth["decode-"+id] = true
			}
		}
		if len(step.Prefill) > 0 {
			sawPrefill = true
			for _, id := range step.Prefill {
				sawBoth["prefill-"+id] = true
			}
		}
	}
	if !sawDecode || !sawPrefill {
		t.Fatalf("expected interleaved prefill and decode steps: %+v", observed)
	}
	for _, key := range []string{"decode-a", "decode-b", "prefill-a", "prefill-b"} {
		if !sawBoth[key] {
			t.Fatalf("missing %s in steps: %+v", key, observed)
		}
	}
	metrics := sched.Metrics()
	if metrics.Completed != 2 || metrics.GeneratedTokens != 6 || metrics.PromptTokens != 8 {
		t.Fatalf("metrics = %+v", metrics)
	}
	if metrics.TTFT.Count != 2 || metrics.TPOT.Count != 2 {
		t.Fatalf("latency summaries = %+v", metrics)
	}
	if metrics.MaxBatchObserved < 2 {
		t.Fatalf("max batch observed = %d", metrics.MaxBatchObserved)
	}
}

func TestRequestTimeoutReleasesTask(t *testing.T) {
	sched := runScheduler(t, Config{MaxActive: 2, MaxQueue: 2, RequestTimeout: 20 * time.Millisecond})
	block := make(chan struct{})
	task := newFakeTask("slow", 2, 2)
	task.block = block
	h, err := sched.Submit(context.Background(), Submission{
		ID: "slow", PromptTokens: 2, MaxOutput: 2,
		Build: func(context.Context) (Task, error) { return task, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	waitHandle(t, h, time.Second)
	if h.FinishReason() != "timeout" {
		t.Fatalf("finish reason = %q, want timeout", h.FinishReason())
	}
	if !errors.Is(h.Err(), context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", h.Err())
	}
	if task.closedCount() != 1 {
		t.Fatalf("task closed %d times", task.closedCount())
	}
	close(block)
}

func TestQueueTimeout(t *testing.T) {
	sched := runScheduler(t, Config{
		MaxActive: 1, MaxQueue: 2, QueueTimeout: 15 * time.Millisecond, RequestTimeout: time.Minute,
	})
	first := newFakeTask("first", 4, 2)
	first.stepDelay = 40 * time.Millisecond
	h1, err := sched.Submit(context.Background(), Submission{
		ID: "first", PromptTokens: 4, MaxOutput: 2,
		Build: func(context.Context) (Task, error) { return first, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForActive(t, sched, 1)
	second := newFakeTask("second", 2, 2)
	h2, err := sched.Submit(context.Background(), Submission{
		ID: "second", PromptTokens: 2, MaxOutput: 2,
		Build: func(context.Context) (Task, error) { return second, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	waitHandle(t, h2, time.Second)
	if h2.FinishReason() != "timeout" {
		t.Fatalf("queued finish reason = %q, want timeout", h2.FinishReason())
	}
	if second.closedCount() != 0 {
		t.Fatalf("queued task must never be built")
	}
	waitHandle(t, h1, time.Second)
}

func TestShutdownDrainsActive(t *testing.T) {
	sched, err := New(Config{MaxActive: 2, MaxQueue: 2, RequestTimeout: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)
	task := newFakeTask("drain", 2, 2)
	task.stepDelay = 5 * time.Millisecond
	h, err := sched.Submit(context.Background(), Submission{
		ID: "drain", PromptTokens: 2, MaxOutput: 2,
		Build: func(context.Context) (Task, error) { return task, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForActive(t, sched, 1)
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer shutdownCancel()
	if err := sched.Shutdown(shutdownCtx); err != nil {
		t.Fatal(err)
	}
	if h.FinishReason() != "length" {
		t.Fatalf("drain finish reason = %q", h.FinishReason())
	}
	if _, err := sched.Submit(context.Background(), Submission{PromptTokens: 1, MaxOutput: 1, Build: func(context.Context) (Task, error) { return newFakeTask("x", 1, 1), nil }}); !errors.Is(err, ErrClosed) {
		t.Fatalf("submit after shutdown = %v, want ErrClosed", err)
	}
}

func TestBuildFailureReleasesSlot(t *testing.T) {
	sched := runScheduler(t, Config{MaxActive: 1, MaxQueue: 1, RequestTimeout: time.Minute})
	failing := func(context.Context) (Task, error) { return nil, errors.New("boom") }
	h, err := sched.Submit(context.Background(), Submission{ID: "bad", PromptTokens: 1, MaxOutput: 1, Build: failing})
	if err != nil {
		t.Fatal(err)
	}
	waitHandle(t, h, time.Second)
	if h.FinishReason() != "error" || h.Err() == nil {
		t.Fatalf("finish = %q err %v", h.FinishReason(), h.Err())
	}
	good := newFakeTask("good", 1, 1)
	h2, err := sched.Submit(context.Background(), Submission{ID: "good", PromptTokens: 1, MaxOutput: 1, Build: func(context.Context) (Task, error) { return good, nil }})
	if err != nil {
		t.Fatal(err)
	}
	waitHandle(t, h2, time.Second)
	if h2.FinishReason() != "length" {
		t.Fatalf("second finish reason = %q", h2.FinishReason())
	}
}
