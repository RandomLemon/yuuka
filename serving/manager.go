package serving

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

var (
	// ErrContextLength reports a prompt plus output budget beyond the configured
	// context or KV capacity.
	ErrContextLength = errors.New("serving: context length exceeded")
)

// Manager owns the scheduler, allocator, prefix cache, and backend for one
// loaded model. It is safe for concurrent Generate calls.
type Manager struct {
	backend Backend
	tok     *tokenizer.Tokenizer
	alloc   *memory.Allocator
	cache   *memory.PrefixCache
	sched   *scheduler.Scheduler
	opts    Options
	model   string

	mu        sync.Mutex
	started   bool
	runCancel context.CancelFunc
	done      chan struct{}
	closeOnce sync.Once
}

type managerBudget struct{ m *Manager }

func (b managerBudget) CheckAdmission(prompt, output int) error {
	return b.m.CheckAdmission(prompt, output)
}

// NewManager validates the backend geometry and builds the scheduler.
func NewManager(backend Backend, tok *tokenizer.Tokenizer, opts Options) (*Manager, error) {
	if backend == nil || tok == nil {
		return nil, fmt.Errorf("serving: backend and tokenizer are required")
	}
	if tok.VocabSize() != backend.VocabSize() {
		return nil, fmt.Errorf("serving: tokenizer vocabulary %d does not match backend %d",
			tok.VocabSize(), backend.VocabSize())
	}
	bs := backend.BlockSize()
	if bs <= 0 {
		return nil, fmt.Errorf("serving: backend block size must be positive")
	}
	capacity := backend.Allocator().Blocks() * bs
	if opts.ContextSize <= 0 || opts.ContextSize > capacity {
		opts.ContextSize = capacity
	}
	m := &Manager{
		backend: backend,
		tok:     tok,
		alloc:   backend.Allocator(),
		opts:    opts,
		model:   backend.ModelKey(),
	}
	if opts.PrefixEnabled {
		m.cache = memory.NewPrefixCache(m.alloc, memory.PrefixConfig{
			Key:        backend.ModelKey(),
			MaxEntries: opts.PrefixEntries,
			MaxBlocks:  opts.PrefixCapacityBlocks,
		})
	}
	cfg := opts.Scheduler
	cfg.Budget = managerBudget{m: m}
	sched, err := scheduler.New(cfg)
	if err != nil {
		return nil, err
	}
	m.sched = sched
	return m, nil
}

// Start launches the scheduler loop. It must be called before Generate.
func (m *Manager) Start(parent context.Context) error {
	if parent == nil {
		parent = context.Background()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.started {
		return fmt.Errorf("serving: manager already started")
	}
	ctx, cancel := context.WithCancel(parent)
	m.runCancel = cancel
	m.done = make(chan struct{})
	m.started = true
	go func() {
		defer close(m.done)
		m.sched.Run(ctx)
	}()
	return nil
}

// Close stops admission, cancels or drains remaining work, and releases the
// backend. It is idempotent.
func (m *Manager) Close() error {
	var closeErr error
	m.closeOnce.Do(func() {
		m.mu.Lock()
		started := m.started
		cancel := m.runCancel
		done := m.done
		m.mu.Unlock()
		if started {
			if cancel != nil {
				cancel()
			}
			_ = m.sched.Close()
			if done != nil {
				<-done
			}
		}
		if m.cache != nil {
			m.cache.Purge()
		}
		if m.backend != nil {
			closeErr = m.backend.Close()
		}
	})
	return closeErr
}

// Shutdown stops admission and waits for active requests to finish, cancelling
// them at the deadline.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	started := m.started
	m.mu.Unlock()
	if !started {
		return nil
	}
	return m.sched.Shutdown(ctx)
}

// CheckAdmission implements the scheduler budget. It is conservative: a
// possible prefix-cache hit does not reduce the admission estimate.
func (m *Manager) CheckAdmission(promptTokens, outputTokens int) error {
	if promptTokens <= 0 {
		return fmt.Errorf("%w: empty prompt", scheduler.ErrCapacity)
	}
	if outputTokens <= 0 {
		return fmt.Errorf("%w: output budget must be positive", scheduler.ErrCapacity)
	}
	positions := promptTokens + outputTokens
	if positions > m.opts.ContextSize {
		return fmt.Errorf("%w: %d positions exceed context %d",
			ErrContextLength, positions, m.opts.ContextSize)
	}
	blocks := (positions + m.backend.BlockSize() - 1) / m.backend.BlockSize()
	if blocks > m.alloc.Blocks() {
		return fmt.Errorf("%w: %d blocks exceed pool %d",
			scheduler.ErrCapacity, blocks, m.alloc.Blocks())
	}
	return nil
}

// Generate runs one request to completion. The sink is invoked with text
// deltas from the scheduler goroutine. A non-nil error may accompany a partial
// Result when generation failed or was cancelled.
func (m *Manager) Generate(ctx context.Context, req Request) (*Result, error) {
	arrival := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	if len(req.Prompt) == 0 {
		return nil, fmt.Errorf("serving: prompt must not be empty")
	}
	if req.MaxTokens <= 0 {
		return nil, fmt.Errorf("serving: max tokens must be positive")
	}
	if len(req.Prompt) >= m.opts.ContextSize {
		return nil, fmt.Errorf("%w: prompt %d exceeds context %d",
			ErrContextLength, len(req.Prompt), m.opts.ContextSize)
	}
	maxTokens := min(req.MaxTokens, m.opts.ContextSize-len(req.Prompt))
	sampler := req.Sampler
	if sampler == nil {
		var err error
		sampler, err = engine.NewSampler(engine.SamplerConfig{}, m.backend.VocabSize())
		if err != nil {
			return nil, err
		}
	}
	req.MaxTokens = maxTokens
	req.Sampler = sampler

	var built *managedTask
	sub := scheduler.Submission{
		ID:           req.ID,
		PromptTokens: len(req.Prompt),
		MaxOutput:    maxTokens,
		Build: func(context.Context) (scheduler.Task, error) {
			task, err := m.build(req, maxTokens, arrival)
			if err != nil {
				return nil, err
			}
			built = task
			return task, nil
		},
	}
	handle, err := m.sched.Submit(ctx, sub)
	if err != nil {
		return nil, err
	}
	if req.Submitted != nil {
		req.Submitted()
	}
	select {
	case <-handle.Done():
	case <-ctx.Done():
		handle.Cancel()
		<-handle.Done()
	}
	if built == nil {
		if err := handle.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("serving: request finished before task construction")
	}
	result := built.result()
	result.QueueWait = handle.QueueWait()
	result.FinishReason = handle.FinishReason()
	if err := handle.Err(); err != nil {
		return result, err
	}
	return result, nil
}

// build allocates the request block table, attaches a prefix hit, and creates
// the backend sequence.
func (m *Manager) build(req Request, maxTokens int, started time.Time) (*managedTask, error) {
	table := memory.NewTable(m.alloc)
	cached := 0
	var state []byte
	var logits []float32
	if m.cache != nil {
		if match := m.cache.Lookup(req.Prompt); match != nil {
			var ok bool
			cached, state, logits, ok = m.applyPrefix(table, match)
			if !ok {
				cached = 0
				state = nil
				logits = nil
			}
		}
	}
	needed := len(req.Prompt) + maxTokens
	if err := table.Cover(needed); err != nil {
		table.Release()
		return nil, fmt.Errorf("%w: allocate %d positions: %v",
			scheduler.ErrCapacity, needed, err)
	}
	capacity := table.Bytes()
	if capacity > m.opts.ContextSize {
		capacity = m.opts.ContextSize
	}
	seq, err := m.backend.NewSession(table, capacity, cached, state)
	if err != nil {
		table.Release()
		return nil, err
	}
	task := newGenerationTask(req.ID, req.Prompt, maxTokens, seq, req.Sampler, m.tok,
		req.StopTokens, req.StopStrings, req.IgnoreEOS, logits, state, cached, req.Sink, started)
	return &managedTask{generationTask: task, manager: m, table: table}, nil
}

// applyPrefix transfers a cache match into the table. Full blocks are shared;
// a partial trailing block is copied so the request can write past the cached
// prompt without corrupting the cache. On failure every match reference is
// released and the table is unchanged.
func (m *Manager) applyPrefix(table *memory.Table, match *memory.Match) (int, []byte, []float32, bool) {
	blocks := match.TakeBlocks()
	logits, valid := deserializeLogits(match.Logits, m.backend.VocabSize())
	bs := m.backend.BlockSize()
	full := len(match.Tokens) / bs
	partial := len(match.Tokens)%bs != 0
	expected := full
	if partial {
		expected++
	}
	fail := func() (int, []byte, []float32, bool) {
		for _, id := range blocks {
			_ = m.alloc.Release(id)
		}
		return 0, nil, nil, false
	}
	if !valid || len(match.Tokens) == 0 || len(blocks) != expected {
		return fail()
	}
	var copyID int = -1
	if partial {
		id, err := m.alloc.Acquire()
		if err != nil {
			return fail()
		}
		if err := m.backend.CopyBlock(id, blocks[full]); err != nil {
			_ = m.alloc.Release(id)
			return fail()
		}
		copyID = id
	}
	if full > 0 {
		if err := table.Share(blocks[:full]); err != nil {
			if copyID >= 0 {
				_ = m.alloc.Release(copyID)
			}
			// Share validated the whole slice before appending, so no block was
			// transferred.
			for _, id := range blocks {
				_ = m.alloc.Release(id)
			}
			return 0, nil, nil, false
		}
	}
	if partial {
		if err := table.Share([]int{copyID}); err != nil {
			_ = m.alloc.Release(copyID)
			for _, id := range blocks[full:] {
				_ = m.alloc.Release(id)
			}
			return 0, nil, nil, false
		}
		_ = m.alloc.Release(blocks[full])
	}
	return len(match.Tokens), match.State, logits, true
}

// Metrics returns a consistent scheduler, KV, and prefix-cache snapshot.
func (m *Manager) Metrics() Metrics {
	out := Metrics{
		Scheduler:  m.sched.Metrics(),
		KV:         m.alloc.Stats(),
		Model:      m.model,
		MaxContext: m.opts.ContextSize,
	}
	if m.cache != nil {
		out.Prefix = m.cache.Stats()
	}
	return out
}

// Limits returns the effective scheduler, KV, and prefix-cache configuration.
func (m *Manager) Limits() Limits {
	sched := m.sched.Limits()
	out := Limits{
		ContextSize:           m.opts.ContextSize,
		BlockSize:             m.backend.BlockSize(),
		KVBlocks:              m.alloc.Blocks(),
		MaxActive:             sched.MaxActive,
		MaxQueue:              sched.MaxQueue,
		MaxBatch:              sched.MaxBatch,
		PrefillChunk:          sched.PrefillChunk,
		PrefillPerCycle:       sched.PrefillPerCycle,
		DecodePerCycle:        sched.DecodePerCycle,
		RequestTimeoutSeconds: sched.RequestTimeout.Seconds(),
		QueueTimeoutSeconds:   sched.QueueTimeout.Seconds(),
		PrefixEnabled:         m.cache != nil,
	}
	if m.cache != nil {
		stats := m.cache.Stats()
		out.PrefixEntries = stats.MaxEntries
		out.PrefixBlocks = stats.MaxBlocks
	}
	return out
}

// ContextSize returns the maximum admitted prompt plus output positions.
func (m *Manager) ContextSize() int { return m.opts.ContextSize }

// VocabSize returns the model vocabulary size.
func (m *Manager) VocabSize() int { return m.backend.VocabSize() }

// Tokenizer returns the shared tokenizer.
func (m *Manager) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// BlockSize returns the KV block size.
func (m *Manager) BlockSize() int { return m.backend.BlockSize() }

// managedTask finalizes prefix caching and releases the request block table
// after the scheduler closes the generation task.
type managedTask struct {
	*generationTask
	manager *Manager
	table   *memory.Table
}

func (t *managedTask) Close() error {
	err := t.generationTask.Close()
	t.manager.finalize(t)
	t.table.Release()
	return err
}

// finalize splits the completed prompt prefix out of the block table and
// inserts it into the cache. Cache failure never changes the request result.
func (m *Manager) finalize(t *managedTask) {
	if m.cache == nil {
		return
	}
	tokens, state, logits, ok := t.PrefixReady()
	if !ok {
		return
	}
	blocks, err := t.table.SplitPrefix(len(tokens), m.backend.CopyBlock)
	if err != nil {
		return
	}
	if err := m.cache.Insert(tokens, blocks, state, logits); err != nil {
		for _, id := range blocks {
			_ = m.alloc.Release(id)
		}
	}
}

func (t *managedTask) result() *Result {
	outcome := t.Outcome()
	return &Result{
		Tokens:         append([]int32(nil), t.generated...),
		Text:           t.text.String(),
		PromptTokens:   outcome.PromptTokens,
		OutputTokens:   outcome.GeneratedTokens,
		FinishReason:   outcome.FinishReason,
		TimeToFirst:    outcome.TimeToFirst,
		PrefillTime:    t.prefill,
		DecodeTime:     t.decode,
		SamplingTime:   t.sampling,
		PrefixCacheHit: t.initialHit,
		PrefixTokens:   t.prefixTokens,
	}
}
