package serving

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

type tinyHarness struct {
	reader *gguf.Reader
	tok    *tokenizer.Tokenizer
	model  *qwen35.Model
}

func newTinyHarness(t *testing.T) *tinyHarness {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	return &tinyHarness{reader: reader, tok: tok, model: model}
}

func (h *tinyHarness) newManager(t *testing.T, blocks, blockSize, contextSize int, cache bool, cfg scheduler.Config) *Manager {
	t.Helper()
	backend, err := NewQwen35Backend(h.model, "tiny-qwen35-test", blocks, blockSize)
	if err != nil {
		t.Fatal(err)
	}
	return h.newManagerWithBackend(t, backend, blocks, contextSize, cache, cfg)
}

func (h *tinyHarness) newManagerWithBackend(t *testing.T, backend Backend, blocks, contextSize int, cache bool, cfg scheduler.Config) *Manager {
	t.Helper()
	manager := h.newStoppedManager(t, backend, blocks, contextSize, cache, cfg)
	if err := manager.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	return manager
}

// newStoppedManager builds a manager without starting the scheduler, so tests
// can enqueue requests and then observe their admission together.
func (h *tinyHarness) newStoppedManager(t *testing.T, backend Backend, blocks, contextSize int, cache bool, cfg scheduler.Config) *Manager {
	t.Helper()
	opts := Options{
		ContextSize:   contextSize,
		PrefixEnabled: cache,
		PrefixEntries: 16,
		Scheduler:     cfg,
	}
	if cache {
		opts.PrefixCapacityBlocks = blocks
	}
	manager, err := NewManager(backend, h.tok, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = manager.Close() })
	return manager
}

func (h *tinyHarness) prompt(t *testing.T, text string) []int32 {
	t.Helper()
	ids, err := h.tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	return ids
}

func samplerFor(t *testing.T, vocab int) *engine.Sampler {
	t.Helper()
	sampler, err := engine.NewSampler(engine.SamplerConfig{}, vocab)
	if err != nil {
		t.Fatal(err)
	}
	return sampler
}

func TestPrefixCapacityRequiresEnable(t *testing.T) {
	h := newTinyHarness(t)
	backend, err := NewQwen35Backend(h.model, "tiny-qwen35-test", 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewManager(backend, h.tok, Options{
		ContextSize:          64,
		PrefixEnabled:        false,
		PrefixEntries:        4,
		PrefixCapacityBlocks: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	if manager.Limits().PrefixEnabled {
		t.Fatal("prefix capacity enabled the cache without PrefixEnabled")
	}
	if got := manager.Metrics().Prefix.MaxEntries; got != 0 {
		t.Fatalf("prefix cache max entries = %d, want 0", got)
	}
}

func TestConcurrentRequestsMatchIsolatedRuns(t *testing.T) {
	h := newTinyHarness(t)
	isolated := h.newManager(t, 32, 8, 96, false, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	concurrentBackend, err := NewQwen35Backend(h.model, "tiny-qwen35-concurrent-test", 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	concurrent := h.newStoppedManager(t, concurrentBackend, 32, 96, false, scheduler.Config{
		MaxActive: 3, MaxQueue: 4, MaxBatch: 3, PrefillChunk: 4, PrefillPerCycle: 1, DecodePerCycle: 3,
		RequestTimeout: time.Minute,
	})
	cases := []string{"Hello world", "The capital of France is", "1+1="}
	want := make(map[string]*Result)
	for _, prompt := range cases {
		ids := h.prompt(t, prompt)
		sampler := samplerFor(t, h.tok.VocabSize())
		result, err := isolated.Generate(context.Background(), Request{
			ID: prompt, Prompt: ids, MaxTokens: 4, Sampler: sampler, IgnoreEOS: true,
		})
		if err != nil {
			t.Fatalf("isolated %q: %v", prompt, err)
		}
		if len(result.Tokens) != 4 || result.FinishReason != "length" {
			t.Fatalf("isolated %q produced %d tokens (%s)", prompt, len(result.Tokens), result.FinishReason)
		}
		want[prompt] = result
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	got := make(map[string]*Result)
	var failures []error
	for _, prompt := range cases {
		wg.Add(1)
		go func(prompt string) {
			defer wg.Done()
			var deltas strings.Builder
			ids := h.prompt(t, prompt)
			sampler := samplerFor(t, h.tok.VocabSize())
			result, err := concurrent.Generate(context.Background(), Request{
				ID: prompt, Prompt: ids, MaxTokens: 4, Sampler: sampler, IgnoreEOS: true,
				Sink: func(_ int32, delta string) error {
					deltas.WriteString(delta)
					return nil
				},
			})
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failures = append(failures, err)
				return
			}
			if deltas.String() != result.Text {
				failures = append(failures, errors.New("sink text does not match result text for "+prompt))
			}
			got[prompt] = result
		}(prompt)
	}
	deadline := time.Now().Add(2 * time.Second)
	for concurrent.Metrics().Scheduler.Submissions < uint64(len(cases)) {
		if time.Now().After(deadline) {
			t.Fatalf("requests were not submitted: %+v", concurrent.Metrics().Scheduler)
		}
		time.Sleep(time.Millisecond)
	}
	if err := concurrent.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	for _, err := range failures {
		t.Fatal(err)
	}
	for _, prompt := range cases {
		if !reflect.DeepEqual(got[prompt].Tokens, want[prompt].Tokens) {
			t.Fatalf("%q concurrent tokens %v, isolated %v",
				prompt, got[prompt].Tokens, want[prompt].Tokens)
		}
		if got[prompt].Text != want[prompt].Text {
			t.Fatalf("%q concurrent text %q, isolated %q",
				prompt, got[prompt].Text, want[prompt].Text)
		}
	}
	metrics := concurrent.Metrics()
	if metrics.Scheduler.MaxActiveObserved < 3 {
		t.Fatalf("requests were not concurrent: %+v", metrics)
	}
	if metrics.KV.Used != 0 {
		t.Fatalf("KV blocks leaked: %+v", metrics.KV)
	}
}

func TestTimeToFirstIncludesQueueWait(t *testing.T) {
	h := newTinyHarness(t)
	backend, err := NewQwen35Backend(h.model, "tiny-ttft-test", 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	manager := h.newStoppedManager(t, backend, 32, 96, false,
		scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	prompt := h.prompt(t, "Hello world")
	sampler := samplerFor(t, h.tok.VocabSize())
	done := make(chan struct {
		result *Result
		err    error
	}, 1)
	go func() {
		result, err := manager.Generate(context.Background(), Request{
			ID: "queued", Prompt: prompt, MaxTokens: 2,
			Sampler: sampler, IgnoreEOS: true,
		})
		done <- struct {
			result *Result
			err    error
		}{result, err}
	}()
	deadline := time.Now().Add(2 * time.Second)
	for manager.Metrics().Scheduler.Submissions == 0 {
		if time.Now().After(deadline) {
			_ = manager.Start(context.Background())
			<-done
			t.Fatal("request was not submitted")
		}
		time.Sleep(time.Millisecond)
	}
	// Hold the scheduler stopped so the request spends measurable time queued.
	time.Sleep(40 * time.Millisecond)
	if err := manager.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	out := <-done
	if out.err != nil {
		t.Fatal(out.err)
	}
	if out.result.QueueWait < 20*time.Millisecond {
		t.Fatalf("queue wait = %s, want at least 20ms", out.result.QueueWait)
	}
	if out.result.TimeToFirst < out.result.QueueWait {
		t.Fatalf("TTFT %s excludes queue wait %s", out.result.TimeToFirst, out.result.QueueWait)
	}
}

func TestPrefixCacheReuseMatchesFreshRun(t *testing.T) {
	h := newTinyHarness(t)
	cached := h.newManager(t, 48, 8, 128, true, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	fresh := h.newManager(t, 48, 8, 128, false, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})

	prefixText := "Hello world"
	prefix := h.prompt(t, prefixText)
	sampler := samplerFor(t, h.tok.VocabSize())
	first, err := cached.Generate(context.Background(), Request{
		ID: "first", Prompt: prefix, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.PrefixCacheHit {
		t.Fatal("first request must not hit the prefix cache")
	}
	if len(first.Tokens) != 3 || first.FinishReason != "length" {
		t.Fatalf("first request produced %d tokens (%s)", len(first.Tokens), first.FinishReason)
	}
	if first.PrefillTime <= 0 || first.DecodeTime <= 0 || first.SamplingTime <= 0 {
		t.Fatalf("timings were not separated: %+v", first)
	}
	if stats := cached.Metrics().Prefix; stats.Entries != 1 || stats.Inserts != 1 {
		t.Fatalf("prefix stats after first request: %+v", stats)
	}

	second, err := cached.Generate(context.Background(), Request{
		ID: "second", Prompt: prefix, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !second.PrefixCacheHit || second.PrefixTokens != len(prefix) {
		t.Fatalf("second request hit=%v tokens=%d, want full prompt %d",
			second.PrefixCacheHit, second.PrefixTokens, len(prefix))
	}
	if !reflect.DeepEqual(second.Tokens, first.Tokens) {
		t.Fatalf("cached tokens %v, fresh tokens %v", second.Tokens, first.Tokens)
	}

	// A longer prompt must reuse the cached shorter prefix and produce the same
	// output as a run that computes the whole prompt.
	longer := append(append([]int32(nil), prefix...), h.prompt(t, " again")...)
	longCached, err := cached.Generate(context.Background(), Request{
		ID: "long-cached", Prompt: longer, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	longFresh, err := fresh.Generate(context.Background(), Request{
		ID: "long-fresh", Prompt: longer, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !longCached.PrefixCacheHit || longCached.PrefixTokens != len(prefix) {
		t.Fatalf("longer request hit=%v tokens=%d, want prefix %d",
			longCached.PrefixCacheHit, longCached.PrefixTokens, len(prefix))
	}
	if !reflect.DeepEqual(longCached.Tokens, longFresh.Tokens) {
		t.Fatalf("prefix reuse tokens %v, fresh tokens %v", longCached.Tokens, longFresh.Tokens)
	}
	if longCached.Text != longFresh.Text {
		t.Fatalf("prefix reuse text %q, fresh text %q", longCached.Text, longFresh.Text)
	}
	// The longer prompt must have been cached after it completed, so a request
	// extending it hits the full longer prefix.
	evenLonger := append(append([]int32(nil), longer...), h.prompt(t, "!")...)
	grown, err := cached.Generate(context.Background(), Request{
		ID: "grown", Prompt: evenLonger, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !grown.PrefixCacheHit || grown.PrefixTokens != len(longer) {
		t.Fatalf("grown hit=%v tokens=%d, want %d",
			grown.PrefixCacheHit, grown.PrefixTokens, len(longer))
	}
	if stats := cached.Metrics().Prefix; stats.Hits < 2 {
		t.Fatalf("prefix hits = %d, want at least 2", stats.Hits)
	}
}

func TestCancellationReleasesBlocks(t *testing.T) {
	h := newTinyHarness(t)
	manager := h.newManager(t, 32, 8, 96, false, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	ctx, cancel := context.WithCancel(context.Background())
	ids := h.prompt(t, "Hello world")
	var once sync.Once
	result, err := manager.Generate(ctx, Request{
		ID: "cancel", Prompt: ids, MaxTokens: 16, Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
		Sink: func(int32, string) error {
			once.Do(cancel)
			return nil
		},
	})
	if err == nil {
		t.Fatalf("expected cancellation error, got result %+v", result)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		stats := manager.Metrics().KV
		if stats.Used == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("cancelled request leaked blocks: %+v", stats)
		}
		time.Sleep(time.Millisecond)
	}
	if stats := manager.Metrics().KV; stats.Allocated != stats.Released {
		t.Fatalf("allocation/release imbalance: %+v", stats)
	}
}

func TestRepeatedRequestsDoNotGrowMemory(t *testing.T) {
	h := newTinyHarness(t)
	manager := h.newManager(t, 32, 8, 96, false, scheduler.Config{MaxActive: 2, MaxQueue: 4, RequestTimeout: time.Minute})
	ids := h.prompt(t, "Once upon a time")
	for i := range 12 {
		_, err := manager.Generate(context.Background(), Request{
			ID: "cycle", Prompt: ids, MaxTokens: 4, Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
		})
		if err != nil {
			t.Fatalf("cycle %d: %v", i, err)
		}
		stats := manager.Metrics().KV
		if stats.Used != 0 {
			t.Fatalf("cycle %d leaked %d blocks: %+v", i, stats.Used, stats)
		}
	}
	stats := manager.Metrics().KV
	if stats.Allocated != stats.Released {
		t.Fatalf("allocation/release imbalance after cycles: %+v", stats)
	}
	if manager.Metrics().Scheduler.Active != 0 {
		t.Fatalf("active requests remain: %+v", manager.Metrics().Scheduler)
	}
}

type failingBackend struct {
	Backend
}

func (b failingBackend) NewSession(table *memory.Table, capacity, position int, state []byte) (Sequence, error) {
	seq, err := b.Backend.NewSession(table, capacity, position, state)
	if err != nil {
		return nil, err
	}
	return &failingSequence{Sequence: seq}, nil
}

type failingSequence struct {
	Sequence
}

func (s *failingSequence) Decode(int32) ([]float32, error) {
	return nil, errors.New("injected decode failure")
}

func TestDecodeFailureReleasesBlocks(t *testing.T) {
	h := newTinyHarness(t)
	backend, err := NewQwen35Backend(h.model, "tiny-qwen35-failure-test", 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	manager := h.newManagerWithBackend(t, failingBackend{Backend: backend}, 32, 96, false,
		scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	_, err = manager.Generate(context.Background(), Request{
		ID: "failure", Prompt: h.prompt(t, "Hello world"), MaxTokens: 4,
		Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
	})
	if err == nil || !strings.Contains(err.Error(), "injected decode failure") {
		t.Fatalf("error = %v", err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		stats := manager.Metrics().KV
		if stats.Used == 0 {
			if stats.Allocated != stats.Released {
				t.Fatalf("allocation/release imbalance: %+v", stats)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("failed request leaked blocks: %+v", stats)
		}
		time.Sleep(time.Millisecond)
	}
}
