//go:build hip && linux

package serving

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/scheduler"
)

func (h *tinyHarness) newGPUManager(t *testing.T, blocks, blockSize, contextSize int, cache, graph bool, cfg scheduler.Config) *Manager {
	t.Helper()
	manager := h.newStoppedGPUManager(t, blocks, blockSize, contextSize, cache, graph, cfg)
	if err := manager.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	return manager
}

func (h *tinyHarness) newStoppedGPUManager(t *testing.T, blocks, blockSize, contextSize int, cache, graph bool, cfg scheduler.Config) *Manager {
	t.Helper()
	backend, err := NewQwen35GPUBackend(h.reader, h.model, "tiny-qwen35-gpu-test", blocks, blockSize, 32, graph)
	if err != nil {
		t.Skipf("HIP backend unavailable: %v", err)
	}
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

func TestGPUConcurrentRequestsMatchIsolated(t *testing.T) {
	h := newTinyHarness(t)
	isolated := h.newGPUManager(t, 32, 8, 96, false, true, scheduler.Config{
		MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute,
	})
	concurrent := h.newStoppedGPUManager(t, 32, 8, 96, false, true, scheduler.Config{
		MaxActive: 3, MaxQueue: 4, MaxBatch: 3, PrefillChunk: 4, PrefillPerCycle: 1, DecodePerCycle: 3,
		RequestTimeout: time.Minute,
	})
	cases := []string{"Hello world", "The capital of France is", "1+1="}
	want := make(map[string]*Result)
	for _, prompt := range cases {
		result, err := isolated.Generate(context.Background(), Request{
			ID: prompt, Prompt: h.prompt(t, prompt), MaxTokens: 4,
			Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
		})
		if err != nil {
			t.Fatalf("isolated %q: %v", prompt, err)
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
			result, err := concurrent.Generate(context.Background(), Request{
				ID: prompt, Prompt: h.prompt(t, prompt), MaxTokens: 4,
				Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
			})
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failures = append(failures, err)
				return
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

func TestGPUPrefixCacheMatchesFreshRun(t *testing.T) {
	h := newTinyHarness(t)
	cached := h.newGPUManager(t, 48, 8, 128, true, true, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	fresh := h.newGPUManager(t, 48, 8, 128, false, false, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})

	prefix := h.prompt(t, "Hello world")
	sampler := samplerFor(t, h.tok.VocabSize())
	first, err := cached.Generate(context.Background(), Request{
		ID: "first", Prompt: prefix, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	second, err := cached.Generate(context.Background(), Request{
		ID: "second", Prompt: prefix, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !second.PrefixCacheHit || second.PrefixTokens != len(prefix) {
		t.Fatalf("prefix hit=%v tokens=%d, want %d",
			second.PrefixCacheHit, second.PrefixTokens, len(prefix))
	}
	if !reflect.DeepEqual(second.Tokens, first.Tokens) {
		t.Fatalf("cached tokens %v, first tokens %v", second.Tokens, first.Tokens)
	}

	longer := append(append([]int32(nil), prefix...), h.prompt(t, " again")...)
	longCached, err := cached.Generate(context.Background(), Request{
		ID: "long", Prompt: longer, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	longFresh, err := fresh.Generate(context.Background(), Request{
		ID: "fresh", Prompt: longer, MaxTokens: 3, Sampler: sampler, IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(longCached.Tokens, longFresh.Tokens) {
		t.Fatalf("prefix tokens %v, fresh tokens %v", longCached.Tokens, longFresh.Tokens)
	}
	if stats := cached.Metrics().Prefix; stats.Hits < 2 {
		t.Fatalf("prefix hits = %d", stats.Hits)
	}
}

func TestGPUCancellationReleasesBlocks(t *testing.T) {
	h := newTinyHarness(t)
	manager := h.newGPUManager(t, 32, 8, 96, false, true, scheduler.Config{MaxActive: 1, MaxQueue: 4, RequestTimeout: time.Minute})
	ctx, cancel := context.WithCancel(context.Background())
	var once sync.Once
	_, err := manager.Generate(ctx, Request{
		ID: "cancel", Prompt: h.prompt(t, "Hello world"), MaxTokens: 16,
		Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
		Sink: func(int32, string) error {
			once.Do(cancel)
			return nil
		},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
	deadline := time.Now().Add(time.Second)
	for manager.Metrics().KV.Used != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("cancelled request leaked blocks: %+v", manager.Metrics().KV)
		}
		time.Sleep(time.Millisecond)
	}
	stats := manager.Metrics().KV
	if stats.Allocated != stats.Released {
		t.Fatalf("cancelled request left allocation imbalance: %+v", stats)
	}
	if active := manager.Metrics().Scheduler.Active; active != 0 {
		t.Fatalf("cancelled request still active: %d", active)
	}
}

func TestGPURepeatedRequestsDoNotGrowMemory(t *testing.T) {
	h := newTinyHarness(t)
	manager := h.newGPUManager(t, 32, 8, 96, false, true, scheduler.Config{
		MaxActive: 2, MaxQueue: 4, RequestTimeout: time.Minute,
	})
	ids := h.prompt(t, "Once upon a time")
	for i := range 12 {
		_, err := manager.Generate(context.Background(), Request{
			ID: "cycle", Prompt: ids, MaxTokens: 4,
			Sampler: samplerFor(t, h.tok.VocabSize()), IgnoreEOS: true,
		})
		if err != nil {
			t.Fatalf("cycle %d: %v", i, err)
		}
		stats := manager.Metrics().KV
		if stats.Used != 0 {
			t.Fatalf("cycle %d leaked %d blocks: %+v", i, stats.Used, stats)
		}
		if stats.Allocated != stats.Released {
			t.Fatalf("cycle %d allocation imbalance: %+v", i, stats)
		}
	}
}
