package memory

import (
	"bytes"
	"errors"
	"testing"
)

func acquireBlocks(t *testing.T, a *Allocator, n int) []int {
	t.Helper()
	blocks := make([]int, n)
	for i := range n {
		id, err := a.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		blocks[i] = id
	}
	return blocks
}

func tokens(n int) []int32 {
	out := make([]int32, n)
	for i := range n {
		out[i] = int32(100 + i)
	}
	return out
}

func TestPrefixLookupAndRelease(t *testing.T) {
	a, err := NewAllocator(12, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model-a", MaxEntries: 4})
	prefix := tokens(9)
	state := []byte{1, 2, 3, 4}
	blocks := acquireBlocks(t, a, 3)
	if err := cache.Insert(prefix, blocks, state, nil); err != nil {
		t.Fatal(err)
	}
	if stats := cache.Stats(); stats.Entries != 1 || stats.Blocks != 3 || stats.Inserts != 1 {
		t.Fatalf("stats after insert: %+v", stats)
	}
	if got := a.Stats().Used; got != 3 {
		t.Fatalf("allocator used = %d, want 3", got)
	}

	prompt := append(tokens(9), 999, 1000)
	match := cache.Lookup(prompt)
	if match == nil {
		t.Fatal("expected prefix hit")
	}
	if !equalTokens(match.Tokens, prefix) {
		t.Fatalf("match tokens = %v, want %v", match.Tokens, prefix)
	}
	if !bytes.Equal(match.State, state) {
		t.Fatalf("snapshot mismatch")
	}
	if got := a.Refs(match.Blocks[0]); got != 2 {
		t.Fatalf("retained block refs = %d, want 2", got)
	}
	taken := match.TakeBlocks()
	table := NewTable(a)
	if err := table.Share(taken); err != nil {
		t.Fatal(err)
	}
	if table.Len() != 3 {
		t.Fatalf("shared table blocks = %d, want 3", table.Len())
	}
	table.Release()
	if stats := cache.Stats(); stats.Hits != 1 || stats.Misses != 0 {
		t.Fatalf("hit stats: %+v", stats)
	}
	if match := cache.Lookup(tokens(3)); match != nil {
		match.Release(a)
		t.Fatal("short prompt must not match a longer cached prefix")
	}
	if stats := cache.Stats(); stats.Misses != 1 {
		t.Fatalf("miss stats: %+v", stats)
	}
	cache.Purge()
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("purge leaked blocks: %+v", stats)
	}
}

func TestPrefixLongestMatch(t *testing.T) {
	a, err := NewAllocator(24, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model-a", MaxEntries: 8})
	short := tokens(4)
	long := tokens(12)
	if err := cache.Insert(short, acquireBlocks(t, a, 1), []byte("short"), nil); err != nil {
		t.Fatal(err)
	}
	if err := cache.Insert(long, acquireBlocks(t, a, 3), []byte("long"), nil); err != nil {
		t.Fatal(err)
	}
	prompt := append(tokens(20), 7)
	match := cache.Lookup(prompt)
	if match == nil {
		t.Fatal("expected prefix hit")
	}
	defer match.Release(a)
	if len(match.Tokens) != 12 {
		t.Fatalf("matched %d tokens, want 12", len(match.Tokens))
	}
	if string(match.State) != "long" {
		t.Fatalf("state = %q, want long", match.State)
	}
}

func TestPrefixKeySeparatesModels(t *testing.T) {
	a, err := NewAllocator(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	cacheA := NewPrefixCache(a, PrefixConfig{Key: "model-a"})
	cacheB := NewPrefixCache(a, PrefixConfig{Key: "model-b"})
	tokens := tokens(8)
	if cacheA.Key(tokens) == cacheB.Key(tokens) {
		t.Fatal("model identities must not share a key")
	}
	if err := cacheA.Insert(tokens, acquireBlocks(t, a, 2), []byte("a"), nil); err != nil {
		t.Fatal(err)
	}
	if match := cacheB.Lookup(tokens); match != nil {
		match.Release(a)
		t.Fatal("key from one model must not hit another cache")
	}
	cacheA.Purge()
}

func TestPrefixLRUEviction(t *testing.T) {
	a, err := NewAllocator(24, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model", MaxEntries: 2})
	first := tokens(4)
	second := append(tokens(4), 50, 51, 52, 53)
	third := append(tokens(4), 60, 61, 62, 63)
	if err := cache.Insert(first, acquireBlocks(t, a, 1), nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cache.Insert(second, acquireBlocks(t, a, 2), nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cache.Insert(third, acquireBlocks(t, a, 2), nil, nil); err != nil {
		t.Fatal(err)
	}
	stats := cache.Stats()
	if stats.Entries != 2 || stats.Evictions != 1 {
		t.Fatalf("LRU stats: %+v", stats)
	}
	if match := cache.Lookup(first); match != nil {
		match.Release(a)
		t.Fatal("least recently used entry should have been evicted")
	}
	cache.Purge()
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("eviction leaked blocks: %+v", stats)
	}
}

func TestPrefixRejectsWithoutTakingOwnership(t *testing.T) {
	a, err := NewAllocator(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model", MaxBlocks: 1})
	blocks := acquireBlocks(t, a, 2)
	if err := cache.Insert(tokens(8), blocks, nil, nil); !errors.Is(err, ErrCacheFull) {
		t.Fatalf("insert error = %v, want ErrCacheFull", err)
	}
	for _, id := range blocks {
		if refs := a.Refs(id); refs != 1 {
			t.Fatalf("rejected block %d refs = %d, want 1", id, refs)
		}
		if err := a.Release(id); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPrefixEvictionUnblocksAllocator(t *testing.T) {
	a, err := NewAllocator(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model"})
	if err := cache.Insert(tokens(8), acquireBlocks(t, a, 2), nil, nil); err != nil {
		t.Fatal(err)
	}
	held := acquireBlocks(t, a, 2)
	id, err := a.Acquire()
	if err != nil {
		t.Fatalf("acquire after eviction: %v", err)
	}
	if stats := cache.Stats(); stats.Entries != 0 {
		t.Fatalf("entry was not evicted: %+v", stats)
	}
	if err := a.Release(id); err != nil {
		t.Fatal(err)
	}
	for _, id := range held {
		if err := a.Release(id); err != nil {
			t.Fatal(err)
		}
	}
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("blocks leaked after eviction: %+v", stats)
	}
}

func TestTableSplitThenCacheInsert(t *testing.T) {
	a, err := NewAllocator(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewPrefixCache(a, PrefixConfig{Key: "model"})
	table := NewTable(a)
	if err := table.Cover(12); err != nil {
		t.Fatal(err)
	}
	copied := 0
	prefix, err := table.SplitPrefix(6, func(dst, src int) error {
		copied++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if copied != 1 {
		t.Fatalf("partial split copied %d blocks, want 1", copied)
	}
	prompt := tokens(6)
	if err := cache.Insert(prompt, prefix, []byte("state"), nil); err != nil {
		t.Fatal(err)
	}
	match := cache.Lookup(prompt)
	if match == nil {
		t.Fatal("expected inserted prefix to hit")
	}
	if len(match.Blocks) != 2 || len(match.Tokens) != 6 {
		t.Fatalf("match = %d tokens / %d blocks", len(match.Tokens), len(match.Blocks))
	}
	match.Release(a)
	table.Release()
	cache.Purge()
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("leaked blocks: %+v", stats)
	}
}
