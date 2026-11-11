package memory

import (
	"errors"
	"sync"
	"testing"
)

func TestAllocatorReferenceCounting(t *testing.T) {
	a, err := NewAllocator(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	first, err := a.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	second, err := a.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Acquire(); !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("third acquire error = %v, want ErrOutOfBlocks", err)
	}
	if err := a.Retain(first); err != nil {
		t.Fatal(err)
	}
	if refs := a.Refs(first); refs != 2 {
		t.Fatalf("refs = %d, want 2", refs)
	}
	if err := a.Release(first); err != nil {
		t.Fatal(err)
	}
	if refs := a.Refs(first); refs != 1 {
		t.Fatalf("refs after one release = %d, want 1", refs)
	}
	if err := a.Release(second); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(second); err == nil {
		t.Fatal("double release must fail")
	}
	if err := a.Release(first); err != nil {
		t.Fatal(err)
	}
	stats := a.Stats()
	if stats.Free != 2 || stats.Used != 0 || stats.Shared != 0 {
		t.Fatalf("stats after release = %+v", stats)
	}
}

func TestAllocatorConcurrent(t *testing.T) {
	a, err := NewAllocator(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				id, err := a.Acquire()
				if err != nil {
					t.Errorf("acquire: %v", err)
					return
				}
				if err := a.Retain(id); err != nil {
					t.Errorf("retain: %v", err)
					return
				}
				if err := a.Release(id); err != nil {
					t.Errorf("release: %v", err)
					return
				}
				if err := a.Release(id); err != nil {
					t.Errorf("release: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	stats := a.Stats()
	if stats.Used != 0 || stats.Free != 8 {
		t.Fatalf("allocator leaked blocks: %+v", stats)
	}
}

func TestTableCoverMapAndShare(t *testing.T) {
	a, err := NewAllocator(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	table := NewTable(a)
	if err := table.Cover(9); err != nil {
		t.Fatal(err)
	}
	if table.Len() != 3 {
		t.Fatalf("blocks = %d, want 3", table.Len())
	}
	block, offset, err := table.Map(5)
	if err != nil {
		t.Fatal(err)
	}
	if block != table.Blocks()[1] || offset != 1 {
		t.Fatalf("map(5) = (%d,%d), blocks %v", block, offset, table.Blocks())
	}
	if _, _, err := table.Map(12); err == nil {
		t.Fatal("unmapped position must fail")
	}

	// Simulate a cache match: retain one block for the caller, transfer it.
	shared := table.Blocks()[2]
	if err := a.Retain(shared); err != nil {
		t.Fatal(err)
	}
	other := NewTable(a)
	if err := other.Share([]int{shared}); err != nil {
		t.Fatal(err)
	}
	if got := a.Refs(shared); got != 2 {
		t.Fatalf("shared refs = %d, want 2", got)
	}
	table.Release()
	if got := a.Refs(shared); got != 1 {
		t.Fatalf("refs after first table release = %d, want 1", got)
	}
	other.Release()
	if got := a.Refs(shared); got != 0 {
		t.Fatalf("refs after second table release = %d, want 0", got)
	}
	stats := a.Stats()
	if stats.Used != 0 {
		t.Fatalf("leaked blocks: %+v", stats)
	}
}

func TestTableSplitPrefixCopiesPartialBlock(t *testing.T) {
	a, err := NewAllocator(6, 4)
	if err != nil {
		t.Fatal(err)
	}
	table := NewTable(a)
	if err := table.Cover(12); err != nil {
		t.Fatal(err)
	}
	contents := map[int]string{}
	for _, id := range table.Blocks() {
		contents[id] = "block-" + string(rune('a'+id))
	}
	prefix, err := table.SplitPrefix(5, func(dst, src int) error {
		contents[dst] = contents[src]
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prefix) != 2 {
		t.Fatalf("prefix blocks = %v, want two", prefix)
	}
	remaining := table.Blocks()
	if len(remaining) != 2 {
		t.Fatalf("remaining blocks = %v, want two", remaining)
	}
	if contents[remaining[0]] != contents[prefix[1]] {
		t.Fatalf("partial block was not copied: cached %q table %q",
			contents[prefix[1]], contents[remaining[0]])
	}
	if prefix[0] == remaining[0] || prefix[1] == remaining[0] {
		t.Fatalf("table still aliases cached blocks: prefix %v table %v", prefix, remaining)
	}
	for _, id := range prefix {
		if err := a.Release(id); err != nil {
			t.Fatal(err)
		}
	}
	table.Release()
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("leaked blocks after split: %+v", stats)
	}
}

func TestTableSplitPrefixAligned(t *testing.T) {
	a, err := NewAllocator(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	table := NewTable(a)
	if err := table.Cover(16); err != nil {
		t.Fatal(err)
	}
	all := table.Blocks()
	prefix, err := table.SplitPrefix(8, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(prefix) != 2 || table.Len() != 2 {
		t.Fatalf("prefix %v, remaining %v", prefix, table.Blocks())
	}
	if prefix[0] != all[0] || prefix[1] != all[1] {
		t.Fatalf("prefix %v does not match original head %v", prefix, all)
	}
	for _, id := range prefix {
		if err := a.Release(id); err != nil {
			t.Fatal(err)
		}
	}
	table.Release()
	if stats := a.Stats(); stats.Used != 0 {
		t.Fatalf("leaked blocks: %+v", stats)
	}
}

func TestTableGrowFailureRollsBack(t *testing.T) {
	a, err := NewAllocator(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	table := NewTable(a)
	if err := table.Grow(1); err != nil {
		t.Fatal(err)
	}
	if err := table.Grow(3); !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("grow error = %v, want ErrOutOfBlocks", err)
	}
	if table.Len() != 1 {
		t.Fatalf("failed grow changed table: %v", table.Blocks())
	}
	if stats := a.Stats(); stats.Used != 1 {
		t.Fatalf("failed grow leaked blocks: %+v", stats)
	}
	table.Release()
}
