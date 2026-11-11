// Package memory owns the paged KV block allocator, per-request block tables,
// and the reference-counted prefix cache used by the continuous scheduler.
//
// Allocator block IDs are stable for the lifetime of the pool. A block is owned
// by an allocator and referenced by one or more tables or cache entries. Every
// Acquire/Retain must be matched by exactly one Release. Blocks are not
// returned to the free list while any reference remains.
package memory

import (
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrOutOfBlocks reports that no free or evictable block remains.
	ErrOutOfBlocks = errors.New("memory: no KV blocks available")
	// ErrInvalidBlock reports a block ID outside the allocator range.
	ErrInvalidBlock = errors.New("memory: invalid KV block")
	// ErrTableReleased reports use of a table after Release.
	ErrTableReleased = errors.New("memory: block table is released")
)

// Stats is a point-in-time allocator view.
type Stats struct {
	BlockSize int    `json:"block_size"`
	Total     int    `json:"total"`
	Used      int    `json:"used"`
	Free      int    `json:"free"`
	Shared    int    `json:"shared"`
	Allocated uint64 `json:"allocated"`
	Released  uint64 `json:"released"`
}

// Allocator hands out fixed-size KV blocks by reference count. It is safe for
// concurrent use. The evictor, when installed, is called without the allocator
// lock when no free block remains.
type Allocator struct {
	blockSize int
	free      []int
	refs      []int32

	mu        sync.Mutex
	evictor   func(need int) int
	evictMu   sync.Mutex
	allocated uint64
	released  uint64
}

// NewAllocator creates an allocator with blocks fixed-size blocks. blocks must
// be positive and blockSize must not exceed the smallest supported context.
func NewAllocator(blocks, blockSize int) (*Allocator, error) {
	if blocks <= 0 {
		return nil, fmt.Errorf("memory: block count %d must be positive", blocks)
	}
	if blockSize <= 0 {
		return nil, fmt.Errorf("memory: block size %d must be positive", blockSize)
	}
	a := &Allocator{
		blockSize: blockSize,
		refs:      make([]int32, blocks),
		free:      make([]int, blocks),
	}
	for i := range blocks {
		a.free[i] = blocks - 1 - i
	}
	return a, nil
}

// BlockSize returns the number of token positions in one block.
func (a *Allocator) BlockSize() int { return a.blockSize }

// Blocks returns the total block count.
func (a *Allocator) Blocks() int { return len(a.refs) }

// SetEvictor installs a callback that releases cached blocks when the pool is
// exhausted. It returns the number of releases performed.
func (a *Allocator) SetEvictor(evict func(need int) int) {
	a.mu.Lock()
	a.evictor = evict
	a.mu.Unlock()
}

// Acquire returns a block with reference count one. When the free list is
// empty, the evictor is called outside the allocator lock.
func (a *Allocator) Acquire() (int, error) {
	for {
		a.mu.Lock()
		if n := len(a.free); n > 0 {
			id := a.free[n-1]
			a.free = a.free[:n-1]
			a.refs[id] = 1
			a.allocated++
			a.mu.Unlock()
			return id, nil
		}
		evict := a.evictor
		a.mu.Unlock()
		if evict == nil {
			return 0, ErrOutOfBlocks
		}
		a.evictMu.Lock()
		freed := evict(1)
		a.evictMu.Unlock()
		if freed == 0 {
			return 0, ErrOutOfBlocks
		}
	}
}

// Retain adds one reference to an existing block.
func (a *Allocator) Retain(id int) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if id < 0 || id >= len(a.refs) {
		return ErrInvalidBlock
	}
	if a.refs[id] <= 0 {
		return fmt.Errorf("memory: Retain of free block %d", id)
	}
	a.refs[id]++
	return nil
}

// Release drops one reference. The block returns to the free list when the
// last reference is released. Releasing a free block is an error.
func (a *Allocator) Release(id int) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if id < 0 || id >= len(a.refs) {
		return ErrInvalidBlock
	}
	if a.refs[id] <= 0 {
		return fmt.Errorf("memory: Release of free block %d", id)
	}
	a.refs[id]--
	a.released++
	if a.refs[id] == 0 {
		a.free = append(a.free, id)
	}
	return nil
}

// Refs returns the current reference count, or zero for an invalid block.
func (a *Allocator) Refs(id int) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	if id < 0 || id >= len(a.refs) {
		return 0
	}
	return int(a.refs[id])
}

// Stats returns the current allocator view.
func (a *Allocator) Stats() Stats {
	a.mu.Lock()
	defer a.mu.Unlock()
	shared := 0
	for _, ref := range a.refs {
		if ref > 1 {
			shared++
		}
	}
	return Stats{
		BlockSize: a.blockSize,
		Total:     len(a.refs),
		Used:      len(a.refs) - len(a.free),
		Free:      len(a.free),
		Shared:    shared,
		Allocated: a.allocated,
		Released:  a.released,
	}
}

// Table maps logical token positions to allocator blocks for one request. It is
// not safe for concurrent use. Every block reference held by the table is
// released exactly once by Release or transferred out by DetachPrefix or
// SplitPrefix.
type Table struct {
	alloc    *Allocator
	blocks   []int
	released bool
}

// NewTable creates an empty table on an allocator.
func NewTable(a *Allocator) *Table {
	return &Table{alloc: a}
}

// Allocator returns the backing allocator.
func (t *Table) Allocator() *Allocator { return t.alloc }

// Len returns the number of mapped blocks.
func (t *Table) Len() int { return len(t.blocks) }

// Blocks returns a copy of the block IDs in logical order.
func (t *Table) Blocks() []int { return append([]int(nil), t.blocks...) }

// Cover grows the table until it maps at least positions token positions.
// Newly acquired blocks remain owned by the table.
func (t *Table) Cover(positions int) error {
	if positions < 0 {
		return fmt.Errorf("memory: negative position coverage %d", positions)
	}
	need := (positions + t.alloc.blockSize - 1) / t.alloc.blockSize
	return t.Grow(need - len(t.blocks))
}

// Grow appends count freshly acquired blocks. On failure, blocks acquired by
// the failed call are released and the table is unchanged.
func (t *Table) Grow(count int) error {
	if t.released {
		return ErrTableReleased
	}
	if count <= 0 {
		return nil
	}
	added := make([]int, 0, count)
	for range count {
		id, err := t.alloc.Acquire()
		if err != nil {
			for _, id := range added {
				_ = t.alloc.Release(id)
			}
			return err
		}
		added = append(added, id)
	}
	t.blocks = append(t.blocks, added...)
	return nil
}

// Share appends blocks whose references have already been retained for the
// caller, transferring ownership into the table. The caller must not release
// those references again.
func (t *Table) Share(blocks []int) error {
	if t.released {
		return ErrTableReleased
	}
	for _, id := range blocks {
		if id < 0 || id >= t.alloc.Blocks() {
			return ErrInvalidBlock
		}
		if t.alloc.Refs(id) <= 0 {
			return fmt.Errorf("memory: Share of free block %d", id)
		}
	}
	t.blocks = append(t.blocks, blocks...)
	return nil
}

// Map resolves a token position to its physical block and in-block offset.
func (t *Table) Map(pos int) (block, offset int, err error) {
	if t.released {
		return 0, 0, ErrTableReleased
	}
	if pos < 0 {
		return 0, 0, fmt.Errorf("memory: negative position %d", pos)
	}
	index := pos / t.alloc.blockSize
	if index >= len(t.blocks) {
		return 0, 0, fmt.Errorf("memory: position %d is not mapped by %d blocks", pos, len(t.blocks))
	}
	return t.blocks[index], pos % t.alloc.blockSize, nil
}

// Bytes returns the maximum positions mapped by the table.
func (t *Table) Bytes() int { return len(t.blocks) * t.alloc.blockSize }

// DetachPrefix transfers ownership of the blocks covering exactly positions and
// removes them from the table. positions must be block-aligned. The caller owns
// the returned references.
func (t *Table) DetachPrefix(positions int) ([]int, error) {
	if t.released {
		return nil, ErrTableReleased
	}
	if positions <= 0 || positions%t.alloc.blockSize != 0 {
		return nil, fmt.Errorf("memory: prefix length %d is not a positive block multiple", positions)
	}
	need := positions / t.alloc.blockSize
	if need > len(t.blocks) {
		return nil, fmt.Errorf("memory: prefix of %d positions exceeds %d mapped positions",
			positions, t.Bytes())
	}
	out := append([]int(nil), t.blocks[:need]...)
	rest := append([]int(nil), t.blocks[need:]...)
	t.blocks = rest
	return out, nil
}

// SplitPrefix transfers the blocks covering the first positions token positions
// to the caller for prefix caching. When positions is not block-aligned, the
// partial block is copied through copyBlock into a fresh table-owned block so
// later writes cannot corrupt the cached prefix. The caller owns the returned
// references and must insert them into a prefix cache or release them.
func (t *Table) SplitPrefix(positions int, copyBlock func(dst, src int) error) ([]int, error) {
	if t.released {
		return nil, ErrTableReleased
	}
	if positions <= 0 {
		return nil, fmt.Errorf("memory: prefix length %d must be positive", positions)
	}
	need := (positions + t.alloc.blockSize - 1) / t.alloc.blockSize
	if need > len(t.blocks) {
		return nil, fmt.Errorf("memory: prefix of %d positions exceeds %d mapped positions",
			positions, t.Bytes())
	}
	out := append([]int(nil), t.blocks[:need]...)
	if positions%t.alloc.blockSize == 0 {
		t.blocks = append([]int(nil), t.blocks[need:]...)
		return out, nil
	}
	if copyBlock == nil {
		return nil, fmt.Errorf("memory: partial prefix split needs a block copy function")
	}
	newID, err := t.alloc.Acquire()
	if err != nil {
		return nil, err
	}
	if err := copyBlock(newID, out[need-1]); err != nil {
		_ = t.alloc.Release(newID)
		return nil, err
	}
	rest := append([]int(nil), t.blocks[need:]...)
	t.blocks = append([]int{newID}, rest...)
	return out, nil
}

// Release drops every block reference still held by the table. It is
// idempotent.
func (t *Table) Release() {
	if t == nil || t.released {
		return
	}
	for _, id := range t.blocks {
		_ = t.alloc.Release(id)
	}
	t.blocks = nil
	t.released = true
}
