package memory

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

// ErrCacheFull reports that a prefix entry does not fit the configured cache
// limits. Ownership of the rejected blocks and state remains with the caller.
var ErrCacheFull = errors.New("memory: prefix cache is full")

// PrefixConfig defines the identity and capacity of a prefix cache.
type PrefixConfig struct {
	// Key identifies the model and every setting that affects KV values. Two
	// models or tokenizer configurations must not share a Key.
	Key string
	// MaxEntries bounds the number of cached prefixes. Zero uses 64.
	MaxEntries int
	// MaxBlocks bounds the number of blocks held by cached prefixes. Zero is
	// bounded only by the allocator.
	MaxBlocks int
}

// CacheStats reports prefix-cache effectiveness and occupancy.
type CacheStats struct {
	Entries    int    `json:"entries"`
	Blocks     int    `json:"blocks"`
	Hits       uint64 `json:"hits"`
	Misses     uint64 `json:"misses"`
	Inserts    uint64 `json:"inserts"`
	Evictions  uint64 `json:"evictions"`
	Rejected   uint64 `json:"rejected"`
	BlockSize  int    `json:"block_size"`
	MaxEntries int    `json:"max_entries"`
	MaxBlocks  int    `json:"max_blocks"`
}

type prefixEntry struct {
	key    [32]byte
	tokens []int32
	blocks []int
	state  []byte
	logits []byte
	elem   *list.Element
}

// PrefixCache stores complete prompt prefixes. Lookup returns the longest
// cached token prefix of a prompt together with retained block references and
// the model state snapshot at that prefix length. It is safe for concurrent
// use.
type PrefixCache struct {
	alloc     *Allocator
	keyPrefix []byte
	maxEntry  int
	maxBlocks int

	mu       sync.Mutex
	entries  map[[32]byte]*prefixEntry
	order    *list.List // most recently used at the front
	blocks   int
	hits     uint64
	misses   uint64
	inserts  uint64
	evicted  uint64
	rejected uint64
}

// NewPrefixCache installs a cache as the allocator's eviction source.
func NewPrefixCache(a *Allocator, cfg PrefixConfig) *PrefixCache {
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 64
	}
	c := &PrefixCache{
		alloc:     a,
		keyPrefix: []byte(cfg.Key),
		maxEntry:  cfg.MaxEntries,
		maxBlocks: cfg.MaxBlocks,
		entries:   make(map[[32]byte]*prefixEntry),
		order:     list.New(),
	}
	a.SetEvictor(c.EvictBlocks)
	return c
}

// BlockSize returns the allocator block size.
func (c *PrefixCache) BlockSize() int { return c.alloc.BlockSize() }

// Key hashes a token prefix together with the configured model identity. The
// hash selects an insertion slot; Lookup also compares the full token slice so
// collisions cannot produce incorrect reuse.
func (c *PrefixCache) Key(tokens []int32) [32]byte {
	h := sha256.New()
	var scratch [8]byte
	h.Write([]byte("yuuka/prefix/v1\x00"))
	binary.LittleEndian.PutUint32(scratch[:4], uint32(c.alloc.BlockSize()))
	h.Write(scratch[:4])
	binary.LittleEndian.PutUint64(scratch[:], uint64(len(c.keyPrefix)))
	h.Write(scratch[:])
	h.Write(c.keyPrefix)
	binary.LittleEndian.PutUint64(scratch[:], uint64(len(tokens)))
	h.Write(scratch[:])
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(scratch[:4], uint32(token))
		h.Write(scratch[:4])
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// Match is a retained prefix snapshot. The cache holds the token and state
// slices; callers receive copies of the token and block lists and must either
// TakeBlocks for a table or Release the match.
type Match struct {
	Tokens []int32
	Blocks []int
	State  []byte
	Logits []byte
	taken  bool
}

// TakeBlocks transfers the cached block references to the caller. The match no
// longer releases them.
func (m *Match) TakeBlocks() []int {
	if m == nil || m.taken {
		return nil
	}
	m.taken = true
	return m.Blocks
}

// Release drops unused retained block references. It is a no-op after
// TakeBlocks.
func (m *Match) Release(a *Allocator) {
	if m == nil || m.taken {
		return
	}
	for _, id := range m.Blocks {
		_ = a.Release(id)
	}
	m.Blocks = nil
	m.taken = true
}

// Lookup returns the longest cached token prefix of prompt. The returned match
// holds retained block references that the caller must take or release.
func (c *PrefixCache) Lookup(prompt []int32) *Match {
	c.mu.Lock()
	defer c.mu.Unlock()
	var best *prefixEntry
	for _, entry := range c.entries {
		if len(entry.tokens) == 0 || len(entry.tokens) > len(prompt) {
			continue
		}
		if best != nil && len(entry.tokens) <= len(best.tokens) {
			continue
		}
		if equalTokens(entry.tokens, prompt[:len(entry.tokens)]) {
			best = entry
		}
	}
	if best == nil {
		c.misses++
		return nil
	}
	blocks := make([]int, len(best.blocks))
	for i, id := range best.blocks {
		if err := c.alloc.Retain(id); err != nil {
			for _, retained := range blocks[:i] {
				_ = c.alloc.Release(retained)
			}
			c.misses++
			return nil
		}
		blocks[i] = id
	}
	c.hits++
	c.order.MoveToFront(best.elem)
	return &Match{
		Tokens: append([]int32(nil), best.tokens...),
		Blocks: blocks,
		State:  best.state,
		Logits: best.logits,
	}
}

// Insert adds a prefix covering len(tokens) token positions. blocks must hold
// exactly the references covering those positions; ownership transfers to the
// cache on success and remains with the caller on error. state is the model
// snapshot at len(tokens) and logits is the serialized model output for the
// same position. Both are retained by the cache without copying.
func (c *PrefixCache) Insert(tokens []int32, blocks []int, state, logits []byte) error {
	if len(tokens) == 0 {
		return fmt.Errorf("memory: prefix tokens must not be empty")
	}
	want := (len(tokens) + c.alloc.BlockSize() - 1) / c.alloc.BlockSize()
	if len(blocks) != want {
		return fmt.Errorf("memory: prefix of %d tokens needs %d blocks, got %d",
			len(tokens), want, len(blocks))
	}
	for _, id := range blocks {
		if c.alloc.Refs(id) <= 0 {
			return fmt.Errorf("memory: prefix block %d is not referenced", id)
		}
	}
	key := c.Key(tokens)

	c.mu.Lock()
	defer c.mu.Unlock()
	if existing, ok := c.entries[key]; ok {
		c.removeLocked(existing, false)
	}
	if c.maxBlocks > 0 && len(blocks) > c.maxBlocks {
		c.rejected++
		return ErrCacheFull
	}
	for len(c.entries)+1 > c.maxEntry {
		if !c.evictLocked() {
			c.rejected++
			return ErrCacheFull
		}
	}
	if c.maxBlocks > 0 {
		for c.blocks+len(blocks) > c.maxBlocks {
			if !c.evictLocked() {
				c.rejected++
				return ErrCacheFull
			}
		}
	}
	entry := &prefixEntry{
		key:    key,
		tokens: append([]int32(nil), tokens...),
		blocks: blocks,
		state:  state,
		logits: logits,
	}
	entry.elem = c.order.PushFront(entry)
	c.entries[key] = entry
	c.blocks += len(blocks)
	c.inserts++
	return nil
}

// EvictBlocks releases least-recently-used entries until need block references
// have been released or no entry remains. It is the allocator eviction hook.
func (c *PrefixCache) EvictBlocks(need int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	released := 0
	for released < need {
		element := c.order.Back()
		if element == nil {
			break
		}
		entry := element.Value.(*prefixEntry)
		released += len(entry.blocks)
		c.removeLocked(entry, true)
	}
	return released
}

// Purge releases every cached prefix entry.
func (c *PrefixCache) Purge() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for c.order.Len() > 0 {
		entry := c.order.Back().Value.(*prefixEntry)
		c.removeLocked(entry, true)
	}
}

// Stats returns the current cache view.
func (c *PrefixCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return CacheStats{
		Entries:    len(c.entries),
		Blocks:     c.blocks,
		Hits:       c.hits,
		Misses:     c.misses,
		Inserts:    c.inserts,
		Evictions:  c.evicted,
		Rejected:   c.rejected,
		BlockSize:  c.alloc.BlockSize(),
		MaxEntries: c.maxEntry,
		MaxBlocks:  c.maxBlocks,
	}
}

func (c *PrefixCache) evictLocked() bool {
	element := c.order.Back()
	if element == nil {
		return false
	}
	entry := element.Value.(*prefixEntry)
	c.removeLocked(entry, true)
	return true
}

func (c *PrefixCache) removeLocked(entry *prefixEntry, eviction bool) {
	if entry.elem != nil {
		c.order.Remove(entry.elem)
	}
	delete(c.entries, entry.key)
	for _, id := range entry.blocks {
		_ = c.alloc.Release(id)
	}
	c.blocks -= len(entry.blocks)
	if eviction {
		c.evicted++
	}
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
