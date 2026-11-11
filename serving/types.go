// Package serving composes the continuous scheduler, paged KV memory, prefix
// cache, sampling, and tokenizer streaming into request-level generation for
// one loaded model.
package serving

import (
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/scheduler"
)

// Sequence is one request's model execution state. It is driven by a single
// scheduler goroutine and must observe cancellation between calls.
type Sequence interface {
	// Prefill processes tokens. final selects whether output logits are
	// required; the returned logits are valid only when final is true.
	Prefill(tokens []int32, final bool) ([]float32, error)
	// Decode feeds one token and returns logits for the next position.
	Decode(token int32) ([]float32, error)
	// Snapshot serializes model state that is not described by the block table.
	Snapshot() ([]byte, error)
	// Close releases per-request execution resources. It does not release the
	// block table, which is owned by the serving manager.
	Close() error
}

// Backend creates sequences over a shared model and KV pool. Implementations
// must serialize model execution; the scheduler interleaves requests but does
// not run steps concurrently.
type Backend interface {
	// ModelKey identifies the model and every setting that affects KV values.
	// It is mixed into prefix-cache keys.
	ModelKey() string
	VocabSize() int
	BlockSize() int
	Allocator() *memory.Allocator
	// NewSession binds a block table with capacity positions at the given prefix
	// position. state is the snapshot to restore, or nil for a fresh sequence.
	NewSession(table *memory.Table, capacity, position int, state []byte) (Sequence, error)
	// CopyBlock copies a complete KV block for private prefix ownership.
	CopyBlock(dst, src int) error
	Close() error
}

// Request describes one generation request.
type Request struct {
	ID          string
	Prompt      []int32
	MaxTokens   int
	Sampler     *engine.Sampler
	StopTokens  []int32
	StopStrings []string
	IgnoreEOS   bool
	// Sink receives emitted text deltas. Returning an error terminates the
	// request with that error. The sink runs on the scheduler goroutine and must
	// not block indefinitely.
	Sink func(token int32, delta string) error
}

// Result reports the terminal state and timings of one request.
type Result struct {
	Tokens         []int32
	Text           string
	PromptTokens   int
	OutputTokens   int
	FinishReason   string
	TimeToFirst    time.Duration
	PrefillTime    time.Duration
	DecodeTime     time.Duration
	SamplingTime   time.Duration
	PrefixCacheHit bool
	PrefixTokens   int
	QueueWait      time.Duration
}

// Options configures the serving manager.
type Options struct {
	// ContextSize is the maximum number of tokens, prompt plus output, admitted
	// for one request. It must not exceed the backend KV capacity.
	ContextSize int
	// Prefix caching is enabled when PrefixCapacityBlocks is non-zero or
	// PrefixEnabled is set.
	PrefixEnabled        bool
	PrefixEntries        int
	PrefixCapacityBlocks int
	Scheduler            scheduler.Config
}

// Metrics combines scheduler, KV allocator, and prefix-cache views.
type Metrics struct {
	Scheduler  scheduler.Metrics `json:"scheduler"`
	KV         memory.Stats      `json:"kv"`
	Prefix     memory.CacheStats `json:"prefix"`
	Model      string            `json:"model"`
	MaxContext int               `json:"max_context"`
}

// Limits is the effective configuration reported by health and metrics
// endpoints. Durations are seconds so they serialize cleanly as JSON and
// Prometheus gauges.
type Limits struct {
	ContextSize           int     `json:"context_size"`
	BlockSize             int     `json:"block_size"`
	KVBlocks              int     `json:"kv_blocks"`
	MaxActive             int     `json:"max_active"`
	MaxQueue              int     `json:"max_queue"`
	MaxBatch              int     `json:"max_batch"`
	PrefillChunk          int     `json:"prefill_chunk"`
	PrefillPerCycle       int     `json:"prefill_per_cycle"`
	DecodePerCycle        int     `json:"decode_per_cycle"`
	RequestTimeoutSeconds float64 `json:"request_timeout_s"`
	QueueTimeoutSeconds   float64 `json:"queue_timeout_s"`
	PrefixEnabled         bool    `json:"prefix_enabled"`
	PrefixEntries         int     `json:"prefix_entries"`
	PrefixBlocks          int     `json:"prefix_blocks"`
}
