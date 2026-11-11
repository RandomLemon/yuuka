package qwen35

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/memory"
)

// KVPool holds host KV blocks for every full-attention layer. Blocks are
// addressed by allocator block ID, so block IDs from a shared memory.Allocator
// have the same meaning for every request using the pool.
type KVPool struct {
	cfg       *Config
	blockSize int
	blocks    int
	kvDim     int
	k         [][]float32
	v         [][]float32
}

// NewKVPool allocates a pool of fixed-size blocks. Recurrent layers have no KV
// storage and are left nil.
func NewKVPool(cfg *Config, blocks, blockSize int) (*KVPool, error) {
	if cfg == nil {
		return nil, fmt.Errorf("qwen35: config is required")
	}
	if blocks <= 0 || blockSize <= 0 {
		return nil, fmt.Errorf("qwen35: invalid KV pool geometry %d blocks of %d tokens", blocks, blockSize)
	}
	kvDim := cfg.KVHeadCount * cfg.HeadDim
	if kvDim <= 0 {
		return nil, fmt.Errorf("qwen35: invalid KV dimension %d", kvDim)
	}
	pool := &KVPool{
		cfg:       cfg,
		blockSize: blockSize,
		blocks:    blocks,
		kvDim:     kvDim,
		k:         make([][]float32, cfg.BlockCount),
		v:         make([][]float32, cfg.BlockCount),
	}
	elements := blocks * blockSize * kvDim
	for i := range cfg.BlockCount {
		if cfg.IsRecurrent(i) {
			continue
		}
		pool.k[i] = make([]float32, elements)
		pool.v[i] = make([]float32, elements)
	}
	return pool, nil
}

// BlockSize returns the number of token positions per block.
func (p *KVPool) BlockSize() int { return p.blockSize }

// Blocks returns the total block count.
func (p *KVPool) Blocks() int { return p.blocks }

// KVDim returns the per-token key/value width.
func (p *KVPool) KVDim() int { return p.kvDim }

// CopyBlock copies a complete block for every full-attention layer. It is used
// to give a request a private copy of a partially shared prefix block.
func (p *KVPool) CopyBlock(dst, src int) error {
	if dst < 0 || dst >= p.blocks || src < 0 || src >= p.blocks {
		return fmt.Errorf("qwen35: KV block copy %d <- %d is out of range", dst, src)
	}
	rows := p.blockSize * p.kvDim
	dstOff := dst * rows
	srcOff := src * rows
	for i := range p.cfg.BlockCount {
		if p.k[i] == nil {
			continue
		}
		copy(p.k[i][dstOff:dstOff+rows], p.k[i][srcOff:srcOff+rows])
		copy(p.v[i][dstOff:dstOff+rows], p.v[i][srcOff:srcOff+rows])
	}
	return nil
}

// ClearBlock zeros one block for every full-attention layer.
func (p *KVPool) ClearBlock(id int) {
	if id < 0 || id >= p.blocks {
		return
	}
	rows := p.blockSize * p.kvDim
	start := id * rows
	for i := range p.cfg.BlockCount {
		if p.k[i] == nil {
			continue
		}
		clear(p.k[i][start : start+rows])
		clear(p.v[i][start : start+rows])
	}
}

// elementOffset resolves a token position through a block table to the first
// element of its KV row.
func (p *KVPool) elementOffset(table *memory.Table, pos int) (int, error) {
	block, offset, err := table.Map(pos)
	if err != nil {
		return 0, err
	}
	if block < 0 || block >= p.blocks {
		return 0, fmt.Errorf("qwen35: block %d is outside the KV pool", block)
	}
	return (block*p.blockSize + offset) * p.kvDim, nil
}
