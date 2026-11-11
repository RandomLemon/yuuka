package qwen35

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/memory"
)

// KVPool holds host KV blocks for every full-attention layer. Blocks are
// addressed by allocator block ID, so block IDs from a shared memory.Allocator
// have the same meaning for every request using the pool. Keys and values use
// one KVFormat selected at construction time.
type KVPool struct {
	cfg       *Config
	blockSize int
	blocks    int
	kvDim     int
	format    KVFormat
	rowBytes  int
	k         [][]byte
	v         [][]byte
}

// NewKVPool allocates a pool of fixed-size F32 blocks. Recurrent layers have no
// KV storage and are left nil.
func NewKVPool(cfg *Config, blocks, blockSize int) (*KVPool, error) {
	return NewKVPoolWithFormat(cfg, blocks, blockSize, KVF32)
}

// NewKVPoolWithFormat allocates a pool of fixed-size blocks in format.
func NewKVPoolWithFormat(cfg *Config, blocks, blockSize int, format KVFormat) (*KVPool, error) {
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
	rowBytes, err := format.RowBytes(kvDim)
	if err != nil {
		return nil, err
	}
	pool := &KVPool{
		cfg:       cfg,
		blockSize: blockSize,
		blocks:    blocks,
		kvDim:     kvDim,
		format:    format,
		rowBytes:  rowBytes,
		k:         make([][]byte, cfg.BlockCount),
		v:         make([][]byte, cfg.BlockCount),
	}
	bytes := blocks * blockSize * rowBytes
	for i := range cfg.BlockCount {
		if cfg.IsRecurrent(i) {
			continue
		}
		pool.k[i] = make([]byte, bytes)
		pool.v[i] = make([]byte, bytes)
	}
	return pool, nil
}

// BlockSize returns the number of token positions per block.
func (p *KVPool) BlockSize() int { return p.blockSize }

// Blocks returns the total block count.
func (p *KVPool) Blocks() int { return p.blocks }

// KVDim returns the per-token key/value width.
func (p *KVPool) KVDim() int { return p.kvDim }

// Format returns the storage format.
func (p *KVPool) Format() KVFormat { return p.format }

// RowBytes returns the stored byte size of one key or value row.
func (p *KVPool) RowBytes() int { return p.rowBytes }

// CopyBlock copies a complete block for every full-attention layer. It is used
// to give a request a private copy of a partially shared prefix block.
func (p *KVPool) CopyBlock(dst, src int) error {
	if dst < 0 || dst >= p.blocks || src < 0 || src >= p.blocks {
		return fmt.Errorf("qwen35: KV block copy %d <- %d is out of range", dst, src)
	}
	bytes := p.blockSize * p.rowBytes
	dstOff := dst * bytes
	srcOff := src * bytes
	for i := range p.cfg.BlockCount {
		if p.k[i] == nil {
			continue
		}
		copy(p.k[i][dstOff:dstOff+bytes], p.k[i][srcOff:srcOff+bytes])
		copy(p.v[i][dstOff:dstOff+bytes], p.v[i][srcOff:srcOff+bytes])
	}
	return nil
}

// ClearBlock zeros one block for every full-attention layer.
func (p *KVPool) ClearBlock(id int) {
	if id < 0 || id >= p.blocks {
		return
	}
	bytes := p.blockSize * p.rowBytes
	start := id * bytes
	for i := range p.cfg.BlockCount {
		if p.k[i] == nil {
			continue
		}
		clear(p.k[i][start : start+bytes])
		clear(p.v[i][start : start+bytes])
	}
}

// rowOffset resolves a token position through a block table to the first byte
// of its KV row in layer's K and V arrays.
func (p *KVPool) rowOffset(table *memory.Table, pos int) (int, error) {
	block, offset, err := table.Map(pos)
	if err != nil {
		return 0, err
	}
	if block < 0 || block >= p.blocks {
		return 0, fmt.Errorf("qwen35: block %d is outside the KV pool", block)
	}
	return (block*p.blockSize + offset) * p.rowBytes, nil
}

// WriteRow stores one key and value row for layer at pos.
func (p *KVPool) WriteRow(layer int, table *memory.Table, pos int, k, v []float32) error {
	if layer < 0 || layer >= p.cfg.BlockCount || p.k[layer] == nil {
		return fmt.Errorf("qwen35: layer %d has no KV storage", layer)
	}
	if len(k) != p.kvDim || len(v) != p.kvDim {
		return fmt.Errorf("qwen35: KV row width %d/%d, want %d", len(k), len(v), p.kvDim)
	}
	offset, err := p.rowOffset(table, pos)
	if err != nil {
		return err
	}
	if err := p.format.EncodeRow(p.k[layer][offset:offset+p.rowBytes], k); err != nil {
		return err
	}
	return p.format.EncodeRow(p.v[layer][offset:offset+p.rowBytes], v)
}

// ReadK decodes one key row for layer at pos into dst.
func (p *KVPool) ReadK(layer int, table *memory.Table, pos int, dst []float32) error {
	return p.readRow(p.k, layer, table, pos, dst)
}

// ReadV decodes one value row for layer at pos into dst.
func (p *KVPool) ReadV(layer int, table *memory.Table, pos int, dst []float32) error {
	return p.readRow(p.v, layer, table, pos, dst)
}

func (p *KVPool) readRow(pool [][]byte, layer int, table *memory.Table, pos int, dst []float32) error {
	if layer < 0 || layer >= p.cfg.BlockCount || pool[layer] == nil {
		return fmt.Errorf("qwen35: layer %d has no KV storage", layer)
	}
	if len(dst) != p.kvDim {
		return fmt.Errorf("qwen35: KV destination width %d, want %d", len(dst), p.kvDim)
	}
	offset, err := p.rowOffset(table, pos)
	if err != nil {
		return err
	}
	return p.format.DecodeRow(dst, pool[layer][offset:offset+p.rowBytes])
}
