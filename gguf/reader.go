// Package gguf reads GGUF model files without copying tensor data. It validates
// the header, metadata, tensor directory, alignment, shard assignment, and
// tensor byte ranges before exposing read-only tensor storage.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/yuuka-ai/yuuka/tensor"
)

const (
	ggufMagic       = "GGUF"
	minVersion      = 2
	maxVersion      = 3
	defaultAlign    = 32
	maxMetadataKeys = 1 << 20
	maxTensors      = 1 << 24
	maxStringBytes  = 1 << 30
	maxRank         = 8
)

var (
	ErrMalformed    = errors.New("malformed GGUF file")
	ErrUnsupported  = errors.New("unsupported GGUF feature")
	ErrInvalidModel = errors.New("invalid model metadata")
)

// Tensor describes one matrix or vector in a GGUF file.
type Tensor struct {
	Name       string
	Dimensions []uint64
	DType      tensor.DType
	Offset     uint64 // relative to the shard data section
	Absolute   uint64 // absolute file offset
	Bytes      uint64
	Shard      int
	shard      *shard
}

// Rows returns the product of dimensions after the first.
func (t *Tensor) Rows() int64 {
	if len(t.Dimensions) <= 1 {
		return 1
	}
	rows := uint64(1)
	for _, dim := range t.Dimensions[1:] {
		rows *= dim
	}
	return int64(rows)
}

// Columns returns the first dimension, which is the contiguous row width.
func (t *Tensor) Columns() int64 {
	if len(t.Dimensions) == 0 {
		return 0
	}
	return int64(t.Dimensions[0])
}

// Elements returns the total element count.
func (t *Tensor) Elements() uint64 {
	elements := uint64(1)
	for _, dim := range t.Dimensions {
		elements *= dim
	}
	return elements
}

// ShapeString formats dimensions in GGML order.
func (t *Tensor) ShapeString() string {
	parts := make([]string, len(t.Dimensions))
	for i, dim := range t.Dimensions {
		parts[i] = strconv.FormatUint(dim, 10)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Data returns the read-only mapped bytes for the tensor.
func (t *Tensor) Data() ([]byte, error) {
	if t.shard == nil {
		return nil, fmt.Errorf("tensor %q has no backing shard", t.Name)
	}
	end := t.Absolute + t.Bytes
	if t.Absolute > uint64(len(t.shard.data)) || end < t.Absolute || end > uint64(len(t.shard.data)) {
		return nil, fmt.Errorf("tensor %q range [%d,%d) exceeds shard %q size %d",
			t.Name, t.Absolute, end, t.shard.path, len(t.shard.data))
	}
	return t.shard.data[t.Absolute:end], nil
}

// Matrix exposes a 2D tensor as a packed matrix with rows=product(dims[1:])
// and columns=dims[0].
func (t *Tensor) Matrix() (tensor.Matrix, error) {
	data, err := t.Data()
	if err != nil {
		return tensor.Matrix{}, err
	}
	return tensor.Matrix{
		Data:  data,
		Rows:  t.Rows(),
		Cols:  t.Columns(),
		DType: t.DType,
	}, nil
}

type shard struct {
	path       string
	data       []byte
	dataOffset uint64
}

// Reader is an open, validated GGUF model. It owns memory mappings for every
// shard and must be closed exactly once.
type Reader struct {
	path      string
	version   uint32
	alignment uint64
	metadata  map[string]Value
	tensors   map[string]*Tensor
	order     []*Tensor
	shards    []*shard
}

type cursor struct {
	path string
	data []byte
	off  int
}

func (c *cursor) need(n uint64) error {
	if n > uint64(len(c.data)-c.off) {
		return fmt.Errorf("%w: %s: truncated at byte %d (need %d, have %d)",
			ErrMalformed, c.path, c.off, n, len(c.data)-c.off)
	}
	return nil
}

func (c *cursor) u8() (uint8, error) {
	if err := c.need(1); err != nil {
		return 0, err
	}
	v := c.data[c.off]
	c.off++
	return v, nil
}

func (c *cursor) u16() (uint16, error) {
	if err := c.need(2); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint16(c.data[c.off:])
	c.off += 2
	return v, nil
}

func (c *cursor) u32() (uint32, error) {
	if err := c.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(c.data[c.off:])
	c.off += 4
	return v, nil
}

func (c *cursor) u64() (uint64, error) {
	if err := c.need(8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(c.data[c.off:])
	c.off += 8
	return v, nil
}

func (c *cursor) bytes(n uint64) ([]byte, error) {
	if err := c.need(n); err != nil {
		return nil, err
	}
	v := c.data[c.off : c.off+int(n)]
	c.off += int(n)
	return v, nil
}

func (c *cursor) str() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringBytes {
		return "", fmt.Errorf("%w: %s: string length %d exceeds limit", ErrMalformed, c.path, n)
	}
	raw, err := c.bytes(n)
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

func mapFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	size := info.Size()
	if size <= 0 {
		return nil, fmt.Errorf("%w: %s: empty file", ErrMalformed, path)
	}
	if size > int64(math.MaxInt) {
		return nil, fmt.Errorf("%w: %s: file too large for this platform", ErrUnsupported, path)
	}
	data, err := syscall.Mmap(int(file.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}

func unmapFile(data []byte) error {
	if data == nil {
		return nil
	}
	return syscall.Munmap(data)
}

type fileMeta struct {
	version   uint32
	alignment uint64
	metadata  map[string]Value
	tensors   []*Tensor
	dataStart uint64
}

func parseFile(path string, data []byte) (*fileMeta, error) {
	c := &cursor{path: path, data: data}
	magic, err := c.bytes(4)
	if err != nil {
		return nil, err
	}
	if string(magic) != ggufMagic {
		return nil, fmt.Errorf("%w: %s: bad magic %q", ErrMalformed, path, magic)
	}
	version, err := c.u32()
	if err != nil {
		return nil, err
	}
	if version < minVersion || version > maxVersion {
		return nil, fmt.Errorf("%w: %s: version %d is not supported (want %d..%d)",
			ErrUnsupported, path, version, minVersion, maxVersion)
	}
	tensorCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	metadataCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	if tensorCount > maxTensors {
		return nil, fmt.Errorf("%w: %s: tensor count %d exceeds limit", ErrMalformed, path, tensorCount)
	}
	if metadataCount > maxMetadataKeys {
		return nil, fmt.Errorf("%w: %s: metadata count %d exceeds limit", ErrMalformed, path, metadataCount)
	}

	metadata := make(map[string]Value, metadataCount)
	for i := uint64(0); i < metadataCount; i++ {
		key, err := c.str()
		if err != nil {
			return nil, fmt.Errorf("%w: metadata key %d: %v", ErrMalformed, i, err)
		}
		if key == "" {
			return nil, fmt.Errorf("%w: %s: empty metadata key", ErrMalformed, path)
		}
		if _, exists := metadata[key]; exists {
			return nil, fmt.Errorf("%w: %s: duplicate metadata key %q", ErrMalformed, path, key)
		}
		value, err := readValue(c, 0)
		if err != nil {
			return nil, fmt.Errorf("%w: %s: metadata %q: %v", ErrMalformed, path, key, err)
		}
		metadata[key] = value
	}

	alignment := uint64(defaultAlign)
	if value, ok := metadata["general.alignment"]; ok {
		alignment, err = value.AsUint()
		if err != nil {
			return nil, fmt.Errorf("%w: %s: general.alignment: %v", ErrMalformed, path, err)
		}
		if alignment == 0 || alignment > 1<<20 || alignment&(alignment-1) != 0 {
			return nil, fmt.Errorf("%w: %s: general.alignment %d is not a power of two",
				ErrMalformed, path, alignment)
		}
	}

	tensors := make([]*Tensor, 0, tensorCount)
	seen := make(map[string]struct{}, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := c.str()
		if err != nil {
			return nil, fmt.Errorf("%w: tensor %d name: %v", ErrMalformed, i, err)
		}
		if name == "" {
			return nil, fmt.Errorf("%w: %s: empty tensor name", ErrMalformed, path)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("%w: %s: duplicate tensor name %q", ErrMalformed, path, name)
		}
		seen[name] = struct{}{}
		rank, err := c.u32()
		if err != nil {
			return nil, err
		}
		if rank == 0 || rank > maxRank {
			return nil, fmt.Errorf("%w: %s: tensor %q has invalid rank %d",
				ErrMalformed, path, name, rank)
		}
		dims := make([]uint64, rank)
		elements := uint64(1)
		for d := range dims {
			dim, err := c.u64()
			if err != nil {
				return nil, err
			}
			if dim == 0 {
				return nil, fmt.Errorf("%w: %s: tensor %q has zero dimension", ErrMalformed, path, name)
			}
			if elements > math.MaxUint64/dim {
				return nil, fmt.Errorf("%w: %s: tensor %q element count overflows",
					ErrMalformed, path, name)
			}
			elements *= dim
			dims[d] = dim
		}
		dtypeRaw, err := c.u32()
		if err != nil {
			return nil, err
		}
		offset, err := c.u64()
		if err != nil {
			return nil, err
		}
		dtype := tensor.DType(dtypeRaw)
		block, err := dtype.BlockLayout()
		if err != nil {
			return nil, fmt.Errorf("%w: %s: tensor %q: %v", ErrUnsupported, path, name, err)
		}
		if block.Elements > 1 && dims[0]%uint64(block.Elements) != 0 {
			return nil, fmt.Errorf("%w: %s: tensor %q row width %d is not a multiple of %d",
				ErrMalformed, path, name, dims[0], block.Elements)
		}
		if elements/uint64(block.Elements) > math.MaxUint64/uint64(block.Bytes) {
			return nil, fmt.Errorf("%w: %s: tensor %q byte size overflows", ErrMalformed, path, name)
		}
		bytes := elements / uint64(block.Elements) * uint64(block.Bytes)
		tensors = append(tensors, &Tensor{
			Name:       name,
			Dimensions: dims,
			DType:      dtype,
			Offset:     offset,
			Bytes:      bytes,
		})
	}

	dataStart := alignUp(uint64(c.off), alignment)
	if dataStart > uint64(len(data)) {
		if tensorCount != 0 {
			return nil, fmt.Errorf("%w: %s: tensor data starts beyond end of file", ErrMalformed, path)
		}
		dataStart = uint64(len(data))
	}
	return &fileMeta{
		version:   version,
		alignment: alignment,
		metadata:  metadata,
		tensors:   tensors,
		dataStart: dataStart,
	}, nil
}

func readValue(c *cursor, depth int) (Value, error) {
	rawType, err := c.u32()
	if err != nil {
		return Value{}, err
	}
	return readRawValue(c, MetadataType(rawType), depth)
}

func readRawValue(c *cursor, typ MetadataType, depth int) (Value, error) {
	if depth > 4 {
		return Value{}, fmt.Errorf("metadata nesting exceeds limit")
	}
	switch typ {
	case MetaUint8:
		v, err := c.u8()
		return Value{Type: typ, Uint: uint64(v)}, err
	case MetaInt8:
		v, err := c.u8()
		return Value{Type: typ, Int: int64(int8(v))}, err
	case MetaUint16:
		v, err := c.u16()
		return Value{Type: typ, Uint: uint64(v)}, err
	case MetaInt16:
		v, err := c.u16()
		return Value{Type: typ, Int: int64(int16(v))}, err
	case MetaUint32:
		v, err := c.u32()
		return Value{Type: typ, Uint: uint64(v)}, err
	case MetaInt32:
		v, err := c.u32()
		return Value{Type: typ, Int: int64(int32(v))}, err
	case MetaFloat32:
		v, err := c.u32()
		return Value{Type: typ, Float: float64(math.Float32frombits(v))}, err
	case MetaBool:
		v, err := c.u8()
		if err == nil && v > 1 {
			return Value{}, fmt.Errorf("invalid bool value %d", v)
		}
		return Value{Type: typ, Bool: v != 0}, err
	case MetaString:
		v, err := c.str()
		return Value{Type: typ, String: v}, err
	case MetaArray:
		elemRaw, err := c.u32()
		if err != nil {
			return Value{}, err
		}
		elemType := MetadataType(elemRaw)
		count, err := c.u64()
		if err != nil {
			return Value{}, err
		}
		minSize := uint64(1)
		switch elemType {
		case MetaUint8, MetaInt8, MetaBool:
			minSize = 1
		case MetaUint16, MetaInt16:
			minSize = 2
		case MetaUint32, MetaInt32, MetaFloat32:
			minSize = 4
		case MetaUint64, MetaInt64, MetaFloat64:
			minSize = 8
		case MetaString:
			minSize = 8
		case MetaArray:
			// A nested array value repeats its own type and count.
			minSize = 16
		default:
			return Value{}, fmt.Errorf("unknown array element type %s", elemType)
		}
		if elemType != MetaString && count > uint64(len(c.data)-c.off)/minSize {
			return Value{}, fmt.Errorf("array length %d exceeds remaining bytes", count)
		}
		if count > 1<<31 {
			return Value{}, fmt.Errorf("array length %d exceeds limit", count)
		}
		items := make([]Value, count)
		for i := range items {
			var item Value
			if elemType == MetaArray {
				item, err = readValue(c, depth+1)
			} else {
				item, err = readRawValue(c, elemType, depth+1)
			}
			if err != nil {
				return Value{}, eErrArrayElement(c, i, err)
			}
			if item.Type != elemType {
				return Value{}, eErrArrayElement(c, i,
					fmt.Errorf("element type %s does not match array type %s", item.Type, elemType))
			}
			items[i] = item
		}
		return Value{Type: typ, ElemType: elemType, Array: items}, nil
	case MetaUint64:
		v, err := c.u64()
		return Value{Type: typ, Uint: v}, err
	case MetaInt64:
		v, err := c.u64()
		return Value{Type: typ, Int: int64(v)}, err
	case MetaFloat64:
		v, err := c.u64()
		return Value{Type: typ, Float: math.Float64frombits(v)}, err
	default:
		return Value{}, fmt.Errorf("unknown metadata type %d", typ)
	}
}

func eErrArrayElement(c *cursor, index int, err error) error {
	return fmt.Errorf("%s: array element %d: %w", c.path, index, err)
}

func alignUp(value, alignment uint64) uint64 {
	if alignment <= 1 {
		return value
	}
	rem := value % alignment
	if rem == 0 {
		return value
	}
	add := alignment - rem
	if value > math.MaxUint64-add {
		return math.MaxUint64
	}
	return value + add
}

var splitPattern = regexp.MustCompile(`^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$`)

func splitPaths(path string, count uint64) ([]string, error) {
	match := splitPattern.FindStringSubmatch(path)
	if match == nil {
		return nil, fmt.Errorf("%w: split model path %q does not use -NNNNN-of-NNNNN.gguf",
			ErrMalformed, path)
	}
	total, err := strconv.ParseUint(match[3], 10, 32)
	if err != nil || total != count {
		return nil, fmt.Errorf("%w: split count %d does not match file name", ErrMalformed, count)
	}
	dir := filepath.Dir(match[1])
	base := filepath.Base(match[1])
	paths := make([]string, count)
	for i := uint64(0); i < count; i++ {
		paths[i] = filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i+1, count))
	}
	return paths, nil
}

// Open maps and validates a GGUF model, including split shards when present.
func Open(path string) (*Reader, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}
	data, err := mapFile(abs)
	if err != nil {
		return nil, err
	}
	meta, err := parseFile(abs, data)
	if err != nil {
		_ = unmapFile(data)
		return nil, err
	}

	shardCount := uint64(1)
	if value, ok := meta.metadata["split.count"]; ok {
		shardCount, err = value.AsUint()
		if err != nil {
			_ = unmapFile(data)
			return nil, fmt.Errorf("%w: %s: split.count: %v", ErrMalformed, abs, err)
		}
		if shardCount == 0 || shardCount > 1<<20 {
			_ = unmapFile(data)
			return nil, fmt.Errorf("%w: %s: invalid split.count %d", ErrMalformed, abs, shardCount)
		}
	}
	paths := []string{abs}
	if shardCount > 1 {
		paths, err = splitPaths(abs, shardCount)
		if err != nil {
			_ = unmapFile(data)
			return nil, err
		}
	}

	r := &Reader{
		path:      abs,
		version:   meta.version,
		alignment: meta.alignment,
		metadata:  meta.metadata,
		tensors:   make(map[string]*Tensor),
		shards:    make([]*shard, 0, shardCount),
	}
	mapped := make([][]byte, 0, shardCount)
	cleanup := func() {
		for _, m := range mapped {
			_ = unmapFile(m)
		}
	}

	for i, shardPath := range paths {
		shardData := data
		var shardMeta *fileMeta
		if i > 0 {
			shardData, err = mapFile(shardPath)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("split shard %d: %w", i, err)
			}
			shardMeta, err = parseFile(shardPath, shardData)
			if err != nil {
				_ = unmapFile(shardData)
				cleanup()
				return nil, err
			}
		} else {
			shardMeta = meta
		}
		mapped = append(mapped, shardData)
		if shardMeta.version != meta.version || shardMeta.alignment != meta.alignment {
			cleanup()
			return nil, fmt.Errorf("%w: split shard %d header differs from shard 0",
				ErrMalformed, i)
		}
		if shardCount > 1 {
			no, ok := shardMeta.metadata["split.no"]
			if !ok {
				cleanup()
				return nil, fmt.Errorf("%w: split shard %d has no split.no", ErrMalformed, i)
			}
			noValue, err := no.AsUint()
			if err != nil || noValue != uint64(i) {
				cleanup()
				return nil, fmt.Errorf("%w: split shard %d has split.no=%v", ErrMalformed, i, noValue)
			}
			count, ok := shardMeta.metadata["split.count"]
			if !ok {
				cleanup()
				return nil, fmt.Errorf("%w: split shard %d has no split.count", ErrMalformed, i)
			}
			countValue, err := count.AsUint()
			if err != nil || countValue != shardCount {
				cleanup()
				return nil, fmt.Errorf("%w: split shard %d has split.count=%v, want %d",
					ErrMalformed, i, countValue, shardCount)
			}
		}
		sh := &shard{path: shardPath, data: shardData, dataOffset: shardMeta.dataStart}
		r.shards = append(r.shards, sh)
		for _, info := range shardMeta.tensors {
			if _, exists := r.tensors[info.Name]; exists {
				cleanup()
				return nil, fmt.Errorf("%w: tensor %q appears in more than one shard",
					ErrMalformed, info.Name)
			}
			absOffset := sh.dataOffset + info.Offset
			if absOffset < sh.dataOffset {
				cleanup()
				return nil, fmt.Errorf("%w: tensor %q offset overflows", ErrMalformed, info.Name)
			}
			if info.Offset%meta.alignment != 0 {
				cleanup()
				return nil, fmt.Errorf("%w: tensor %q offset %d is not aligned to %d",
					ErrMalformed, info.Name, info.Offset, meta.alignment)
			}
			end := absOffset + info.Bytes
			if end < absOffset || end > uint64(len(shardData)) {
				cleanup()
				return nil, fmt.Errorf("%w: tensor %q range [%d,%d) exceeds shard size %d",
					ErrMalformed, info.Name, absOffset, end, len(shardData))
			}
			info.Absolute = absOffset
			info.Shard = i
			info.shard = sh
			r.tensors[info.Name] = info
			r.order = append(r.order, info)
		}
	}

	if err := r.validateOverlaps(); err != nil {
		cleanup()
		return nil, err
	}
	return r, nil
}

func (r *Reader) validateOverlaps() error {
	type span struct {
		start, end uint64
		name       string
	}
	byShard := make([][]span, len(r.shards))
	for _, t := range r.order {
		byShard[t.Shard] = append(byShard[t.Shard], span{t.Absolute, t.Absolute + t.Bytes, t.Name})
	}
	for shardIndex, spans := range byShard {
		sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
		for i := 1; i < len(spans); i++ {
			if spans[i].start < spans[i-1].end {
				return fmt.Errorf("%w: shard %d: tensors %q and %q overlap",
					ErrMalformed, shardIndex, spans[i-1].name, spans[i].name)
			}
		}
	}
	return nil
}

// Close releases every mapped shard. It is safe to call once; repeated calls
// return nil.
func (r *Reader) Close() error {
	if r == nil {
		return nil
	}
	var first error
	for _, sh := range r.shards {
		if err := unmapFile(sh.data); err != nil && first == nil {
			first = err
		}
		sh.data = nil
	}
	r.shards = nil
	return first
}

// Path returns the canonical path of the first shard.
func (r *Reader) Path() string { return r.path }

// Version returns the GGUF version.
func (r *Reader) Version() uint32 { return r.version }

// Alignment returns the data-section alignment.
func (r *Reader) Alignment() uint64 { return r.alignment }

// Metadata returns the decoded metadata map. Callers must not mutate it.
func (r *Reader) Metadata() map[string]Value { return r.metadata }

// MetadataValue returns one metadata entry.
func (r *Reader) MetadataValue(key string) (Value, bool) {
	v, ok := r.metadata[key]
	return v, ok
}

// String returns string metadata or an error.
func (r *Reader) String(key string) (string, error) {
	v, ok := r.metadata[key]
	if !ok {
		return "", fmt.Errorf("%w: missing metadata key %q", ErrInvalidModel, key)
	}
	s, err := v.AsString()
	if err != nil {
		return "", fmt.Errorf("%w: %q: %v", ErrInvalidModel, key, err)
	}
	return s, nil
}

// Uint returns unsigned integer metadata or an error.
func (r *Reader) Uint(key string) (uint64, error) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, fmt.Errorf("%w: missing metadata key %q", ErrInvalidModel, key)
	}
	n, err := v.AsUint()
	if err != nil {
		return 0, fmt.Errorf("%w: %q: %v", ErrInvalidModel, key, err)
	}
	return n, nil
}

// Int returns signed integer metadata or an error.
func (r *Reader) Int(key string) (int64, error) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, fmt.Errorf("%w: missing metadata key %q", ErrInvalidModel, key)
	}
	n, err := v.AsInt()
	if err != nil {
		return 0, fmt.Errorf("%w: %q: %v", ErrInvalidModel, key, err)
	}
	return n, nil
}

// Float returns float metadata or an error.
func (r *Reader) Float(key string) (float64, error) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, fmt.Errorf("%w: missing metadata key %q", ErrInvalidModel, key)
	}
	f, err := v.AsFloat()
	if err != nil {
		return 0, fmt.Errorf("%w: %q: %v", ErrInvalidModel, key, err)
	}
	return f, nil
}

// Tensor returns one tensor descriptor.
func (r *Reader) Tensor(name string) (*Tensor, bool) {
	t, ok := r.tensors[name]
	return t, ok
}

// Tensors returns all tensor descriptors in file order.
func (r *Reader) Tensors() []*Tensor { return r.order }

// TensorData returns the mapped bytes for a named tensor.
func (r *Reader) TensorData(name string) ([]byte, error) {
	t, ok := r.tensors[name]
	if !ok {
		return nil, fmt.Errorf("%w: missing tensor %q", ErrInvalidModel, name)
	}
	return t.Data()
}

// DTypeSummary counts tensors by dtype.
func (r *Reader) DTypeSummary() map[tensor.DType]int {
	out := make(map[tensor.DType]int)
	for _, t := range r.order {
		out[t.DType]++
	}
	return out
}
