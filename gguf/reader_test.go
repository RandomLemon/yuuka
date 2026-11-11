package gguf

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

type testKV struct {
	key   string
	value Value
}

type testTensor struct {
	name   string
	dims   []uint64
	dtype  tensor.DType
	offset uint64
}

type testFile struct {
	version   uint32
	alignment uint64
	metadata  []testKV
	tensors   []testTensor
	data      []byte
}

func strValue(s string) Value  { return Value{Type: MetaString, String: s} }
func u32Value(n uint32) Value  { return Value{Type: MetaUint32, Uint: uint64(n)} }
func u64Value(n uint64) Value  { return Value{Type: MetaUint64, Uint: n} }
func f32Value(f float32) Value { return Value{Type: MetaFloat32, Float: float64(f)} }
func boolValue(b bool) Value   { return Value{Type: MetaBool, Bool: b} }
func strArray(v []string) Value {
	items := make([]Value, len(v))
	for i := range v {
		items[i] = Value{Type: MetaString, String: v[i]}
	}
	return Value{Type: MetaArray, ElemType: MetaString, Array: items}
}

func writeString(buf *strings.Builder, s string) {
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], uint64(len(s)))
	buf.Write(tmp[:])
	buf.WriteString(s)
}

func writeValue(buf *strings.Builder, v Value) {
	var tmp [8]byte
	binary.LittleEndian.PutUint32(tmp[:], uint32(v.Type))
	buf.Write(tmp[:4])
	switch v.Type {
	case MetaUint8:
		buf.WriteByte(byte(v.Uint))
	case MetaUint16:
		binary.LittleEndian.PutUint16(tmp[:], uint16(v.Uint))
		buf.Write(tmp[:2])
	case MetaUint32:
		binary.LittleEndian.PutUint32(tmp[:], uint32(v.Uint))
		buf.Write(tmp[:4])
	case MetaUint64:
		binary.LittleEndian.PutUint64(tmp[:], v.Uint)
		buf.Write(tmp[:8])
	case MetaInt32:
		binary.LittleEndian.PutUint32(tmp[:], uint32(int32(v.Int)))
		buf.Write(tmp[:4])
	case MetaFloat32:
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(v.Float)))
		buf.Write(tmp[:4])
	case MetaFloat64:
		binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(v.Float))
		buf.Write(tmp[:8])
	case MetaBool:
		if v.Bool {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case MetaString:
		writeString(buf, v.String)
	case MetaArray:
		binary.LittleEndian.PutUint32(tmp[:], uint32(v.ElemType))
		buf.Write(tmp[:4])
		binary.LittleEndian.PutUint64(tmp[:], uint64(len(v.Array)))
		buf.Write(tmp[:8])
		for _, item := range v.Array {
			writeRawValue(buf, item)
		}
	default:
		panic("unsupported test metadata type")
	}
}

func writeRawValue(buf *strings.Builder, v Value) {
	var tmp [8]byte
	switch v.Type {
	case MetaUint8:
		buf.WriteByte(byte(v.Uint))
	case MetaUint16:
		binary.LittleEndian.PutUint16(tmp[:], uint16(v.Uint))
		buf.Write(tmp[:2])
	case MetaUint32:
		binary.LittleEndian.PutUint32(tmp[:], uint32(v.Uint))
		buf.Write(tmp[:4])
	case MetaUint64:
		binary.LittleEndian.PutUint64(tmp[:], v.Uint)
		buf.Write(tmp[:8])
	case MetaFloat32:
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(v.Float)))
		buf.Write(tmp[:4])
	case MetaFloat64:
		binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(v.Float))
		buf.Write(tmp[:8])
	case MetaBool:
		if v.Bool {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case MetaString:
		writeString(buf, v.String)
	default:
		panic("unsupported test array element type")
	}
}

func writeTestFile(t *testing.T, path string, tf testFile) {
	t.Helper()
	if tf.version == 0 {
		tf.version = 3
	}
	if tf.alignment == 0 {
		tf.alignment = defaultAlign
	}
	var buf strings.Builder
	buf.WriteString(ggufMagic)
	var tmp [8]byte
	binary.LittleEndian.PutUint32(tmp[:], tf.version)
	buf.Write(tmp[:4])
	binary.LittleEndian.PutUint64(tmp[:], uint64(len(tf.tensors)))
	buf.Write(tmp[:8])
	binary.LittleEndian.PutUint64(tmp[:], uint64(len(tf.metadata)))
	buf.Write(tmp[:8])
	for _, kv := range tf.metadata {
		writeString(&buf, kv.key)
		writeValue(&buf, kv.value)
	}
	for _, info := range tf.tensors {
		writeString(&buf, info.name)
		binary.LittleEndian.PutUint32(tmp[:], uint32(len(info.dims)))
		buf.Write(tmp[:4])
		for _, dim := range info.dims {
			binary.LittleEndian.PutUint64(tmp[:], dim)
			buf.Write(tmp[:8])
		}
		binary.LittleEndian.PutUint32(tmp[:], uint32(info.dtype))
		buf.Write(tmp[:4])
		binary.LittleEndian.PutUint64(tmp[:], info.offset)
		buf.Write(tmp[:8])
	}
	for buf.Len()%int(tf.alignment) != 0 {
		buf.WriteByte(0)
	}
	buf.Write(tf.data)
	if err := os.WriteFile(path, []byte(buf.String()), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReaderValidF32(t *testing.T) {
	path := filepath.Join(t.TempDir(), "valid.gguf")
	data := make([]byte, 16)
	for i := range 4 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)+1))
	}
	writeTestFile(t, path, testFile{
		metadata: []testKV{
			{"general.architecture", strValue("test")},
			{"test.block_count", u32Value(2)},
			{"test.rating", f32Value(0.5)},
			{"test.enabled", boolValue(true)},
			{"tokenizer.ggml.tokens", strArray([]string{"a", "b", "c"})},
		},
		tensors: []testTensor{{"weight", []uint64{2, 2}, tensor.F32, 0}},
		data:    data,
	})
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.Version() != 3 || r.Alignment() != 32 {
		t.Fatalf("version=%d alignment=%d", r.Version(), r.Alignment())
	}
	if got, _ := r.String("general.architecture"); got != "test" {
		t.Fatalf("architecture=%q", got)
	}
	if got, _ := r.Uint("test.block_count"); got != 2 {
		t.Fatalf("block_count=%d", got)
	}
	if got, _ := r.Float("test.rating"); got != 0.5 {
		t.Fatalf("rating=%v", got)
	}
	value, _ := r.MetadataValue("tokenizer.ggml.tokens")
	tokens, err := value.AsStrings()
	if err != nil || len(tokens) != 3 || tokens[2] != "c" {
		t.Fatalf("tokens=%v err=%v", tokens, err)
	}
	tensorInfo, ok := r.Tensor("weight")
	if !ok {
		t.Fatal("missing tensor")
	}
	if tensorInfo.Rows() != 2 || tensorInfo.Columns() != 2 || tensorInfo.Bytes != 16 {
		t.Fatalf("unexpected tensor meta: rows=%d cols=%d bytes=%d",
			tensorInfo.Rows(), tensorInfo.Columns(), tensorInfo.Bytes)
	}
	raw, err := r.TensorData("weight")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := tensorInfo.Matrix()
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 2)
	if err := matrix.MatVec(got, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if got[0] != 1 || got[1] != 3 {
		t.Fatalf("matvec=%v", got)
	}
	if len(raw) != 16 {
		t.Fatalf("raw len=%d", len(raw))
	}
}

func TestReaderRejectsMalformed(t *testing.T) {
	tests := []struct {
		name string
		file testFile
		mut  func([]byte) []byte
		want error
	}{
		{
			name: "bad magic",
			file: testFile{},
			mut:  func(b []byte) []byte { b[0] = 'X'; return b },
			want: ErrMalformed,
		},
		{
			name: "unsupported version",
			file: testFile{version: 1},
			want: ErrUnsupported,
		},
		{
			name: "duplicate metadata",
			file: testFile{metadata: []testKV{
				{"a", u32Value(1)},
				{"a", u32Value(2)},
			}},
			want: ErrMalformed,
		},
		{
			name: "duplicate tensor",
			file: testFile{
				tensors: []testTensor{
					{"w", []uint64{2}, tensor.F32, 0},
					{"w", []uint64{2}, tensor.F32, 32},
				},
				data: make([]byte, 64),
			},
			want: ErrMalformed,
		},
		{
			name: "zero dimension",
			file: testFile{
				tensors: []testTensor{{"w", []uint64{0}, tensor.F32, 0}},
			},
			want: ErrMalformed,
		},
		{
			name: "quant width not divisible",
			file: testFile{
				tensors: []testTensor{{"w", []uint64{31, 2}, tensor.Q8_0, 0}},
				data:    make([]byte, 128),
			},
			want: ErrMalformed,
		},
		{
			name: "range outside file",
			file: testFile{
				tensors: []testTensor{{"w", []uint64{2}, tensor.F32, 1 << 20}},
				data:    make([]byte, 32),
			},
			want: ErrMalformed,
		},
		{
			name: "unaligned tensor offset",
			file: testFile{
				tensors: []testTensor{{"w", []uint64{2}, tensor.F32, 8}},
				data:    make([]byte, 64),
			},
			want: ErrMalformed,
		},
		{
			name: "overlapping tensors",
			file: testFile{
				tensors: []testTensor{
					{"a", []uint64{4}, tensor.F32, 0},
					{"b", []uint64{4}, tensor.F32, 0},
				},
				data: make([]byte, 32),
			},
			want: ErrMalformed,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.gguf")
			writeTestFile(t, path, tc.file)
			if tc.mut != nil {
				raw, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, tc.mut(raw), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			r, err := Open(path)
			if err == nil {
				r.Close()
				t.Fatal("expected error")
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("error %v is not %v", err, tc.want)
			}
		})
	}
}

func TestReaderSplitShards(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "model-00001-of-00002.gguf"),
		filepath.Join(dir, "model-00002-of-00002.gguf"),
	}
	for i, path := range paths {
		writeTestFile(t, path, testFile{
			metadata: []testKV{
				{"general.architecture", strValue("test")},
				{"split.no", u32Value(uint32(i))},
				{"split.count", u32Value(2)},
			},
			tensors: []testTensor{{
				name:   "blk." + string(rune('0'+i)),
				dims:   []uint64{2},
				dtype:  tensor.F32,
				offset: 0,
			}},
			data: make([]byte, 8),
		})
	}
	r, err := Open(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if len(r.Tensors()) != 2 {
		t.Fatalf("tensor count=%d", len(r.Tensors()))
	}
	if _, ok := r.Tensor("blk.1"); !ok {
		t.Fatal("second shard tensor missing")
	}
}

func TestReaderRealModel(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	arch, err := r.String("general.architecture")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("architecture=%s tensors=%d dtypes=%v", arch, len(r.Tensors()), r.DTypeSummary())
	for _, name := range []string{"token_embd.weight", "output_norm.weight"} {
		info, ok := r.Tensor(name)
		if !ok {
			t.Fatalf("missing tensor %q", name)
		}
		if _, err := info.Data(); err != nil {
			t.Fatalf("tensor %q data: %v", name, err)
		}
	}
}
