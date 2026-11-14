package tokenizer

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
)

type tokenizerMetadata struct {
	key   string
	typ   gguf.MetadataType
	value []byte
}

func writeTokenizerFixture(t *testing.T, entries []tokenizerMetadata) string {
	t.Helper()
	var file bytes.Buffer
	file.WriteString("GGUF")
	writeTestU32(&file, 3)
	writeTestU64(&file, 0)
	writeTestU64(&file, uint64(len(entries)))
	for _, entry := range entries {
		writeTestU64(&file, uint64(len(entry.key)))
		file.WriteString(entry.key)
		writeTestU32(&file, uint32(entry.typ))
		file.Write(entry.value)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.gguf")
	if err := os.WriteFile(path, file.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeTestU32(w *bytes.Buffer, value uint32) {
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], value)
	w.Write(raw[:])
}

func writeTestU64(w *bytes.Buffer, value uint64) {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], value)
	w.Write(raw[:])
}

func testStringValue(value string) []byte {
	var out bytes.Buffer
	writeTestU64(&out, uint64(len(value)))
	out.WriteString(value)
	return out.Bytes()
}

func testStringArrayValue(values []string) []byte {
	var out bytes.Buffer
	writeTestU32(&out, uint32(gguf.MetaString))
	writeTestU64(&out, uint64(len(values)))
	for _, value := range values {
		out.Write(testStringValue(value))
	}
	return out.Bytes()
}

func testBoolValue(value bool) []byte {
	if value {
		return []byte{1}
	}
	return []byte{0}
}

func testUint32Value(value uint32) []byte {
	var out bytes.Buffer
	writeTestU32(&out, value)
	return out.Bytes()
}

func validTokenizerMetadata() []tokenizerMetadata {
	return []tokenizerMetadata{
		{"tokenizer.ggml.model", gguf.MetaString, testStringValue("gpt2")},
		{"tokenizer.ggml.pre", gguf.MetaString, testStringValue("qwen35")},
		{"tokenizer.ggml.tokens", gguf.MetaArray, testStringArrayValue([]string{"a", "b", "c"})},
		{"tokenizer.ggml.merges", gguf.MetaArray, testStringArrayValue(nil)},
		{"tokenizer.ggml.add_bos_token", gguf.MetaBool, testBoolValue(false)},
		{"tokenizer.ggml.add_space_prefix", gguf.MetaBool, testBoolValue(true)},
		{"tokenizer.chat_template", gguf.MetaString, testStringValue("<|im_start|>")},
	}
}

func replaceTokenizerMetadata(entries []tokenizerMetadata, key string, typ gguf.MetadataType, value []byte) []tokenizerMetadata {
	out := make([]tokenizerMetadata, 0, len(entries))
	for _, entry := range entries {
		if entry.key != key {
			out = append(out, entry)
		}
	}
	return append(out, tokenizerMetadata{key: key, typ: typ, value: value})
}

func TestNewFromGGUFRejectsMalformedOptionalMetadata(t *testing.T) {
	cases := []struct {
		name  string
		key   string
		value []byte
	}{
		{"pre", "tokenizer.ggml.pre", testUint32Value(1)},
		{"add_bos_token", "tokenizer.ggml.add_bos_token", testUint32Value(1)},
		{"add_space_prefix", "tokenizer.ggml.add_space_prefix", testUint32Value(1)},
		{"chat_template", "tokenizer.chat_template", testUint32Value(1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			entries := replaceTokenizerMetadata(validTokenizerMetadata(), tc.key, gguf.MetaUint32, tc.value)
			path := writeTokenizerFixture(t, entries)
			reader, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			if _, err := NewFromGGUF(reader); err == nil {
				t.Fatalf("malformed %s metadata was accepted", tc.key)
			}
		})
	}
}

func TestNonQwenTemplateRejectedAtRender(t *testing.T) {
	entries := replaceTokenizerMetadata(validTokenizerMetadata(),
		"tokenizer.chat_template", gguf.MetaString, testStringValue("plain template"))
	path := writeTokenizerFixture(t, entries)
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	tok, err := NewFromGGUF(reader)
	if err != nil {
		t.Fatalf("load with a non-Qwen template: %v", err)
	}
	if _, err := tok.RenderChat([]Message{{Role: "user", Content: "hello"}}, ChatOptions{}); err == nil {
		t.Fatal("non-Qwen template was accepted by RenderChat")
	}
}
