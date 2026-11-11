package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func openSPMVocab(t *testing.T) *Tokenizer {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_SPM_VOCAB")
	if path == "" {
		t.Skip("YUUKA_TEST_SPM_VOCAB is not set")
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// Golden IDs are produced by llama.cpp's llama-tokenize with --no-bos on the
// checked-in ggml-vocab-llama-spm.gguf vocabulary.
func TestSentencePieceGolden(t *testing.T) {
	tok := openSPMVocab(t)
	cases := []struct {
		text string
		want []int32
	}{
		{"Hello", []int32{15043}},
		{"Hello world", []int32{15043, 3186}},
		{"Hello, world!", []int32{15043, 29892, 3186, 29991}},
		{"  leading", []int32{259, 8236}},
		{"trailing  ", []int32{25053, 259}},
		{"你好", []int32{29871, 30919, 31076}},
		{"😀🚀", []int32{29871, 243, 162, 155, 131, 243, 162, 157, 131}},
		{"a\nb", []int32{263, 13, 29890}},
		{"don't", []int32{1016, 29915, 29873}},
		{"we are", []int32{591, 526}},
		{"<s>", []int32{1}},
		{"1 2 3", []int32{29871, 29896, 29871, 29906, 29871, 29941}},
	}
	for _, tc := range cases {
		got, err := tok.Encode(tc.text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", tc.text, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func TestSentencePieceDecode(t *testing.T) {
	tok := openSPMVocab(t)
	for _, text := range []string{"Hello world", "你好", "a\nb", "😀🚀"} {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tok.Decode(ids)
		if err != nil {
			t.Fatal(err)
		}
		if got != text {
			t.Fatalf("Decode(Encode(%q)) = %q", text, got)
		}
	}
}

func TestTinySentencePieceFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-spm.gguf")
	if err := fixture.WriteTinySPMVocab(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	tok, err := NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{263, 268}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("Encode = %v, want %v", ids, want)
	}
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hello world" {
		t.Fatalf("Decode = %q", text)
	}
	piece, err := tok.DecodeToken(263)
	if err != nil {
		t.Fatal(err)
	}
	if piece != " Hello" {
		t.Fatalf("DecodeToken = %q", piece)
	}
	special, err := tok.Encode("<s>Hello")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(special, []int32{256, 263}) {
		t.Fatalf("special Encode = %v", special)
	}
}
