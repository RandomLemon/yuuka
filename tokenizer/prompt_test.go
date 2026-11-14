package tokenizer

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func openFixtureTokenizer(t *testing.T, write func(string) error) *Tokenizer {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer-fixture.gguf")
	if err := write(path); err != nil {
		t.Fatal(err)
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

func TestEncodePromptInsertsBOS(t *testing.T) {
	tok := openFixtureTokenizer(t, fixture.WriteTinySPMVocab)
	if !tok.AddBOS() {
		t.Fatal("fixture did not request BOS insertion")
	}

	raw, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{263, 268}; !reflect.DeepEqual(raw, want) {
		t.Fatalf("Encode = %v, want %v", raw, want)
	}

	cases := []struct {
		text string
		want []int32
	}{
		{"", []int32{256}},
		{"H", []int32{256, 259}},
		{"Hello world", []int32{256, 263, 268}},
		{"<s>Hello", []int32{256, 263}},
	}
	for _, tc := range cases {
		got, err := tok.EncodePrompt(tc.text)
		if err != nil {
			t.Fatalf("EncodePrompt(%q): %v", tc.text, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("EncodePrompt(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func TestEncodePromptAppendsEOS(t *testing.T) {
	tok := openFixtureTokenizer(t, fixture.WriteTinyQwen35)
	if tok.AddBOS() {
		t.Fatal("Qwen fixture unexpectedly requests BOS insertion")
	}
	eos := tok.EOSToken()
	if eos < 0 {
		t.Fatal("fixture has no EOS token")
	}
	tok.addEOS = true

	raw, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.EncodePrompt("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if want := append(append([]int32(nil), raw...), eos); !reflect.DeepEqual(got, want) {
		t.Fatalf("EncodePrompt = %v, want %v", got, want)
	}

	// An encoded trailing EOS is not duplicated.
	text, err := tok.DecodeToken(eos)
	if err != nil {
		t.Fatal(err)
	}
	withEOS, err := tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(withEOS) > 0 && withEOS[len(withEOS)-1] == eos {
		got, err = tok.EncodePrompt(text)
		if err != nil {
			t.Fatal(err)
		}
		if n := len(got); n == 0 || got[n-1] != eos || (n > 1 && got[n-2] == eos) {
			t.Fatalf("EncodePrompt(%q) duplicated EOS: %v", text, got)
		}
	}

	tok.eos = -1
	if _, err := tok.EncodePrompt("Hello"); err == nil {
		t.Fatal("add_eos_token without eos_token_id was accepted")
	}
}

func TestEncodePromptRespectsBOSSuppression(t *testing.T) {
	tok := openFixtureTokenizer(t, fixture.WriteTinyQwen35)
	if tok.AddBOS() {
		t.Fatal("Qwen fixture unexpectedly requests BOS insertion")
	}
	raw, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.EncodePrompt("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prompt, raw) {
		t.Fatalf("EncodePrompt = %v, Encode = %v", prompt, raw)
	}
}
