package tokenizer

import (
	"os"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
)

func openTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL is not set")
	}
	r, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	tok, err := NewFromGGUF(r)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestEncodeGolden(t *testing.T) {
	tok := openTestTokenizer(t)
	cases := []struct {
		text string
		want []int32
	}{
		{"Hello", []int32{9419}},
		{"Hello world", []int32{9419, 1814}},
		{"你好", []int32{109266}},
		{"1 2 3", []int32{16, 220, 17, 220, 18}},
		{"  leading", []int32{220, 6187}},
		{"trailing  ", []int32{371, 13868, 256}},
		{"a\nb", []int32{64, 198, 65}},
		{"don't", []int32{14572, 914}},
		{"we're I'LL", []int32{868, 2224, 353, 6, 3950}},
		{"Hello, world!", []int32{9419, 11, 1814, 0}},
		{"😀🚀", []int32{169706, 9008, 248, 222}},
		{"<|im_start|>user\nHi<|im_end|>\n", []int32{248045, 846, 198, 12675, 248046, 198}},
		{"3.14159", []int32{18, 13, 16, 19, 16, 20, 24}},
		{"a   b", []int32{64, 256, 292}},
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

func TestDecodeGolden(t *testing.T) {
	tok := openTestTokenizer(t)
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

func TestRenderChatPrompt(t *testing.T) {
	tok := openTestTokenizer(t)
	prompt, err := tok.RenderPrompt("You are helpful.", "Hi", ChatOptions{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if prompt != want {
		t.Fatalf("prompt = %q, want %q", prompt, want)
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []int32{248045, 8678, 198, 2523, 513, 10631, 13, 248046, 198,
		248045, 846, 198, 12675, 248046, 198,
		248045, 74455, 198, 248068, 271, 248069, 271}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Fatalf("chat token IDs = %v, want %v", ids, wantIDs)
	}
}
