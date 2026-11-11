package serving

import (
	"strings"
	"testing"
)

func TestStopFilter(t *testing.T) {
	filter := newStopFilter([]string{"END"})
	first, stopped := filter.Feed("hello ")
	if stopped {
		t.Fatal("false stop")
	}
	out, stopped := filter.Feed("world END extra")
	if !stopped {
		t.Fatal("stop sequence was not detected")
	}
	if got := first + out; got != "hello world " {
		t.Fatalf("output %q, want %q", got, "hello world ")
	}
	if got := filter.Flush(); got != "" {
		t.Fatalf("flush after stop = %q", got)
	}
}

func TestStopFilterFlushWithoutStop(t *testing.T) {
	filter := newStopFilter([]string{"XYZ"})
	var got strings.Builder
	for _, part := range []string{"ab", "cd", "ef"} {
		out, _ := filter.Feed(part)
		got.WriteString(out)
	}
	got.WriteString(filter.Flush())
	if got.String() != "abcdef" {
		t.Fatalf("output %q", got.String())
	}
}

func TestUTF8StreamKeepsRunesTogether(t *testing.T) {
	var stream utf8Stream
	if got := stream.Feed("\xe4\xb8"); got != "" {
		t.Fatalf("partial rune emitted %q", got)
	}
	if got := stream.Feed("\xad"); got != "\u4e2d" {
		t.Fatalf("rune = %q", got)
	}
}
