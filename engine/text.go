package engine

import (
	"strings"
	"unicode/utf8"
)

// textStream emits only complete UTF-8 sequences. Byte-level BPE can split a
// multi-byte rune across tokens, so callers receive deltas as they become valid.
type textStream struct {
	pending []byte
	text    strings.Builder
}

func (s *textStream) Feed(piece string) string {
	s.pending = append(s.pending, piece...)
	var out strings.Builder
	for len(s.pending) > 0 {
		r, width := utf8.DecodeRune(s.pending)
		if r == utf8.RuneError && width == 1 {
			if !utf8.FullRune(s.pending) {
				break
			}
			// Invalid complete byte: replace it rather than leaking bad UTF-8.
			s.pending = s.pending[1:]
			out.WriteRune(utf8.RuneError)
			continue
		}
		out.Write(s.pending[:width])
		s.pending = s.pending[width:]
	}
	delta := out.String()
	s.text.WriteString(delta)
	return delta
}

func (s *textStream) Flush() string {
	if len(s.pending) == 0 {
		return ""
	}
	delta := strings.ToValidUTF8(string(s.pending), "\uFFFD")
	s.pending = nil
	s.text.WriteString(delta)
	return delta
}

func (s *textStream) Text() string { return s.text.String() }
