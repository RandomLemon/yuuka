package serving

import (
	"strings"
	"unicode/utf8"
)

// utf8Stream emits only complete UTF-8 sequences. Byte-level BPE can split a
// multi-byte rune across tokens.
type utf8Stream struct {
	pending []byte
}

func (s *utf8Stream) Feed(piece string) string {
	s.pending = append(s.pending, piece...)
	var out strings.Builder
	for len(s.pending) > 0 {
		r, width := utf8.DecodeRune(s.pending)
		if r == utf8.RuneError && width == 1 {
			if !utf8.FullRune(s.pending) {
				break
			}
			s.pending = s.pending[1:]
			out.WriteRune(utf8.RuneError)
			continue
		}
		out.Write(s.pending[:width])
		s.pending = s.pending[width:]
	}
	return out.String()
}

func (s *utf8Stream) Flush() string {
	if len(s.pending) == 0 {
		return ""
	}
	delta := strings.ToValidUTF8(string(s.pending), "\uFFFD")
	s.pending = nil
	return delta
}

// stopFilter buffers enough trailing text to detect any stop string without
// leaking buffered characters into the response.
type stopFilter struct {
	stops   []string
	maxLen  int
	buf     strings.Builder
	stopped bool
}

func newStopFilter(stops []string) *stopFilter {
	filter := &stopFilter{stops: stops}
	for _, stop := range stops {
		if len(stop) > filter.maxLen {
			filter.maxLen = len(stop)
		}
	}
	return filter
}

func (f *stopFilter) Feed(delta string) (string, bool) {
	if f.stopped {
		return "", true
	}
	if len(f.stops) == 0 {
		return delta, false
	}
	f.buf.WriteString(delta)
	text := f.buf.String()
	for _, stop := range f.stops {
		if index := strings.Index(text, stop); index >= 0 {
			prefix := text[:index]
			f.buf.Reset()
			f.stopped = true
			return prefix, true
		}
	}
	keep := f.maxLen - 1
	if keep < 0 {
		keep = 0
	}
	if len(text) <= keep {
		f.buf.Reset()
		f.buf.WriteString(text)
		return "", false
	}
	out := text[:len(text)-keep]
	f.buf.Reset()
	f.buf.WriteString(text[len(text)-keep:])
	return out, false
}

func (f *stopFilter) Flush() string {
	if f.stopped {
		return ""
	}
	text := f.buf.String()
	f.buf.Reset()
	return text
}

func (f *stopFilter) Stopped() bool { return f.stopped }
