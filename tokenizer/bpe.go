// Package tokenizer implements the tokenizer families used by supported GGUF
// models. The first implementation covers GPT-2-style byte-level BPE with the
// Qwen2/Qwen3.5 pre-tokenization rules.
package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/yuuka-ai/yuuka/gguf"
)

// ErrUnsupported is returned for tokenizer metadata that this build cannot
// construct. Callers must report it as a model capability failure.
var ErrUnsupported = fmt.Errorf("unsupported tokenizer")

// Tokenizer is immutable after construction and safe for concurrent encoding.
type Tokenizer struct {
	vocab          []string
	ids            map[string]int32
	merges         map[string]int
	byteRune       [256]rune
	runeByte       map[rune]byte
	specials       []special
	eos            int32
	bos            int32
	pad            int32
	model          string
	pre            string
	template       string
	scores         []float32
	tokenTypes     []int64
	byteIDs        [256]int32
	byteOfID       map[int32]byte
	spm            bool
	addSpacePrefix bool
	addBOS         bool
}

type special struct {
	text string
	id   int32
}

// NewFromGGUF builds a tokenizer from validated GGUF metadata.
func NewFromGGUF(r *gguf.Reader) (*Tokenizer, error) {
	model, err := r.String("tokenizer.ggml.model")
	if err != nil {
		return nil, err
	}
	spm := model == "llama" || model == "spm"
	if model != "gpt2" && !spm {
		return nil, fmt.Errorf("%w: tokenizer model %q", ErrUnsupported, model)
	}
	pre, _ := r.String("tokenizer.ggml.pre")
	if pre == "" {
		if spm {
			pre = "default"
		} else {
			pre = "gpt2"
		}
	}
	if !spm {
		switch pre {
		case "qwen35", "qwen2":
		default:
			return nil, fmt.Errorf("%w: tokenizer pre-tokenizer %q", ErrUnsupported, pre)
		}
	}
	tokensValue, ok := r.MetadataValue("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("%w: missing tokenizer.ggml.tokens", ErrUnsupported)
	}
	tokens, err := tokensValue.AsStrings()
	if err != nil {
		return nil, fmt.Errorf("tokenizer.ggml.tokens: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%w: empty token vocabulary", ErrUnsupported)
	}

	t := &Tokenizer{
		vocab:          tokens,
		ids:            make(map[string]int32, len(tokens)),
		merges:         make(map[string]int, 256),
		runeByte:       make(map[rune]byte, 256),
		model:          model,
		pre:            pre,
		byteIDs:        [256]int32{},
		byteOfID:       make(map[int32]byte, 256),
		spm:            spm,
		addSpacePrefix: spm,
		eos:            -1,
		bos:            -1,
		pad:            -1,
	}
	for i := range t.byteIDs {
		t.byteIDs[i] = -1
	}
	if !spm {
		t.byteRune, t.runeByte = buildByteMaps()
	}
	for i, token := range tokens {
		if _, exists := t.ids[token]; exists {
			return nil, fmt.Errorf("duplicate vocabulary token %q", token)
		}
		t.ids[token] = int32(i)
	}

	if !spm {
		mergeValue, ok := r.MetadataValue("tokenizer.ggml.merges")
		if !ok {
			return nil, fmt.Errorf("%w: missing tokenizer.ggml.merges", ErrUnsupported)
		}
		merges, err := mergeValue.AsStrings()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.merges: %w", err)
		}
		for i, merge := range merges {
			space := strings.IndexByte(merge, ' ')
			if space <= 0 || space == len(merge)-1 {
				return nil, fmt.Errorf("merge %d has invalid format %q", i, merge)
			}
			left, right := merge[:space], merge[space+1:]
			if strings.IndexByte(right, ' ') >= 0 {
				return nil, fmt.Errorf("merge %d has more than two symbols: %q", i, merge)
			}
			key := pairKey(left, right)
			if _, exists := t.merges[key]; exists {
				return nil, fmt.Errorf("duplicate merge %q", merge)
			}
			t.merges[key] = i
		}
	}

	if value, ok := r.MetadataValue("tokenizer.ggml.eos_token_id"); ok {
		id, err := value.AsInt()
		if err != nil || id < 0 || int(id) >= len(tokens) {
			return nil, fmt.Errorf("invalid eos_token_id %v", value)
		}
		t.eos = int32(id)
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.bos_token_id"); ok {
		id, err := value.AsInt()
		if err != nil || id < 0 || int(id) >= len(tokens) {
			return nil, fmt.Errorf("invalid bos_token_id %v", value)
		}
		t.bos = int32(id)
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.padding_token_id"); ok {
		id, err := value.AsInt()
		if err != nil || id < 0 || int(id) >= len(tokens) {
			return nil, fmt.Errorf("invalid padding_token_id %v", value)
		}
		t.pad = int32(id)
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.add_bos_token"); ok {
		add, err := value.AsBool()
		if err == nil {
			t.addBOS = add
		}
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.add_space_prefix"); ok {
		add, err := value.AsBool()
		if err == nil {
			t.addSpacePrefix = add
		}
	}
	if value, ok := r.MetadataValue("tokenizer.chat_template"); ok {
		text, err := value.AsString()
		if err == nil {
			t.template = text
		}
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.scores"); ok {
		scores, err := value.AsFloats()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.scores: %w", err)
		}
		if len(scores) != len(tokens) {
			return nil, fmt.Errorf("score count %d does not match vocabulary %d", len(scores), len(tokens))
		}
		t.scores = make([]float32, len(scores))
		for i := range scores {
			t.scores[i] = float32(scores[i])
		}
	} else if spm {
		return nil, fmt.Errorf("%w: missing tokenizer.ggml.scores", ErrUnsupported)
	}

	if value, ok := r.MetadataValue("tokenizer.ggml.token_type"); ok {
		types, err := value.AsInts()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.token_type: %w", err)
		}
		if len(types) != len(tokens) {
			return nil, fmt.Errorf("token_type count %d does not match vocabulary %d", len(types), len(tokens))
		}
		t.tokenTypes = types
		for i, typ := range types {
			switch typ {
			case 3, 4: // control and user-defined tokens are matched literally
				t.specials = append(t.specials, special{text: tokens[i], id: int32(i)})
			case 6:
				if spm {
					if byteValue, ok := parseByteToken(tokens[i]); ok {
						t.byteIDs[byteValue] = int32(i)
						t.byteOfID[int32(i)] = byteValue
					}
				}
			}
		}
	}
	if spm {
		for i, id := range t.byteIDs {
			if id < 0 {
				return nil, fmt.Errorf("%w: SentencePiece vocabulary has no byte token for 0x%02X", ErrUnsupported, i)
			}
		}
	}
	sort.Slice(t.specials, func(i, j int) bool {
		if len(t.specials[i].text) != len(t.specials[j].text) {
			return len(t.specials[i].text) > len(t.specials[j].text)
		}
		return t.specials[i].id < t.specials[j].id
	})
	return t, nil
}

func pairKey(a, b string) string {
	return a + "\x00" + b
}

func buildByteMaps() ([256]rune, map[rune]byte) {
	var byteToRune [256]rune
	runeToByte := make(map[rune]byte, 256)
	next := 0
	for b := 0; b < 256; b++ {
		if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
			byteToRune[b] = rune(b)
		} else {
			byteToRune[b] = rune(256 + next)
			next++
		}
	}
	for b := 0; b < 256; b++ {
		runeToByte[byteToRune[b]] = byte(b)
	}
	return byteToRune, runeToByte
}

// VocabSize returns the number of token IDs.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// EOSToken returns the EOS ID, or -1 when absent.
func (t *Tokenizer) EOSToken() int32 { return t.eos }

// BOSToken returns the BOS ID, or -1 when absent.
func (t *Tokenizer) BOSToken() int32 { return t.bos }

// PadToken returns the padding ID, or -1 when absent.
func (t *Tokenizer) PadToken() int32 { return t.pad }

// AddBOS reports whether the model metadata requests automatic BOS insertion.
func (t *Tokenizer) AddBOS() bool { return t.addBOS }

// ChatTemplate returns the GGUF chat template source, if present.
func (t *Tokenizer) ChatTemplate() string { return t.template }

// TokenText returns the vocabulary string for an ID.
func (t *Tokenizer) TokenText(id int32) (string, error) {
	if id < 0 || int(id) >= len(t.vocab) {
		return "", fmt.Errorf("token id %d is out of range", id)
	}
	return t.vocab[id], nil
}

// Encode converts UTF-8 text to token IDs. Special control tokens in the input
// are matched literally and are not split by BPE.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	out := make([]int32, 0, len(text)/3+1)
	start := 0
	prevSpecial := true
	encodeRaw := func(raw string) error {
		if raw == "" {
			return nil
		}
		if t.spm && prevSpecial && t.addSpacePrefix {
			raw = " " + raw
		}
		prevSpecial = false
		return t.encodeOrdinary(raw, &out)
	}
	for i := 0; i < len(text); {
		if id, n, ok := t.matchSpecial(text[i:]); ok {
			if err := encodeRaw(text[start:i]); err != nil {
				return nil, err
			}
			out = append(out, id)
			i += n
			start = i
			prevSpecial = true
			continue
		}
		_, width := utf8.DecodeRuneInString(text[i:])
		if width == 0 {
			break
		}
		i += width
	}
	if start < len(text) {
		if err := encodeRaw(text[start:]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (t *Tokenizer) matchSpecial(s string) (int32, int, bool) {
	for _, sp := range t.specials {
		if strings.HasPrefix(s, sp.text) {
			return sp.id, len(sp.text), true
		}
	}
	return 0, 0, false
}

func (t *Tokenizer) encodeOrdinary(text string, out *[]int32) error {
	if t.spm {
		return t.encodeSPM(text, out)
	}
	for _, piece := range t.preTokenize(text) {
		ids, err := t.bpe(piece)
		if err != nil {
			return err
		}
		*out = append(*out, ids...)
	}
	return nil
}

func (t *Tokenizer) bpe(piece string) ([]int32, error) {
	raw := []byte(piece)
	symbols := make([]string, len(raw))
	for i, b := range raw {
		symbols[i] = string(t.byteRune[b])
	}
	for len(symbols) > 1 {
		bestRank := int(^uint(0) >> 1)
		bestPos := -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := t.merges[pairKey(symbols[i], symbols[i+1])]; ok && rank < bestRank {
				bestRank = rank
				bestPos = i
			}
		}
		if bestPos < 0 {
			break
		}
		symbols[bestPos] += symbols[bestPos+1]
		symbols = append(symbols[:bestPos+1], symbols[bestPos+2:]...)
	}
	ids := make([]int32, len(symbols))
	for i, symbol := range symbols {
		id, ok := t.ids[symbol]
		if !ok {
			return nil, fmt.Errorf("BPE produced unknown symbol %q from %q", symbol, piece)
		}
		ids[i] = id
	}
	return ids, nil
}

// DecodeToken converts one token ID to its text piece. Unlike Decode, it does
// not remove a SentencePiece dummy-prefix space, so callers can stream pieces
// without losing inter-token spaces.
func (t *Tokenizer) DecodeToken(id int32) (string, error) {
	if t.spm {
		return t.decodeSPMToken(id)
	}
	return t.Decode([]int32{id})
}

// Decode converts token IDs to bytes. Invalid IDs return an error; the returned
// string may contain invalid UTF-8 when the model emits a partial byte token.
func (t *Tokenizer) Decode(ids []int32) (string, error) {
	if t.spm {
		return t.decodeSPM(ids)
	}
	var out strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.vocab) {
			return "", fmt.Errorf("token id %d is out of range", id)
		}
		for _, r := range t.vocab[id] {
			if b, ok := t.runeByte[r]; ok {
				out.WriteByte(b)
			} else {
				out.WriteRune(r)
			}
		}
	}
	return out.String(), nil
}

func (t *Tokenizer) preTokenize(text string) []string {
	runes := []rune(text)
	var pieces []string
	add := func(start, end int) {
		if end > start {
			pieces = append(pieces, string(runes[start:end]))
		}
	}
	isLetter := func(i int) bool {
		return i >= 0 && i < len(runes) && unicode.IsLetter(runes[i])
	}
	isMark := func(i int) bool {
		return i >= 0 && i < len(runes) && unicode.IsMark(runes[i])
	}
	isSpace := func(i int) bool {
		return i >= 0 && i < len(runes) && unicode.IsSpace(runes[i])
	}
	for pos := 0; pos < len(runes); {
		c := runes[pos]
		if c == '\'' && pos+1 < len(runes) {
			next := unicode.ToLower(runes[pos+1])
			if next == 's' || next == 't' || next == 'm' || next == 'd' {
				add(pos, pos+2)
				pos += 2
				continue
			}
			if pos+2 < len(runes) {
				nextNext := unicode.ToLower(runes[pos+2])
				if (next == 'r' && nextNext == 'e') ||
					(next == 'v' && nextNext == 'e') ||
					(next == 'l' && nextNext == 'l') {
					add(pos, pos+3)
					pos += 3
					continue
				}
			}
		}

		if !(c == '\r' || c == '\n' || unicode.IsNumber(c)) {
			if unicode.IsLetter(c) || unicode.IsMark(c) || isMark(pos+1) || isLetter(pos+1) {
				start := pos
				pos++
				for isLetter(pos) || isMark(pos) {
					pos++
				}
				add(start, pos)
				continue
			}
		}

		if unicode.IsNumber(c) {
			add(pos, pos+1)
			pos++
			continue
		}

		prefix := pos
		check := pos
		if c == ' ' {
			check = pos + 1
		}
		if check < len(runes) && !(unicode.IsSpace(runes[check]) || unicode.IsLetter(runes[check]) || unicode.IsMark(runes[check]) || unicode.IsNumber(runes[check])) {
			pos += (check - pos)
			for pos < len(runes) && !(unicode.IsSpace(runes[pos]) || unicode.IsLetter(runes[pos]) || unicode.IsMark(runes[pos]) || unicode.IsNumber(runes[pos])) {
				pos++
			}
			for pos < len(runes) && (runes[pos] == '\r' || runes[pos] == '\n') {
				pos++
			}
			add(prefix, pos)
			continue
		}

		count := 0
		lastNewline := 0
		for isSpace(pos + count) {
			if runes[pos+count] == '\r' || runes[pos+count] == '\n' {
				lastNewline = pos + count + 1
			}
			count++
		}
		if lastNewline > 0 {
			add(pos, lastNewline)
			pos = lastNewline
			continue
		}
		if count > 1 && pos+count < len(runes) {
			add(pos, pos+count-1)
			pos += count - 1
			continue
		}
		if count > 0 {
			add(pos, pos+count)
			pos += count
			continue
		}
		add(pos, pos+1)
		pos++
	}
	return pieces
}
