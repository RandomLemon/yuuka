package tokenizer

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

const spmSpace = "\u2581"

// parseByteToken recognizes SentencePiece byte-fallback tokens such as
// "<0x0A>".
func parseByteToken(text string) (byte, bool) {
	if len(text) != 6 || !strings.HasPrefix(text, "<0x") || text[5] != '>' {
		return 0, false
	}
	value, err := strconv.ParseUint(text[3:5], 16, 8)
	if err != nil {
		return 0, false
	}
	return byte(value), true
}

type spmSymbol struct {
	start int
	end   int
	prev  int
	next  int
}

// encodeSPM applies the SentencePiece merge algorithm used by Llama-family
// vocabularies. It greedily merges the highest-scoring adjacent pair whose
// concatenation exists in the vocabulary, then resegments any remaining span
// with byte fallback.
func (t *Tokenizer) encodeSPM(text string, out *[]int32) error {
	escaped := strings.ReplaceAll(text, " ", spmSpace)
	symbols := make([]spmSymbol, 0, len(escaped))
	for off := 0; off < len(escaped); {
		_, width := utf8.DecodeRuneInString(escaped[off:])
		if width == 0 {
			break
		}
		index := len(symbols)
		symbols = append(symbols, spmSymbol{
			start: off,
			end:   off + width,
			prev:  index - 1,
			next:  -1,
		})
		if index > 0 {
			symbols[index-1].next = index
		}
		off += width
	}

	revMerge := make(map[string][2]int)
	for {
		bestLeft := -1
		bestRight := -1
		bestText := ""
		bestScore := float32(0)
		for left := 0; left != -1; left = symbols[left].next {
			right := symbols[left].next
			if right < 0 {
				break
			}
			candidate := escaped[symbols[left].start:symbols[right].end]
			id, ok := t.ids[candidate]
			if !ok {
				continue
			}
			score := float32(0)
			if int(id) < len(t.scores) {
				score = t.scores[id]
			}
			if bestLeft < 0 || score > bestScore || (score == bestScore && left < bestLeft) {
				bestLeft = left
				bestRight = right
				bestText = candidate
				bestScore = score
			}
		}
		if bestLeft < 0 {
			break
		}
		symbols[bestLeft].end = symbols[bestRight].end
		symbols[bestLeft].next = symbols[bestRight].next
		if symbols[bestRight].next >= 0 {
			symbols[symbols[bestRight].next].prev = bestLeft
		}
		revMerge[bestText] = [2]int{bestLeft, bestRight}
	}

	for index := 0; index != -1; index = symbols[index].next {
		t.resegmentSPM(escaped, symbols, revMerge, index, out)
	}
	return nil
}

func (t *Tokenizer) resegmentSPM(text string, symbols []spmSymbol, revMerge map[string][2]int, index int, out *[]int32) {
	piece := text[symbols[index].start:symbols[index].end]
	if id, ok := t.ids[piece]; ok {
		*out = append(*out, id)
		return
	}
	if pair, ok := revMerge[piece]; ok {
		t.resegmentSPM(text, symbols, revMerge, pair[0], out)
		t.resegmentSPM(text, symbols, revMerge, pair[1], out)
		return
	}
	for i := 0; i < len(piece); i++ {
		id := t.byteIDs[piece[i]]
		if id < 0 {
			return
		}
		*out = append(*out, id)
	}
}

func (t *Tokenizer) decodeSPMToken(id int32) (string, error) {
	if id < 0 || int(id) >= len(t.vocab) {
		return "", fmt.Errorf("token id %d is out of range", id)
	}
	typ := int64(1)
	if int(id) < len(t.tokenTypes) {
		typ = t.tokenTypes[id]
	}
	switch typ {
	case 3, 4: // control and user-defined tokens are emitted literally
		return t.vocab[id], nil
	case 6:
		value, ok := t.byteOfID[id]
		if !ok {
			return "", fmt.Errorf("byte token %d has no byte value", id)
		}
		return string([]byte{value}), nil
	default:
		return strings.ReplaceAll(t.vocab[id], spmSpace, " "), nil
	}
}

func (t *Tokenizer) decodeSPM(ids []int32) (string, error) {
	var out strings.Builder
	removeSpace := t.addSpacePrefix
	for i, id := range ids {
		piece, err := t.decodeSPMToken(id)
		if err != nil {
			return "", err
		}
		if i == 0 {
			if t.bos >= 0 && id == t.bos {
				removeSpace = false
			}
			if removeSpace {
				piece = strings.TrimPrefix(piece, " ")
			}
		}
		out.WriteString(piece)
	}
	return out.String(), nil
}
