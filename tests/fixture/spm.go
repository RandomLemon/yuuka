package fixture

// WriteTinySPMVocab writes a tokenizer-only GGUF with a SentencePiece-style
// vocabulary. The vocabulary has all 256 byte-fallback tokens, control tokens,
// and enough merge-chain tokens to cover "Hello world" without external files.
func WriteTinySPMVocab(path string) error {
	tokens := make([]string, 0, 256+8+10)
	types := make([]int64, 0, 256+8+10)
	scores := make([]float64, 0, 256+8+10)
	for b := range 256 {
		tokens = append(tokens, byteToken(b))
		types = append(types, 6) // byte
		scores = append(scores, 0)
	}
	specials := []string{"<s>", "</s>", "<unk>"}
	for _, token := range specials {
		tokens = append(tokens, token)
		types = append(types, 3)
		scores = append(scores, 0)
	}
	normal := []struct {
		text  string
		score float64
	}{
		{"▁H", 1},
		{"▁He", 2},
		{"▁Hel", 3},
		{"▁Hell", 4},
		{"▁Hello", 5},
		{"▁w", 1},
		{"▁wo", 2},
		{"▁wor", 3},
		{"▁worl", 4},
		{"▁world", 5},
	}
	for _, token := range normal {
		tokens = append(tokens, token.text)
		types = append(types, 1)
		scores = append(scores, token.score)
	}

	b := newBuilder()
	b.addMetadata("general.architecture", stringValue("llama"))
	b.addMetadata("general.name", stringValue("yuuka-tiny-spm"))
	b.addMetadata("general.alignment", uintValue(alignment))
	b.addMetadata("tokenizer.ggml.model", stringValue("llama"))
	b.addMetadata("tokenizer.ggml.pre", stringValue("default"))
	b.addMetadata("tokenizer.ggml.tokens", stringArrayValue(tokens))
	b.addMetadata("tokenizer.ggml.scores", floatArrayValue(scores))
	b.addMetadata("tokenizer.ggml.token_type", intArrayValue(types))
	b.addMetadata("tokenizer.ggml.bos_token_id", uintValue(256))
	b.addMetadata("tokenizer.ggml.eos_token_id", uintValue(257))
	b.addMetadata("tokenizer.ggml.unknown_token_id", uintValue(258))
	b.addMetadata("tokenizer.ggml.add_bos_token", boolValue(true))
	b.addMetadata("tokenizer.ggml.add_space_prefix", boolValue(true))
	return b.write(path)
}

func byteToken(value int) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string(hex[value>>4]) + string(hex[value&0x0f]) + ">"
}

func floatArrayValue(values []float64) metadataValue {
	items := make([]metadataValue, len(values))
	for i, value := range values {
		items[i] = metadataValue{typ: 6, float: value}
	}
	return metadataValue{typ: 9, elem: 6, array: items}
}
