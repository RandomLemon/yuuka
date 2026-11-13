package tokenizer

import (
	"fmt"
	"strings"
)

// Message is one chat turn.
type Message struct {
	Role      string
	Content   string
	Reasoning string
}

// ChatOptions controls generation-prompt rendering.
type ChatOptions struct {
	AddGenerationPrompt bool
	// EnableThinking emits an open reasoning block for the next assistant
	// turn. When false the model receives an empty closed block.
	EnableThinking bool
}

// RenderChat renders the built-in Qwen chat format after checking that the
// GGUF metadata declares a Qwen-style template. The stored template is not
// executed. Unsupported message roles and non-Qwen templates are rejected
// rather than approximated.
func (t *Tokenizer) RenderChat(messages []Message, opts ChatOptions) (string, error) {
	if !strings.Contains(t.template, "<|im_start|>") {
		return "", fmt.Errorf("%w: chat template is not a Qwen-style template", ErrUnsupported)
	}
	if len(messages) == 0 {
		return "", fmt.Errorf("no chat messages provided")
	}
	lastUser := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUser = i
			break
		}
	}
	if lastUser < 0 {
		return "", fmt.Errorf("chat has no user message")
	}

	var out strings.Builder
	for i, message := range messages {
		role := message.Role
		content := strings.TrimSpace(message.Content)
		switch role {
		case "system":
			if i != 0 {
				return "", fmt.Errorf("system message must be first")
			}
			out.WriteString("<|im_start|>system\n")
			out.WriteString(content)
			out.WriteString("<|im_end|>\n")
		case "user":
			out.WriteString("<|im_start|>user\n")
			out.WriteString(content)
			out.WriteString("<|im_end|>\n")
		case "assistant":
			reasoning := strings.TrimSpace(message.Reasoning)
			if reasoning == "" && strings.Contains(content, "</think>") {
				before, after, _ := strings.Cut(content, "</think>")
				if _, inner, ok := strings.Cut(before, "<think>"); ok {
					reasoning = strings.TrimSpace(inner)
				}
				content = strings.TrimLeft(after, "\n")
			}
			out.WriteString("<|im_start|>assistant\n")
			if i > lastUser {
				out.WriteString("<think>\n")
				out.WriteString(reasoning)
				out.WriteString("\n</think>\n\n")
			}
			out.WriteString(content)
			out.WriteString("<|im_end|>\n")
		case "tool":
			return "", fmt.Errorf("%w: tool messages", ErrUnsupported)
		default:
			return "", fmt.Errorf("unsupported chat role %q", role)
		}
	}
	if opts.AddGenerationPrompt {
		out.WriteString("<|im_start|>assistant\n")
		if opts.EnableThinking {
			out.WriteString("<think>\n")
		} else {
			out.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return out.String(), nil
}

// RenderPrompt is a convenience wrapper for a single user turn with an optional
// system message and generation prompt.
func (t *Tokenizer) RenderPrompt(system, user string, opts ChatOptions) (string, error) {
	messages := make([]Message, 0, 2)
	if system != "" {
		messages = append(messages, Message{Role: "system", Content: system})
	}
	messages = append(messages, Message{Role: "user", Content: user})
	return t.RenderChat(messages, opts)
}
