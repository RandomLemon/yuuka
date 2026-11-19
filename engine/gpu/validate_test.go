//go:build hip && linux

package gpu

import (
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

func TestValidateModelSupportRejectsUnsupportedDType(t *testing.T) {
	model := &qwen35.Model{
		Config:    &qwen35.Config{},
		Embedding: tensor.Matrix{DType: tensor.Q2_K, Rows: 1, Cols: 32},
		Output:    tensor.Matrix{DType: tensor.F32, Rows: 1, Cols: 32},
	}
	err := validateModelSupport(model)
	if err == nil {
		t.Fatal("unsupported dtype was accepted")
	}
	if !strings.Contains(err.Error(), "token_embd.weight") || !strings.Contains(err.Error(), "Q2_K") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateModelSupportAcceptsQwen35MoE(t *testing.T) {
	_, model, _ := tinyMoEFixture(t)
	if err := validateModelSupport(model); err != nil {
		t.Fatalf("qwen35moe fixture was rejected: %v", err)
	}
	for i := range model.Layers {
		if model.Layers[i].MoE == nil {
			t.Fatalf("layer %d has no MoE tensors", i)
		}
	}
}

func TestValidateModelSupportRejectsMoEExpertDType(t *testing.T) {
	_, model, _ := tinyMoEFixture(t)
	model.Layers[0].MoE.GateExps.DType = tensor.Q2_K
	err := validateModelSupport(model)
	if err == nil {
		t.Fatal("unsupported expert dtype was accepted")
	}
	if !strings.Contains(err.Error(), "ffn_gate_exps.weight") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateModelSupportRejectsBadRowWidth(t *testing.T) {
	model := &qwen35.Model{
		Config:    &qwen35.Config{},
		Embedding: tensor.Matrix{DType: tensor.Q4_K, Rows: 1, Cols: 100},
		Output:    tensor.Matrix{DType: tensor.F32, Rows: 1, Cols: 32},
	}
	if err := validateModelSupport(model); err == nil {
		t.Fatal("non-block-aligned row was accepted")
	}
}
