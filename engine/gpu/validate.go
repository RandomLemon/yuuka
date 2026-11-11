//go:build hip && linux

package gpu

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

func validateMatrix(name string, matrix tensor.Matrix) error {
	if !matrix.DType.Supported() {
		return fmt.Errorf("tensor %q uses unsupported HIP dtype %s", name, matrix.DType)
	}
	if _, err := matrix.DType.RowBytes(matrix.Cols); err != nil {
		return fmt.Errorf("tensor %q: %w", name, err)
	}
	return nil
}

// validateModelSupport rejects unsupported dtypes and row widths before any
// device allocation. The planner never silently substitutes another path.
func validateModelSupport(model *qwen35.Model) error {
	if model == nil || model.Config == nil {
		return fmt.Errorf("nil model")
	}
	if err := validateMatrix("token_embd.weight", model.Embedding); err != nil {
		return err
	}
	if err := validateMatrix("output.weight", model.Output); err != nil {
		return err
	}
	for i := range model.Layers {
		layer := &model.Layers[i]
		prefix := fmt.Sprintf("blk.%d.", i)
		if layer.Full != nil {
			for name, matrix := range map[string]tensor.Matrix{
				prefix + "attn_q.weight":      layer.Full.Q,
				prefix + "attn_k.weight":      layer.Full.K,
				prefix + "attn_v.weight":      layer.Full.V,
				prefix + "attn_output.weight": layer.Full.O,
			} {
				if err := validateMatrix(name, matrix); err != nil {
					return err
				}
			}
		} else if layer.Rec != nil {
			for name, matrix := range map[string]tensor.Matrix{
				prefix + "attn_qkv.weight":  layer.Rec.QKV,
				prefix + "attn_gate.weight": layer.Rec.Gate,
				prefix + "ssm_out.weight":   layer.Rec.Out,
				prefix + "ssm_beta.weight":  layer.Rec.Beta,
				prefix + "ssm_alpha.weight": layer.Rec.Alpha,
			} {
				if err := validateMatrix(name, matrix); err != nil {
					return err
				}
			}
		}
		for name, matrix := range map[string]tensor.Matrix{
			prefix + "ffn_gate.weight": layer.FFN.Gate,
			prefix + "ffn_up.weight":   layer.FFN.Up,
			prefix + "ffn_down.weight": layer.FFN.Down,
		} {
			if err := validateMatrix(name, matrix); err != nil {
				return err
			}
		}
	}
	return nil
}
