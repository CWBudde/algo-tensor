// From github.com/cwbudde/go-whistle internal/ops/tolerance.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/ops/tolerance.go @ 9faba38.
// go-whistle change: without the layer_norm, mlp and convtranspose1d entries; values unchanged.
// algo-tensor change: those three entries restored, so the map is go-pocket-tts' again.
// algo-tensor change: a conv2d entry for Conv2DPlan against Conv2D, conv1d's 2e-4; no other value changes.

package ops

import "fmt"

// Tolerance defines acceptable numeric drift of a kernel versus a float64 or
// framework reference output.
type Tolerance struct {
	Abs float64
	Rel float64
}

// KernelTolerances defines per-kernel parity targets. The values are
// go-pocket-tts' ONNX parity targets (docs/FINDINGS.md F-008). They are
// kernel-level bounds; a consumer's model-level fixtures (go-whistle's Needle
// blocks, for one) set their own, tighter acceptance.
var KernelTolerances = map[string]Tolerance{
	"matmul":          {Abs: 1e-4, Rel: 1e-4},
	"linear":          {Abs: 1e-4, Rel: 1e-4},
	"softmax":         {Abs: 1e-4, Rel: 1e-4},
	"layer_norm":      {Abs: 1e-4, Rel: 1e-4},
	"causal_mask":     {Abs: 0, Rel: 0},
	"rope":            {Abs: 2e-4, Rel: 2e-4},
	"attention":       {Abs: 2e-4, Rel: 2e-4},
	"mlp":             {Abs: 2e-4, Rel: 2e-4},
	"conv1d":          {Abs: 2e-4, Rel: 2e-4},
	"conv2d":          {Abs: 2e-4, Rel: 2e-4},
	"convtranspose1d": {Abs: 2e-4, Rel: 2e-4},
}

func KernelTolerance(name string) (Tolerance, error) {
	t, ok := KernelTolerances[name]
	if !ok {
		return Tolerance{}, fmt.Errorf("ops: no tolerance configured for kernel %q", name)
	}

	return t, nil
}
