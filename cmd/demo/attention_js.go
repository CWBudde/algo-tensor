package main

import (
	"syscall/js"

	"github.com/cwbudde/algo-tensor/internal/demo"
)

func init() {
	register("attention", map[string]Func{
		"weights": attentionWeights,
	})
}

// attentionWeights is weights(n: number, causal: boolean): Float32Array,
// the n × n attention weights of a toy sequence, row by row
// (demo.AttentionWeights). It is the scaffold's stub.
func attentionWeights(args []js.Value) (any, error) {
	n, err := argInt(args, 0, "n")
	if err != nil {
		return nil, err
	}

	causal, err := argBool(args, 1, "causal")
	if err != nil {
		return nil, err
	}

	w, err := demo.AttentionWeights(n, causal)
	if err != nil {
		return nil, err
	}

	return float32sToJS(w), nil
}
