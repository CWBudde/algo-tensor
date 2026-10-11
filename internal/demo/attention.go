package demo

import (
	"fmt"
	"math"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)

// attentionDepth is the query/key depth of the stub's toy sequence.
const attentionDepth = 4

// AttentionWeights returns the n × n attention weights softmax(q·kᵀ / √d)
// of a fixed toy sequence of n positions, row by row: row i is query i's
// distribution over the keys (tensor.MatMul, tensor.Softmax). With causal,
// ops.CausalMask limits it to keys 0…i. n is 1…256.
//
// It is the scaffold's stub for the attention demo.
func AttentionWeights(n int, causal bool) ([]float32, error) {
	if n < 1 || n > 256 {
		return nil, fmt.Errorf("demo: sequence length %d is not in 1…256", n)
	}

	q := make([]float32, n*attentionDepth)
	k := make([]float32, n*attentionDepth)

	for i := range n {
		for j := range attentionDepth {
			q[i*attentionDepth+j] = float32(math.Sin(float64((i + 1) * (j + 1))))
			k[i*attentionDepth+j] = float32(math.Cos(float64((i + 1) * (j + 2))))
		}
	}

	weights, err := attentionWeights(q, k, n, causal)
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	return weights, nil
}

// attentionWeights computes softmax(q·kᵀ / √d) for [n, d] q and k, with
// the causal mask if causal.
func attentionWeights(q, k []float32, n int, causal bool) ([]float32, error) {
	shape := []int64{int64(n), attentionDepth}

	qt, err := tensor.New(q, shape)
	if err != nil {
		return nil, err
	}

	kt, err := tensor.New(k, shape)
	if err != nil {
		return nil, err
	}

	kT, err := kt.Transpose(0, 1)
	if err != nil {
		return nil, err
	}

	scores, err := tensor.MatMul(qt, kT)
	if err != nil {
		return nil, err
	}

	scale := float32(1 / math.Sqrt(attentionDepth))
	for i, v := range scores.RawData() {
		scores.RawData()[i] = v * scale
	}

	if causal {
		scores, err = ops.CausalMask(scores, 0)
		if err != nil {
			return nil, err
		}
	}

	weights, err := tensor.Softmax(scores, -1)
	if err != nil {
		return nil, err
	}

	return weights.Data(), nil
}
