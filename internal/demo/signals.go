package demo

import (
	"errors"
	"fmt"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)

// SignalsMovingAverage smooths a signal with a centered moving average of
// taps values: one ops.Conv1D with (taps − 1) / 2 zero padding on each side,
// so the output has the input's length. taps is odd and at most the length.
//
// It is the scaffold's stub for the signals demo.
func SignalsMovingAverage(signal []float32, taps int) ([]float32, error) {
	if len(signal) == 0 {
		return nil, errors.New("demo: the signal is empty")
	}

	if taps < 1 || taps%2 == 0 || taps > len(signal) {
		return nil, fmt.Errorf("demo: %d taps must be odd and in 1…%d", taps, len(signal))
	}

	// input [batch, channels, length], kernel [out, in, taps]
	input, err := tensor.New(signal, []int64{1, 1, int64(len(signal))})
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	kernel, err := tensor.Full([]int64{1, 1, int64(taps)}, 1/float32(taps))
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	out, err := ops.Conv1D(input, kernel, nil, 1, int64(taps-1)/2, 1, 1)
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	return out.Data(), nil
}
