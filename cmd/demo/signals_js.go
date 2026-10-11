package main

import (
	"syscall/js"

	"github.com/cwbudde/algo-tensor/internal/demo"
)

func init() {
	register("signals", map[string]Func{
		"movingAverage": signalsMovingAverage,
	})
}

// signalsMovingAverage is movingAverage(signal: Float32Array, taps: number):
// Float32Array, a centered moving average through ops.Conv1D
// (demo.SignalsMovingAverage). It is the scaffold's stub.
func signalsMovingAverage(args []js.Value) (any, error) {
	signal, err := argFloat32s(args, 0, "signal")
	if err != nil {
		return nil, err
	}

	taps, err := argInt(args, 1, "taps")
	if err != nil {
		return nil, err
	}

	out, err := demo.SignalsMovingAverage(signal, taps)
	if err != nil {
		return nil, err
	}

	return float32sToJS(out), nil
}
