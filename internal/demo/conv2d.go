package demo

import (
	"fmt"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)

// conv2dKernels are the stub's 3 × 3 filters, row by row.
var conv2dKernels = map[string][9]float32{
	"identity": {0, 0, 0, 0, 1, 0, 0, 0, 0},
	"box":      {1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9},
}

// Conv2DFilter filters width × height RGBA pixels with the 3 × 3 filter
// "identity" or "box" (a box blur): one depthwise ops.Conv2D over the red,
// green and blue channels with "SAME" zero padding, so a blur darkens the
// border. The result is opaque.
//
// It is the scaffold's stub for the conv2d demo.
func Conv2DFilter(pix []byte, width, height int, filter string) ([]byte, error) {
	const taps, channels = 3, 3

	taps3, ok := conv2dKernels[filter]
	if !ok {
		return nil, fmt.Errorf("demo: unknown filter %q", filter)
	}

	img, err := RGBAToNHWC(pix, width, height)
	if err != nil {
		return nil, err
	}

	// HWIO with groups = channels: one input channel per group, the same
	// taps for each.
	k := make([]float32, 0, taps*taps*channels)
	for _, v := range taps3 {
		k = append(k, v, v, v)
	}

	kernel, err := tensor.New(k, []int64{taps, taps, 1, channels})
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	top, bottom := ops.SamePadding(height, taps, 1)
	left, right := ops.SamePadding(width, taps, 1)

	out, err := ops.Conv2D(img, kernel, nil, ops.Conv2DParams{
		StrideH: 1, StrideW: 1,
		PadTop: top, PadBottom: bottom,
		PadLeft: left, PadRight: right,
		Groups: channels,
	})
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	filtered, _, _, err := NHWCToRGBA(out)

	return filtered, err
}
