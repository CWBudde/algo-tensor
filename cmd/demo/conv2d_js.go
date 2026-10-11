package main

import (
	"syscall/js"

	"github.com/cwbudde/algo-tensor/internal/demo"
)

func init() {
	register("conv2d", map[string]Func{
		"filter": conv2dFilter,
	})
}

// conv2dFilter is filter(rgba: Uint8ClampedArray, width: number,
// height: number, filter: "identity" | "box"): Uint8ClampedArray, RGBA
// pixels filtered with a 3 × 3 depthwise Conv2D (demo.Conv2DFilter). It is
// the scaffold's stub.
func conv2dFilter(args []js.Value) (any, error) {
	pix, err := argBytes(args, 0, "rgba")
	if err != nil {
		return nil, err
	}

	width, err := argInt(args, 1, "width")
	if err != nil {
		return nil, err
	}

	height, err := argInt(args, 2, "height")
	if err != nil {
		return nil, err
	}

	filter, err := argString(args, 3, "filter")
	if err != nil {
		return nil, err
	}

	out, err := demo.Conv2DFilter(pix, width, height, filter)
	if err != nil {
		return nil, err
	}

	return bytesToJS(out), nil
}
