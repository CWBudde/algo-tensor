package demo

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/algo-tensor/tensor"
)

// maxImageSide bounds an image's width and height, so that the byte count
// 4·width·height cannot overflow.
const maxImageSide = 1 << 14

// RGBAToNHWC converts width × height RGBA pixels, row by row (the layout of
// a canvas ImageData), to a [1, height, width, 3] tensor of the red, green
// and blue channels scaled to [0, 1]. Alpha is dropped. Width and height
// are 1…16384.
func RGBAToNHWC(pix []byte, width, height int) (*tensor.Tensor, error) {
	if width <= 0 || height <= 0 || width > maxImageSide || height > maxImageSide {
		return nil, fmt.Errorf("demo: image size %d×%d is not in 1…%d", width, height, maxImageSide)
	}

	if len(pix) != 4*width*height {
		return nil, fmt.Errorf("demo: %d bytes do not hold %d×%d RGBA pixels", len(pix), width, height)
	}

	data := make([]float32, 0, 3*width*height)
	for px := range slices.Chunk(pix, 4) {
		data = append(data, float32(px[0])/255, float32(px[1])/255, float32(px[2])/255)
	}

	t, err := tensor.New(data, []int64{1, int64(height), int64(width), 3})
	if err != nil {
		return nil, fmt.Errorf("demo: %w", err)
	}

	return t, nil
}

// NHWCToRGBA converts a [1, height, width, channels] tensor with values in
// [0, 1] to opaque RGBA pixels. Three channels are red, green and blue; one
// channel is gray. Values are clamped to [0, 1] and rounded; NaN is black.
func NHWCToRGBA(t *tensor.Tensor) (pix []byte, width, height int, err error) {
	shape := t.Shape()
	if len(shape) != 4 || shape[0] != 1 || (shape[3] != 1 && shape[3] != 3) {
		return nil, 0, 0, fmt.Errorf("demo: want a [1, height, width, 1 or 3] tensor, got %v", shape)
	}

	height, width, channels := int(shape[1]), int(shape[2]), int(shape[3])
	data := t.RawData()

	pix = make([]byte, 0, 4*width*height)

	for px := range slices.Chunk(data, channels) {
		if channels == 1 {
			g := toByte(px[0])
			pix = append(pix, g, g, g, 255)
		} else {
			pix = append(pix, toByte(px[0]), toByte(px[1]), toByte(px[2]), 255)
		}
	}

	return pix, width, height, nil
}

// toByte maps [0, 1] to 0…255, rounding to nearest and clamping.
func toByte(v float32) byte {
	if math.IsNaN(float64(v)) || v <= 0 {
		return 0
	}

	if v >= 1 {
		return 255
	}

	return byte(v*255 + 0.5)
}
