package demo

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Float32sFromBytes decodes little-endian IEEE 754 float32 values, the
// memory layout of a JavaScript Float32Array under wasm.
func Float32sFromBytes(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("demo: %d bytes are not a whole number of float32 values", len(b))
	}

	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}

	return out, nil
}

// BytesFromFloat32s encodes values as little-endian IEEE 754 float32, the
// inverse of Float32sFromBytes.
func BytesFromFloat32s(values []float32) []byte {
	b := make([]byte, 0, 4*len(values))
	for _, v := range values {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}

	return b
}

// ExactInt converts a JavaScript number to an int if it is a whole number
// within ±2⁵³, the range in which a float64 holds every integer exactly.
func ExactInt(f float64) (int, bool) {
	const maxSafe = 1<<53 - 1

	if math.Trunc(f) != f || f < -maxSafe || f > maxSafe {
		return 0, false
	}

	return int(f), true
}
