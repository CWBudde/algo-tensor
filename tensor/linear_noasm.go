//go:build !amd64 || purego

// algo-tensor's own, not in go-whistle or go-pocket-tts.

package tensor

// linearDot3Asm reports whether linearDot3 runs an assembly kernel: not
// without amd64's.
const linearDot3Asm = false

// linearPanel computes the outputs of tokens [b0, b1) and weight rows
// [p0, p1), one dotF32 per output: arm64's dotF32NEON already takes every
// row of 48 values or more.
func linearPanel(y, x, w []float32, in, out, b0, b1, p0, p1 int) {
	linearDots(y, x, w, in, out, b0, b1, p0, p1)
}

// linearDot3 sets y[r*ldy] = dotF32(x[r*ldx:r*ldx+len(w)], w) for r < 3.
func linearDot3(y []float32, ldy int, x []float32, ldx int, w []float32) {
	for r := range linearGroup {
		y[r*ldy] = dotF32(x[r*ldx:r*ldx+len(w)], w)
	}
}
