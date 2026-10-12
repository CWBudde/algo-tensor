//go:build amd64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts.

package tensor

// linearDot3Asm reports whether linearDot3 runs linearDot3AVX2: with AVX2
// and FMA, as dotF32AVX2.
var linearDot3Asm = useAVX2FMA

// linearPanel computes the outputs of tokens [b0, b1) and weight rows
// [p0, p1). Where dotF32 would take dotF32AVX2 (AVX2 + FMA, rows of 8
// values or more), groups of three tokens run over each weight row in
// linearDot3AVX2, which gives each output dotF32AVX2's bits; the tokens
// left over, and every row elsewhere, take dotF32 itself.
func linearPanel(y, x, w []float32, in, out, b0, b1, p0, p1 int) {
	if !useAVX2FMA || in < 8 {
		linearDots(y, x, w, in, out, b0, b1, p0, p1)
		return
	}

	b := b0
	for ; b+linearGroup <= b1; b += linearGroup {
		xs := x[b*in:]
		ys := y[b*out:]

		for o := p0; o < p1; o++ {
			linearDot3(ys[o:], out, xs, in, w[o*in:(o+1)*in])
		}
	}

	linearDots(y, x, w, in, out, b, b1, p0, p1)
}

// linearDot3 sets y[r*ldy] = dotF32AVX2(x[r*ldx:r*ldx+len(w)], w) for
// r < 3. len(w) must be at least 8 and the CPU must have AVX2 and FMA.
func linearDot3(y []float32, ldy int, x []float32, ldx int, w []float32) {
	n := len(w)
	_ = y[2*ldy]
	_ = x[2*ldx+n-1]

	linearDot3AVX2(&x[0], ldx, &w[0], n, &y[0], ldy)
}

// linearDot3AVX2 computes three dot products of n float32 values, of the
// rows x, x+ldx and x+2·ldx against the row w, into y, y+ldy and y+2·ldy,
// each in dotF32AVX2's order of operations and with its operands in the
// same places. n must be >= 1.
//
//go:noescape
func linearDot3AVX2(x *float32, ldx int, w *float32, n int, y *float32, ldy int)
