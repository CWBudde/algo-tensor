// algo-tensor's own, not in go-whistle or go-pocket-tts: Linear's loop order
// (go-laya Backlog B.20).

package tensor

// Linear computes every output as dotF32(x row, weight row), then + bias in
// float32, as it always has; only the order of the outputs changes here, so
// no bit does. One dot product per output streamed the whole weight matrix
// once per token, from memory for go-laya's 3–21 MiB matrices. linearRun
// instead runs every token over a panel of weight rows that fits in L2
// before it moves on to the next panel.

// linearPanelFloats is the weight budget of one panel: 256 KiB, which stays
// in a P-core's 1.25 MiB L2 next to the x rows (i7-1255U).
const linearPanelFloats = 1 << 16

// linearPanelMaxRows caps a panel of short rows: from about 128 values per
// row on, a longer panel saves nothing more.
const linearPanelMaxRows = 512

// linearGroup is the number of tokens the amd64 kernel runs over one weight
// row at once (linearPanel); parallel pieces of tokens start at multiples of
// it, so no group is cut short.
const linearGroup = 3

// linearParallelMinFMAs is the least work Linear splits over workers, as
// before the blocked loop.
const linearParallelMinFMAs = 1 << 18

// linearPieceFMAs is the least work of one parallel piece.
const linearPieceFMAs = 1 << 15

// linearPanelRows is the number of weight rows of in values in one panel.
func linearPanelRows(in int) int {
	return min(linearPanelMaxRows, max(1, linearPanelFloats/max(in, 1)))
}

// linearRun sets y[b*out+o] = dotF32(x[b*in:(b+1)*in], w[o*in:(o+1)*in])
// (+ bias[o] if bias is non-nil) for every b < batch and o < out. Above
// linearParallelMinFMAs it splits over Parallel: over outputs where that
// gives as many pieces as tokens would, else over tokens. Each output is
// still one call in one place, so the split does not change a bit.
func linearRun(y, x, w, bias []float32, batch, in, out int) {
	fmas := int64(batch) * int64(out) * int64(in)
	if getWorkers() <= 1 || fmas < linearParallelMinFMAs {
		linearBlock(y, x, w, bias, in, out, 0, batch, 0, out)
		return
	}

	if out >= min(batch, parallelSplit*getWorkers()) {
		Parallel(out, linearGrain(batch*in, 1), func(lo, hi int) {
			linearBlock(y, x, w, bias, in, out, 0, batch, lo, hi)
		})

		return
	}

	Parallel(batch, linearGrain(out*in, linearGroup), func(lo, hi int) {
		linearBlock(y, x, w, bias, in, out, lo, hi, 0, out)
	})
}

// linearGrain is the number of items, a multiple of unit, that holds at
// least linearPieceFMAs multiply-adds at perItem each.
func linearGrain(perItem, unit int) int {
	g := (linearPieceFMAs + perItem - 1) / max(perItem, 1)

	return (g + unit - 1) / unit * unit
}

// linearBlock computes the outputs of tokens [b0, b1) and weight rows
// [o0, o1), panel by panel: every token runs over one panel of weight rows
// before the next panel starts.
func linearBlock(y, x, w, bias []float32, in, out, b0, b1, o0, o1 int) {
	panel := linearPanelRows(in)

	for p0 := o0; p0 < o1; p0 += panel {
		p1 := min(o1, p0+panel)
		linearPanel(y, x, w, in, out, b0, b1, p0, p1)

		if bias != nil {
			for b := b0; b < b1; b++ {
				yRow := y[b*out : (b+1)*out]
				for o := p0; o < p1; o++ {
					yRow[o] += bias[o]
				}
			}
		}
	}
}

// linearDots sets y[b*out+o] = dotF32(x row b, w row o) for tokens
// [b0, b1) and weight rows [p0, p1), one call per output.
func linearDots(y, x, w []float32, in, out, b0, b1, p0, p1 int) {
	for b := b0; b < b1; b++ {
		xRow := x[b*in : (b+1)*in]
		yRow := y[b*out : (b+1)*out]

		for o := p0; o < p1; o++ {
			yRow[o] = dotF32(xRow, w[o*in:(o+1)*in])
		}
	}
}

// linearPanel computes the outputs of tokens [b0, b1) and weight rows
// [p0, p1).
func linearPanel(y, x, w []float32, in, out, b0, b1, p0, p1 int) {
	linearDots(y, x, w, in, out, b0, b1, p0, p1)
}
