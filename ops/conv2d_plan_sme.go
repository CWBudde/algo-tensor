// algo-tensor's own, not in go-whistle or go-pocket-tts: Conv2DPlan's GEMM on
// tensor's SME2 kernel with the weights packed once (PLAN.md 2.2).

package ops

import (
	"github.com/cwbudde/algo-tensor/internal/sme"
	"github.com/cwbudde/algo-tensor/tensor"
)

// The plans and calls that run on the SME2 kernel (sme.GEMM), from
// BenchmarkConv2DPlanBasicPitch, BenchmarkConv2DPlanWhistleStem,
// BenchmarkConv2DRouting and a sweep of 3×3 convolutions on an Apple M5 Pro
// (ops/PERFORMANCE.md). A plan packs its weights for the kernel when its
// GEMM is one gemmNEON4x4 takes (k >= 4, outC a multiple of 4) and
//
//   - k >= conv2DPlanSMEMinK: with less depth the reduction of the four ZA
//     tiles outweighs the few outer products (go-whistle's stem, k = 9,
//     took 1.3 times as long);
//   - k · outC · outC/(16⌈outC/16⌉) >= conv2DPlanSMEMinWork: the work per
//     output position, counted in the columns that fill a 16-column tile
//     (8 out channels leave half of every block empty);
//   - outH · outW >= conv2DPlanSMEMinPositions: Conv2D builds a plan per
//     call, and on small images the packing, the larger scratch and the
//     kernel's start cost more than they save (all routing shapes lost).
//
// A call of Forward or ForwardRows then runs on the kernel when its rows
// hold at least conv2DPlanSMEMinFMAs multiply-adds: the SME unit takes a
// few microseconds longer to start after Go code ran (3 µs after 1 µs, 10
// µs after 66 µs), which a few rows of a light layer do not repay. Both
// paths give every output gemmNEON4x4's bits. Tests lower the thresholds.
var (
	conv2DPlanSMEMinK         = 32
	conv2DPlanSMEMinWork      = 1536
	conv2DPlanSMEMinPositions = 1024
	conv2DPlanSMEMinFMAs      = 1 << 19
)

// conv2DPlanSMEScratch is the number of values of im2col rows a plan
// unrolls per sme.GEMM call. The SME unit idles while Go unrolls the next
// chunk and starts slowly again, so the larger the chunk, the faster, as
// long as the rows stay in the cache: basic-pitch's conv2d_1 took 7.2 ms in
// chunks of 256 positions (0.9 MiB), 5.4 ms in chunks of 1024 (3.7 MiB) and
// 7.1 ms with its 170 MiB of rows in one chunk.
const conv2DPlanSMEScratch = 1 << 20

// setupSME packs the weights for sme.GEMM and sizes the scratch for it
// where the thresholds above select the kernel; elsewhere it leaves pb nil.
func (c *Conv2DPlan) setupSME() {
	if sme.GEMM == nil || !conv2DPlanSMETakes(c.k, c.outC, c.outH*c.outW) {
		return
	}

	c.pb = make([]float32, sme.PackedLen(c.outC, c.k))
	sme.Pack(c.pb, c.wt, c.outC, c.k)

	positions := (c.outH*c.outW + sme.PanelRows - 1) &^ (sme.PanelRows - 1)
	c.smeChunk = min(positions, max(sme.PanelRows, conv2DPlanSMEScratch/c.k&^(sme.PanelRows-1)))
	c.colLen = max(c.colLen, c.smeChunk*c.k+sme.PanelRows*sme.Steps(c.k))
}

// conv2DPlanSMETakes reports whether a plan of depth k, outC out channels
// and the given number of output positions packs its weights for sme.GEMM.
func conv2DPlanSMETakes(k, outC, positions int) bool {
	if k < 4 || outC%4 != 0 || k < conv2DPlanSMEMinK || positions < conv2DPlanSMEMinPositions {
		return false
	}

	tile := (outC + sme.PanelRows - 1) &^ (sme.PanelRows - 1)

	return float64(k)*float64(outC)*float64(outC)/float64(tile) >= float64(conv2DPlanSMEMinWork)
}

// rowsSME computes output rows [r0, r1) as rows does, on sme.GEMM: the
// output positions, row after row, in chunks of c.smeChunk, each unrolled
// into col and run through the kernel with the packed weights; col's tail
// is the kernel's panel scratch. A chunk the kernel reports as not
// bit-exact (a NaN or infinite output, or a block given up after repeated
// signals) is computed again on NEON from the same rows, in blocks of
// conv2DPlanBlockM as rows does.
func (c *Conv2DPlan) rowsSME(out, x, col []float32, r0, r1 int) {
	chunk := c.smeChunk
	a, po := col[:chunk*c.k], col[chunk*c.k:]
	p1 := r1 * c.outW

	for p := r0 * c.outW; p < p1; p += chunk {
		m := min(chunk, p1-p)
		dst := out[p*c.outC:]

		for i := 0; i < m; {
			y, x0 := (p+i)/c.outW, (p+i)%c.outW
			n := min(m-i, c.outW-x0)
			c.im2col(a[i*c.k:], x, y, x0, n)
			i += n
		}

		if sme.GEMM(dst, c.outC, a, po, c.pb, c.bias, m, c.outC, c.k) {
			continue
		}

		for i := 0; i < m; i += conv2DPlanBlockM {
			mb := min(conv2DPlanBlockM, m-i)
			tensor.MatMulTransBBias(dst[i*c.outC:], c.outC, a[i*c.k:(i+mb)*c.k], c.wt, c.bias, mb, c.outC, c.k)
		}
	}
}
