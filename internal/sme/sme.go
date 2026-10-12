// algo-tensor's own, not in go-whistle or go-pocket-tts: the bridge from
// ops.Conv2DPlan to tensor's SME2 kernel for prepacked operands (PLAN.md 2.2).

// Package sme lets ops.Conv2DPlan run its GEMM on the SME2 kernel of
// tensor.MatMulTransBBias (tensor/gemm_bias_sme_arm64.s) with operands it
// packed itself, without an exported entry point in tensor: tensor sets
// GEMM at start-up where it selects the kernel, and ops packs its
// operands with Pack in the kernel's panel layout.
//
// The layout (PLAN.md Phase 2): an operand of rows × k values, row-major,
// becomes ⌈rows/16⌉ panels of 16 rows. A panel holds Steps(k) = 4⌈k/4⌉
// steps of 16 values, k-major: lane l of step s is the value of the
// panel's row l at column s, for s < full = 4⌊k/4⌋. The tail group, steps
// full..full+3 (for k%4 > 0), replays gemmNEON4x4's reload of columns
// k-4..k-1 with the columns it already summed zeroed: step full+s holds
// column k-4+s for s >= 4-k%4 and +0 below. Lanes past the last row are
// +0.
package sme

// PanelRows is the number of rows in one panel.
const PanelRows = 16

// GEMM, when not nil, computes
//
//	c[i*ldc+j] = bias[j] + dot(a[i*k:(i+1)*k], b_j)   for i < m, j < n
//
// on the SME2 kernel, with the rows b_j packed by Pack into pb, a as rows
// and po as scratch of at least PanelRows·Steps(k) values, into which the
// kernel packs a panel by panel. It needs m >= 1, n a positive multiple of
// 4, k >= 4, ldc >= n and len(bias) >= n. It reports whether every output
// has gemmNEON4x4's bits; when it returns false (a NaN or infinite output,
// whose NaN bits the kernel does not keep, or a block given up after
// repeated signals) the caller must compute all of them again on NEON,
// for example with tensor.MatMulTransBBias in blocks of fewer than 16
// rows. tensor sets it once at start-up, on darwin/arm64 CPUs with SME2
// only; elsewhere it stays nil.
var GEMM func(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool

// Steps returns the number of steps of one panel for a depth of k, 4⌈k/4⌉.
func Steps(k int) int { return (k + 3) &^ 3 }

// PackedLen returns the number of values Pack writes for an operand of
// the given number of rows and depth k.
func PackedLen(rows, k int) int { return (rows + PanelRows - 1) / PanelRows * PanelRows * Steps(k) }

// Step returns the step that holds column col (< k) of a row of depth k,
// col itself before the tail group and col + 4 - k%4 in it.
func Step(col, k int) int {
	if full := k &^ 3; col >= full {
		return col + 4 - (k - full)
	}

	return col
}

// Pack packs the rows × k values of src, row-major, into the first
// PackedLen(rows, k) values of dst in the panel layout.
func Pack(dst, src []float32, rows, k int) {
	kp := Steps(k)
	dst = dst[:PackedLen(rows, k)]
	clear(dst)

	for r := range rows {
		panel := dst[r/PanelRows*PanelRows*kp:]
		row := src[r*k : (r+1)*k]

		for col, v := range row {
			panel[Step(col, k)*PanelRows+r%PanelRows] = v
		}
	}
}
