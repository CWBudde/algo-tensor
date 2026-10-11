//go:build arm64 && !purego

// From github.com/cwbudde/algo-transcribe nn/gemm_arm64.go @ 21ad164.
// algo-tensor change: haveSIMD and gemmSIMD renamed to gemmBiasHaveSIMD and gemmBiasSIMD;
// out and wt renamed to c and b; the output row stride ldc instead of the contiguous n,
// passed on to both kernels, with c bounded to (m-1)*ldc+n elements and a panic for
// ldc < n.

package tensor

// gemmBiasHaveSIMD reports whether gemmBias has native SIMD kernels on
// this target. NEON is mandatory on arm64, so there is no runtime
// detection.
const gemmBiasHaveSIMD = true

// gemmNEON4x4 is the multi-channel NEON kernel (gemm_bias_arm64.s): 4×4
// tiles plus 1×4 tiles for the last m%4 rows. It needs m >= 1, k >= 4, n a
// positive multiple of 4 and ldc >= n, and reads and writes exactly the
// elements gemmBias does.
//
//go:noescape
func gemmNEON4x4(c, a, b, bias *float32, m, n, k, ldc int)

// gemmNEON8x1 is the single-channel (n = 1) NEON kernel: 8×1 tiles plus a
// 1×1 loop for the last m%8 rows. It needs m >= 1, k >= 4 and ldc >= 1.
//
//go:noescape
func gemmNEON8x1(c, a, b, bias *float32, m, k, ldc int)

// gemmBiasSIMD runs gemmBias with the NEON kernels and reports whether it
// did. It returns false, without touching c, for shapes no kernel covers:
// k < 4 (the kernels reload the last four elements of a row for the k%4
// tail), and n = len(bias) neither 1 nor a multiple of 4. The caller then
// runs the Go tiles.
//
// The kernels vectorise over k and reduce the partial sums at the end, so
// they round differently from the Go tiles, but each call is
// deterministic.
func gemmBiasSIMD(c []float32, ldc int, a, b, bias []float32, m, k int) bool {
	n := len(bias)
	if k < 4 || (n != 1 && n%4 != 0) {
		return false
	}

	// Bound the slices by their lengths before handing raw pointers to the
	// assembly, which reads exactly the first m*k and n*k elements of a and
	// b and writes only within the first (m-1)*ldc+n of c: s[0:x:len(s)]
	// panics when x > len(s), whereas s[:x] would only check the capacity.
	a = a[0 : m*k : len(a)]
	b = b[0 : n*k : len(b)]

	if m == 0 || n == 0 {
		return true
	}

	// Rows closer than n would overlap, and with a negative ldc the bound
	// below no longer covers every row.
	if ldc < n {
		panic("tensor: gemmBiasSIMD ldc < n")
	}

	c = c[0 : (m-1)*ldc+n : len(c)]

	if n == 1 {
		gemmNEON8x1(&c[0], &a[0], &b[0], &bias[0], m, k, ldc)
	} else {
		gemmNEON4x4(&c[0], &a[0], &b[0], &bias[0], m, n, k, ldc)
	}

	return true
}
