//go:build amd64 && !purego

// From github.com/cwbudde/algo-transcribe nn/gemm_amd64.go @ 58c8332.
// algo-tensor change: haveSIMD and gemmSIMD renamed to gemmBiasHaveSIMD and gemmBiasSIMD;
// out and wt renamed to c and b; the output row stride ldc instead of the contiguous n,
// passed on to both kernels, with c bounded to (m-1)*ldc+n elements and a panic for
// ldc < n, as in gemm_bias_arm64.go.

package tensor

import "golang.org/x/sys/cpu"

// gemmBiasHaveSIMD reports whether gemmBias has native SIMD kernels on this
// machine. The AVX2 kernels also need FMA (VFMADD231PS), so both features
// are checked, once, at package initialisation; without them gemmBias runs
// the Go tiles.
var gemmBiasHaveSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// gemmAVX2N4 is the multi-channel AVX2 kernel (gemm_bias_amd64.s): 2×4
// tiles plus 1×4 tiles for the last m%2 rows. It needs m >= 1, k >= 1, n a
// positive multiple of 4 and ldc >= n, and reads and writes exactly the
// elements gemmBias does.
//
//go:noescape
func gemmAVX2N4(c, a, b, bias *float32, m, n, k, ldc int)

// gemmAVX2N1 is the single-channel (n = 1) AVX2 kernel: 8×1 tiles plus a
// 1×1 loop for the last m%8 rows. It needs m >= 1, k >= 1 and ldc >= 1.
//
//go:noescape
func gemmAVX2N1(c, a, b, bias *float32, m, k, ldc int)

// gemmBiasSIMD runs gemmBias with the AVX2 kernels and reports whether it
// did. It returns false, without touching c, on CPUs without AVX2 and FMA
// and for shapes no kernel covers: n = len(bias) neither 1 nor a multiple
// of 4. It also declines k < 4, which the kernels would handle, so that
// both architectures take exactly the same shapes. The caller then runs
// the Go tiles.
//
// The kernels vectorise over k and reduce the partial sums at the end, so
// they round differently from the Go tiles, but each call is
// deterministic.
func gemmBiasSIMD(c []float32, ldc int, a, b, bias []float32, m, k int) bool {
	n := len(bias)
	if !gemmBiasHaveSIMD || k < 4 || (n != 1 && n%4 != 0) {
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
		gemmAVX2N1(&c[0], &a[0], &b[0], &bias[0], m, k, ldc)
	} else {
		gemmAVX2N4(&c[0], &a[0], &b[0], &bias[0], m, n, k, ldc)
	}

	return true
}
