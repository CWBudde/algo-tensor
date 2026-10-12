// algo-tensor's own, not in go-whistle or go-pocket-tts.

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// MatMulTransB's contract off arm64's NEON kernel: every output is
// dotF32(a row, b row), as one DotProduct per output always gave, on every
// configuration and for any worker count. The tests in this file compare it
// with that loop bit for bit (math.Float32bits), so a faster loop order or
// kernel may change the speed, never a bit (PLAN.md 1.4).

// matMulTransBReference is MatMulTransB off arm64 as it was before 1.4: one
// dotF32 per output.
func matMulTransBReference(c []float32, ldc int, a, b []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			c[i*ldc+j] = dotF32(a[i*k:(i+1)*k], b[j*k:(j+1)*k])
		}
	}
}

// skipNEONGemm skips on arm64, whose NEON kernel sums in another order.
func skipNEONGemm(t *testing.T) {
	t.Helper()

	if runtime.GOARCH == "arm64" {
		t.Skip("arm64's NEON MatMulTransB is not one dotF32 per output")
	}
}

// checkMatMulTransBBits runs MatMulTransB into a sentinel-filled c with row
// stride ldc and compares every output with matMulTransBReference bit for
// bit, and every gap between the rows with the sentinel.
func checkMatMulTransBBits(t *testing.T, name string, a, b []float32, m, n, k, ldc int) {
	t.Helper()

	size := 0
	if m > 0 {
		size = (m-1)*ldc + n
	}

	got := make([]float32, size)
	want := make([]float32, size)

	for i := range got {
		got[i] = gemmSentinel
		want[i] = gemmSentinel
	}

	MatMulTransB(got, ldc, a, b, m, n, k)
	matMulTransBReference(want, ldc, a, b, m, n, k)

	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: c[%d][%d] = %v (%#08x), dotF32 gives %v (%#08x)",
				name, i/ldc, i%ldc, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// TestMatMulTransBBitIdentical sweeps k around the amd64 SIMD cutover and
// the tails of the 8- and 32-wide loops, m around groups of three rows of
// a, n around one and two panels of b rows, with ldc = n and wider, special
// values and misaligned rows, at several worker counts.
func TestMatMulTransBBitIdentical(t *testing.T) {
	skipNEONGemm(t)

	rng := rand.New(rand.NewPCG(31, 32))

	for _, k := range []int{0, 1, 7, 8, 9, 31, 32, 33, 40, 70, 1031} {
		for _, m := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17} {
			for _, n := range []int{1, 2, 3, 4, 5, 8, 9, 70, 600} {
				for _, specials := range []bool{false, true} {
					a := misaligned(rng, m*k, 1+(m+k)%7)
					b := misaligned(rng, n*k, 1+(n+k)%7)

					if specials {
						linearSpecials(a, b, nil, m, k, n)
					}

					for _, ldc := range []int{n, n + 3} {
						for _, w := range []int{1, 3, runtime.GOMAXPROCS(0)} {
							withWorkers(t, w, func() {
								name := fmt.Sprintf("m %d n %d k %d ldc %d specials %v workers %d", m, n, k, ldc, specials, w)
								checkMatMulTransBBits(t, name, a, b, m, n, k, ldc)
							})
						}
					}
				}
			}
		}
	}
}

// TestMatMulTransBWorkersBits runs products above the parallel threshold,
// split over rows of a (m >= 8) and of b, at the consumers' shapes and off
// them, and checks every worker count against one dotF32 per output.
func TestMatMulTransBWorkersBits(t *testing.T) {
	skipNEONGemm(t)

	rng := rand.New(rand.NewPCG(33, 34))

	shapes := [][3]int{ // m, n, k
		{1, 4096, 1024},
		{2, 1027, 1024},
		{5, 1024, 1031},
		{64, 301, 1024},
		{125, 512, 64},
		{17, 2305, 768},
		{512, 32, 512},
	}

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		a := misaligned(rng, m*k, 3)
		b := misaligned(rng, n*k, 5)
		linearSpecials(a, b, nil, m, k, n)

		for _, w := range []int{1, 2, 3, 7, 16, runtime.GOMAXPROCS(0)} {
			withWorkers(t, w, func() {
				checkMatMulTransBBits(t, fmt.Sprintf("m %d n %d k %d workers %d", m, n, k, w), a, b, m, n, k, n+1)
			})
		}
	}
}
