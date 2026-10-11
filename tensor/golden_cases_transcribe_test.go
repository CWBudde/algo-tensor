package tensor

import (
	"fmt"
	"testing"
)

// goldenBiasShapes are m, n, k of the MatMulTransBBias cases: the NEON
// shapes (n a multiple of 4 or 1, k >= 4; the basic-pitch GEMMs of
// ops.Conv2DPlan among them), the Go tiles (k < 4, n neither), row counts
// off the 4- and 8-row blocks, and two products above the parallel
// threshold.
var goldenBiasShapes = [][3]int{
	{1, 1, 1},
	{3, 5, 2},
	{8, 8, 936},
	{8, 1, 200},
	{8, 32, 49},
	{7, 1, 297},
	{13, 6, 7},
	{17, 12, 13},
	{9, 3, 33},
	{200, 36, 300},
	{203, 7, 299},
}

// transcribeGoldenCases are the kernels taken from algo-transcribe nn @
// 21ad164. MatMulTransBBias has no counterpart there to hash (its gemm is
// unexported and writes contiguous rows); the hashes are algo-tensor's,
// and ops.Conv2DPlan, which runs on it, was checked bit for bit against
// nn's Conv2D (see ops/golden_test.go).
func transcribeGoldenCases() []goldenCase {
	cases := make([]goldenCase, 0, 2*len(goldenBiasShapes))

	for i, s := range goldenBiasShapes {
		m, n, k := s[0], s[1], s[2]
		cases = append(cases, goldenCase{fmt.Sprintf("MatMulTransBBias/%dx%dx%d", m, n, k), func(testing.TB) []float32 {
			r := &goldenRand{uint64(200 + i)}
			ldc := n + 3
			c := r.floats((m-1)*ldc + n) // the entries between rows must stay untouched
			MatMulTransBBias(c, ldc, r.floats(m*k), r.floats(n*k), r.floats(n), m, n, k)

			return c
		}})
	}

	for i, s := range goldenBiasShapes {
		m, n, k := s[0], s[1], s[2]
		cases = append(cases, goldenCase{fmt.Sprintf("MatMulTransBBias/nobias/%dx%dx%d", m, n, k), func(testing.TB) []float32 {
			r := &goldenRand{uint64(300 + i)}
			c := r.floats(m * n)
			MatMulTransBBias(c, n, r.floats(m*k), r.floats(n*k), nil, m, n, k)

			return c
		}})
	}

	return cases
}
