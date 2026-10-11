package demo

import (
	"fmt"
	"time"

	"github.com/cwbudde/algo-tensor/tensor"
)

// BenchResult is one timing of a kernel.
type BenchResult struct {
	N        int     // vector length
	Reps     int     // calls timed
	Seconds  float64 // wall time of all calls
	GFLOPS   float64 // 2·N·Reps / Seconds / 10⁹; 0 if the clock did not advance
	Checksum float32 // sum of the results, so the calls cannot be skipped
}

// BenchDotProduct times reps calls of tensor.DotProduct on two vectors of n
// values (2·n floating-point operations each). n and reps are at least 1,
// n is at most 2²⁴ and n·reps at most 2³⁰, about a second in the browser:
// the call blocks its thread and cannot be cancelled.
//
// It is the scaffold's stub for the bench demo.
func BenchDotProduct(n, reps int) (BenchResult, error) {
	const maxN, maxWork = 1 << 24, 1 << 30

	if n < 1 || n > maxN || reps < 1 || reps > maxWork/n {
		return BenchResult{}, fmt.Errorf("demo: want 1 ≤ n ≤ %d and 1 ≤ n·reps ≤ %d, got n = %d, reps = %d",
			maxN, maxWork, n, reps)
	}

	a := make([]float32, n)
	b := make([]float32, n)

	for i := range n {
		a[i] = float32(i%7) - 3
		b[i] = float32(i%5) - 2
	}

	var sum float32

	start := time.Now()

	for range reps {
		sum += tensor.DotProduct(a, b)
	}

	seconds := time.Since(start).Seconds()

	r := BenchResult{N: n, Reps: reps, Seconds: seconds, Checksum: sum}
	if seconds > 0 {
		r.GFLOPS = 2 * float64(n) * float64(reps) / seconds / 1e9
	}

	return r, nil
}
