package tensor

import "testing"

// pocketGoldenCases are the kernels only go-pocket-tts has.
func pocketGoldenCases() []goldenCase {
	return []goldenCase{
		{"LayerNorm", func(tb testing.TB) []float32 {
			r := &goldenRand{7}
			x := r.tensor(tb, 3, 5, 64)
			out := goldenMust(tb)(LayerNorm(x, r.tensor(tb, 64), r.tensor(tb, 64), 1e-5))

			return append(out, goldenMust(tb)(LayerNorm(x, nil, nil, 1e-6))...)
		}},
	}
}
