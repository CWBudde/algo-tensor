package ops

import "testing"

// pocketGoldenCases are the operators only go-pocket-tts has.
func pocketGoldenCases() []goldenCase {
	return []goldenCase{
		{"AttentionWithPositions", func(tb testing.TB) []float32 {
			r := &goldenRand{31}
			must := goldenMust(tb)
			q, k, v := r.tensor(tb, 2, 2, 4, 8), r.tensor(tb, 2, 2, 6, 8), r.tensor(tb, 2, 2, 6, 8)
			posQ := []int64{5, 6, 7, 8}
			// Strictly increasing keys (the key-window path), then a ring
			// buffer and padding (the masking path).
			out := must(AttentionWithPositions(q, k, v, posQ, []int64{3, 4, 5, 6, 7, 8}, 4))
			out = append(out, must(AttentionWithPositions(q, k, v, posQ, []int64{6, 7, 8, 3, 4, 5}, AttentionNoContext))...)

			return append(out, must(AttentionWithPositions(q, k, v, posQ, []int64{-1, -1, 3, 4, 5, 6}, 3))...)
		}},
		{"ConvTranspose1D", func(tb testing.TB) []float32 {
			r := &goldenRand{32}
			must := goldenMust(tb)
			x, w, b := r.tensor(tb, 1, 6, 10), r.tensor(tb, 6, 4, 3), r.tensor(tb, 4)
			out := must(ConvTranspose1D(x, w, b, 2, 1, 1, 1, 1))
			out = append(out, must(ConvTranspose1D(x, r.tensor(tb, 6, 2, 3), b, 2, 1, 1, 2, 2))...)
			out = append(out, must(ConvTranspose1DRightTrim(x, w, b, 2, 1, 1, 1, 1, 2))...)
			// A larger one through the repacked-kernel GEMM.
			x2, w2 := r.tensor(tb, 2, 64, 40), r.tensor(tb, 64, 32, 4)

			return append(out, must(ConvTranspose1D(x2, w2, r.tensor(tb, 32), 2, 1, 0, 1, 1))...)
		}},
		{"ConvTranspose1DPrePacked", func(tb testing.TB) []float32 {
			r := &goldenRand{33}
			must := goldenMust(tb)
			x, w, b := r.tensor(tb, 2, 64, 40), r.tensor(tb, 64, 32, 4), r.tensor(tb, 32)
			kernelT := RepackConvTransposeKernel(w)
			out := append([]float32(nil), kernelT...)
			out = append(out, must(ConvTranspose1DPrePacked(x, w, b, kernelT, 2, 1, 0, 1, 1))...)

			return append(out, must(ConvTranspose1DPrePackedRightTrim(x, w, b, kernelT, 2, 1, 0, 1, 1, 2))...)
		}},
		{"MLP", func(tb testing.TB) []float32 {
			r := &goldenRand{34}

			return goldenMust(tb)(MLP(r.tensor(tb, 2, 3, 16), r.tensor(tb, 32, 16), r.tensor(tb, 32), r.tensor(tb, 16, 32), nil))
		}},
	}
}
