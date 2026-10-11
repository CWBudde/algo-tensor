package ops

import "testing"

// whistleGoldenCases are the operators only go-whistle has.
func whistleGoldenCases() []goldenCase {
	return []goldenCase{
		{"Conv2D", func(tb testing.TB) []float32 {
			r := &goldenRand{21}
			must := goldenMust(tb)
			x := r.tensor(tb, 2, 9, 8, 4)
			out := must(Conv2D(x, r.tensor(tb, 3, 3, 4, 6), r.tensor(tb, 6), Conv2DParams{1, 1, 1, 1, 1, 1, 1}))
			out = append(out, must(Conv2D(x, r.tensor(tb, 3, 3, 2, 6), nil, Conv2DParams{1, 2, 0, 1, 2, 0, 2}))...)
			top, bottom := SamePadding(9, 3, 2)
			left, right := SamePadding(8, 3, 2)

			return append(out, must(Conv2D(x, r.tensor(tb, 3, 3, 1, 8), r.tensor(tb, 8), Conv2DParams{2, 2, top, bottom, left, right, 4}))...)
		}},
	}
}
