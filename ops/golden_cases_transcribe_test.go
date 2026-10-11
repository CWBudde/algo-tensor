package ops

import (
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// goldenPlanConfigs are Conv2DPlan cases: input h, w, c, kernel kh, kw,
// out channels, strides and whether there is a bias. They cover the NEON
// shapes of tensor.MatMulTransBBias (out channels 1 or a multiple of 4,
// k >= 4), the Go tiles (k < 4, other channel counts), output rows that
// end off the 8-position blocks, and small versions of basic-pitch's
// layers, all with "same" padding.
var goldenPlanConfigs = []struct {
	h, w, c, kh, kw, o, sh, sw int
	bias                       bool
}{
	{9, 8, 4, 3, 3, 6, 1, 1, true},
	{6, 40, 8, 3, 39, 8, 1, 1, true},
	{7, 30, 8, 5, 5, 1, 1, 1, true},
	{9, 27, 1, 7, 7, 32, 1, 3, true},
	{8, 11, 32, 7, 3, 1, 1, 1, false},
	{5, 13, 3, 1, 1, 5, 1, 1, false},
	{12, 21, 2, 3, 3, 6, 2, 2, true},
	{10, 17, 3, 4, 6, 7, 1, 3, false},
}

// transcribeGoldenCases are the operators taken from algo-transcribe nn @
// 21ad164.
func transcribeGoldenCases() []goldenCase {
	return []goldenCase{
		{"Conv2DPlan", func(tb testing.TB) []float32 {
			r := &goldenRand{31}

			var out []float32

			for _, g := range goldenPlanConfigs {
				var b *tensor.Tensor
				if g.bias {
					b = r.tensor(tb, int64(g.o))
				}

				top, bottom := SamePadding(g.h, g.kh, g.sh)
				left, right := SamePadding(g.w, g.kw, g.sw)

				plan, err := NewConv2DPlan(r.tensor(tb, int64(g.kh), int64(g.kw), int64(g.c), int64(g.o)), b, g.h, g.w,
					Conv2DParams{g.sh, g.sw, top, bottom, left, right, 1})
				if err != nil {
					tb.Fatal(err)
				}

				oh, ow, oc := plan.OutShape()

				dst, err := tensor.Zeros([]int64{1, int64(oh), int64(ow), int64(oc)})
				if err != nil {
					tb.Fatal(err)
				}

				err = plan.Forward(dst, r.tensor(tb, 1, int64(g.h), int64(g.w), int64(g.c)))
				if err != nil {
					tb.Fatal(err)
				}

				out = append(out, dst.Data()...)
			}

			return out
		}},
	}
}
