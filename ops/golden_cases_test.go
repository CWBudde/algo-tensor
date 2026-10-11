package ops

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// The golden cases run every operator of the package on fixed inputs; see
// tensor/golden_cases_test.go, whose helpers these repeat. Copied with the
// helpers into go-whistle internal/ops @ 8926ee3 (with
// golden_cases_whistle_test.go) and go-pocket-tts internal/runtime/ops @
// 9faba38 (with golden_cases_pocket_test.go), they produced the constants of
// golden_test.go.

// goldenRand is a splitmix64 stream, the same on every platform and Go
// release.
type goldenRand struct{ s uint64 }

func (r *goldenRand) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb

	return z ^ (z >> 31)
}

// floats returns n values in [-1, 1).
func (r *goldenRand) floats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(int32(r.next()>>32)) / (1 << 31)
	}

	return out
}

func (r *goldenRand) tensor(tb testing.TB, shape ...int64) *tensor.Tensor {
	tb.Helper()

	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	t, err := tensor.New(r.floats(int(n)), shape)
	if err != nil {
		tb.Fatalf("golden tensor %v: %v", shape, err)
	}

	return t
}

// goldenHash is the FNV-1a hash of the values' bit patterns, little endian.
func goldenHash(v []float32) string {
	h := fnv.New64a()

	var b [4]byte
	for _, x := range v {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
		_, _ = h.Write(b[:])
	}

	return fmt.Sprintf("%016x", h.Sum64())
}

type goldenCase struct {
	name string
	run  func(tb testing.TB) []float32
}

// goldenMust returns a function that unwraps an operator's (tensor, error)
// result into its data.
func goldenMust(tb testing.TB) func(*tensor.Tensor, error) []float32 {
	return func(t *tensor.Tensor, err error) []float32 {
		tb.Helper()

		if err != nil {
			tb.Fatal(err)
		}

		return t.Data()
	}
}

// commonGoldenCases are the operators go-whistle and go-pocket-tts both have.
func commonGoldenCases() []goldenCase {
	return []goldenCase{
		{"Attention", func(tb testing.TB) []float32 {
			r := &goldenRand{11}
			must := goldenMust(tb)
			q, k, v := r.tensor(tb, 2, 3, 5, 16), r.tensor(tb, 2, 3, 9, 16), r.tensor(tb, 2, 3, 9, 12)
			out := must(Attention(q, k, v, false, 0))
			out = append(out, must(Attention(q, k, v, true, 4))...)
			q2, k2, v2 := r.tensor(tb, 1, 8, 40, 48), r.tensor(tb, 1, 8, 40, 48), r.tensor(tb, 1, 8, 40, 64)

			return append(out, must(Attention(q2, k2, v2, true, 0))...)
		}},
		{"CausalMask", func(tb testing.TB) []float32 {
			r := &goldenRand{12}

			return goldenMust(tb)(CausalMask(r.tensor(tb, 1, 2, 5, 8), 3))
		}},
		{"RoPE", func(tb testing.TB) []float32 {
			r := &goldenRand{13}

			return goldenMust(tb)(RoPE(r.tensor(tb, 1, 2, 6, 16), r.tensor(tb, 12, 8), r.tensor(tb, 12, 8), 3))
		}},
		{"Conv1D", func(tb testing.TB) []float32 {
			r := &goldenRand{14}
			must := goldenMust(tb)
			// groups=1 through the tiled im2col GEMM, then stride 2.
			out := must(Conv1D(r.tensor(tb, 1, 64, 300), r.tensor(tb, 96, 64, 3), r.tensor(tb, 96), 1, 1, 1, 1))
			out = append(out, must(Conv1D(r.tensor(tb, 2, 8, 31), r.tensor(tb, 12, 8, 4), nil, 2, 1, 1, 1))...)
			// grouped with dilation, then depthwise.
			out = append(out, must(Conv1D(r.tensor(tb, 2, 8, 50), r.tensor(tb, 8, 2, 5), r.tensor(tb, 8), 2, 3, 2, 4))...)

			return append(out, must(Conv1D(r.tensor(tb, 1, 16, 40), r.tensor(tb, 16, 1, 9), r.tensor(tb, 16), 1, 4, 1, 16))...)
		}},
		{"Conv1DLeftPad", func(tb testing.TB) []float32 {
			r := &goldenRand{15}

			return goldenMust(tb)(Conv1DLeftPad(r.tensor(tb, 1, 8, 30), r.tensor(tb, 12, 8, 4), r.tensor(tb, 12), 1, 3, 1, 1))
		}},
	}
}
