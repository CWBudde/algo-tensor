package tensor

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"testing"
)

// The golden cases run every numeric kernel of the package on fixed inputs.
// golden_test.go compares the FNV-1a hash of each output with the hash the
// same case gave in the package the kernel was taken from, so a kernel here
// is bit-identical to its source (AGENTS.md, "Numerics"). This file and
// golden_config_test.go are also the generator: copied into go-whistle
// internal/tensor @ 8926ee3 and go-pocket-tts internal/runtime/tensor @
// 9faba38, they produced the constants of golden_test.go (see the comment
// there).

// goldenRand is a splitmix64 stream. Its floats are the same on every
// platform and Go release, which math/rand/v2 does not promise for its
// float methods.
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

func (r *goldenRand) tensor(tb testing.TB, shape ...int64) *Tensor {
	tb.Helper()

	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	t, err := New(r.floats(int(n)), shape)
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

// goldenMust returns a function that unwraps a kernel's (tensor, error)
// result into its data.
func goldenMust(tb testing.TB) func(*Tensor, error) []float32 {
	return func(t *Tensor, err error) []float32 {
		tb.Helper()

		if err != nil {
			tb.Fatal(err)
		}

		return t.Data()
	}
}

// goldenLengths covers both sides of every SIMD cutover and loop tail.
var goldenLengths = []int{1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 255, 256, 257, 1023, 1024, 4099}

// goldenGEMMShapes are m, n, k. The last two are above MatMulTransB's
// parallel threshold, split over the rows of a and of b.
var goldenGEMMShapes = [][3]int{
	{1, 1, 1},
	{3, 5, 7},
	{4, 4, 4},
	{5, 9, 16},
	{8, 12, 33},
	{17, 9, 64},
	{64, 96, 128},
	{130, 70, 257},
	{6, 300, 700},
}

// commonGoldenCases are the kernels go-whistle and go-pocket-tts both have.
func commonGoldenCases() []goldenCase {
	cases := make([]goldenCase, 0, 6+len(goldenGEMMShapes))
	cases = append(cases, []goldenCase{
		{"DotProduct", func(testing.TB) []float32 {
			r := &goldenRand{1}

			out := make([]float32, 0, len(goldenLengths))

			for _, n := range goldenLengths {
				out = append(out, DotProduct(r.floats(n), r.floats(n)))
			}

			return out
		}},
		{"Axpy", func(testing.TB) []float32 {
			r := &goldenRand{2}

			var out []float32

			for _, n := range goldenLengths {
				dst := r.floats(n)
				Axpy(dst, 0.37, r.floats(n))
				out = append(out, dst...)
			}

			return out
		}},
		{"Linear", func(tb testing.TB) []float32 {
			r := &goldenRand{3}
			out := goldenMust(tb)(Linear(r.tensor(tb, 2, 5, 33), r.tensor(tb, 19, 33), r.tensor(tb, 19)))

			return append(out, goldenMust(tb)(Linear(r.tensor(tb, 1, 40, 128), r.tensor(tb, 96, 128), nil))...)
		}},
		{"MatMul", func(tb testing.TB) []float32 {
			r := &goldenRand{4}
			out := goldenMust(tb)(MatMul(r.tensor(tb, 2, 3, 7, 9), r.tensor(tb, 3, 9, 11)))

			return append(out, goldenMust(tb)(MatMul(r.tensor(tb, 1, 16, 64), r.tensor(tb, 1, 64, 64)))...)
		}},
		{"Softmax", func(tb testing.TB) []float32 {
			r := &goldenRand{5}
			x := r.tensor(tb, 3, 7, 13)
			out := goldenMust(tb)(Softmax(x, -1))

			return append(out, goldenMust(tb)(Softmax(x, 1))...)
		}},
		{"Broadcast", func(tb testing.TB) []float32 {
			r := &goldenRand{6}
			a, b := r.tensor(tb, 2, 3, 4), r.tensor(tb, 3, 1)
			out := goldenMust(tb)(BroadcastAdd(a, b))

			return append(out, goldenMust(tb)(BroadcastMul(a, b))...)
		}},
	}...)

	for i, s := range goldenGEMMShapes {
		m, n, k := s[0], s[1], s[2]
		cases = append(cases, goldenCase{fmt.Sprintf("MatMulTransB/%dx%dx%d", m, n, k), func(testing.TB) []float32 {
			r := &goldenRand{uint64(100 + i)}
			ldc := n + 3
			c := r.floats((m-1)*ldc + n) // the entries between rows must stay untouched
			MatMulTransB(c, ldc, r.floats(m*k), r.floats(n*k), m, n, k)

			return c
		}})
	}

	return cases
}
