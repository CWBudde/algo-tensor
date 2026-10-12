// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of the panel
// layout of the SME2 kernel's prepacked operands (PLAN.md 2.2).

package sme

import (
	"fmt"
	"math"
	"testing"
)

// packGuard is the value after PackedLen in the destination of Pack, which
// Pack must not touch: a NaN with a payload, compared by its bits.
var packGuard = math.Float32frombits(0x7fc0dead)

func TestSteps(t *testing.T) {
	t.Parallel()

	for _, c := range []struct{ k, want int }{
		{4, 4}, {5, 8}, {6, 8}, {7, 8}, {8, 8}, {9, 12}, {12, 12}, {13, 16}, {16, 16}, {297, 300}, {936, 936},
	} {
		if got := Steps(c.k); got != c.want {
			t.Errorf("Steps(%d) = %d, want %d", c.k, got, c.want)
		}
	}
}

func TestPackedLen(t *testing.T) {
	t.Parallel()

	for _, c := range []struct{ rows, k, want int }{
		{0, 4, 0},
		{1, 4, 64},
		{16, 4, 64},
		{17, 4, 128},
		{1, 5, 128},
		{17, 5, 256},
		{3, 13, 256},
		{32, 8, 256},
		{33, 9, 576},
		{36, 297, 3 * 16 * 300},
	} {
		if got := PackedLen(c.rows, c.k); got != c.want {
			t.Errorf("PackedLen(%d, %d) = %d, want %d", c.rows, c.k, got, c.want)
		}
	}
}

// TestStep checks the step of every column for every k%4 tail: the
// columns before full = 4⌊k/4⌋ keep their index, the k%4 columns after it
// go to the last steps of the tail group.
func TestStep(t *testing.T) {
	t.Parallel()

	for _, c := range []struct {
		k     int
		steps []int
	}{
		{4, []int{0, 1, 2, 3}},
		{5, []int{0, 1, 2, 3, 7}},
		{6, []int{0, 1, 2, 3, 6, 7}},
		{7, []int{0, 1, 2, 3, 5, 6, 7}},
		{8, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{9, []int{0, 1, 2, 3, 4, 5, 6, 7, 11}},
		{10, []int{0, 1, 2, 3, 4, 5, 6, 7, 10, 11}},
		{11, []int{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11}},
		{13, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15}},
	} {
		for col, want := range c.steps {
			if got := Step(col, c.k); got != want {
				t.Errorf("Step(%d, %d) = %d, want %d", col, c.k, got, want)
			}
		}
	}
}

// packSource returns rows × k distinct nonzero values, row-major, with -0
// in place of every 7th: Pack must copy the sign of a zero.
func packSource(rows, k int) []float32 {
	src := make([]float32, rows*k)
	for i := range src {
		src[i] = float32(i + 1)
		if i%7 == 6 {
			src[i] = float32(math.Copysign(0, -1))
		}
	}

	return src
}

// packDst returns n NaNs, the garbage Pack must overwrite, and a guard.
func packDst(n int) []float32 {
	dst := make([]float32, n+1)
	for i := range dst {
		dst[i] = float32(math.NaN())
	}

	dst[n] = packGuard

	return dst
}

// packWant returns the layout of PLAN.md Phase 2, written out
// independently of Step: lane l of step s of panel p is the value of row
// 16p+l at column s for s < full = 4⌊k/4⌋; step full+t of the tail group
// holds column k-4+t for t >= 4-k%4 and +0 below; lanes past the last row
// are +0.
func packWant(src []float32, rows, k int) []float32 {
	full, kp := k/4*4, (k+3)/4*4
	panels := (rows + 15) / 16
	want := make([]float32, panels*16*kp)

	for p := range panels {
		for s := range kp {
			for l := range 16 {
				r := 16*p + l
				if r >= rows {
					continue
				}

				col := s
				if s >= full {
					col = k - 4 + (s - full)
					if s-full < 4-k%4 {
						continue
					}
				}

				want[(p*kp+s)*16+l] = src[r*k+col]
			}
		}
	}

	return want
}

// checkPack fails t unless dst holds want bit for bit, followed by the
// untouched guard.
func checkPack(t *testing.T, dst, want []float32, rows, k int) {
	t.Helper()

	kp := Steps(k)

	for i, w := range want {
		if math.Float32bits(dst[i]) == math.Float32bits(w) {
			continue
		}

		p, s, l := i/(16*kp), i/16%kp, i%16
		what := "a value"

		switch {
		case 16*p+l >= rows:
			what = "a lane past the last row, +0"
		case math.Float32bits(w) == 0:
			what = "+0 (a tail step no column fills)"
		}

		t.Fatalf("rows=%d k=%d: dst[%d] (panel %d, step %d, lane %d) = %v (%#08x), want %s %v (%#08x)",
			rows, k, i, p, s, l, dst[i], math.Float32bits(dst[i]), what, w, math.Float32bits(w))
	}

	if g := dst[len(want)]; math.Float32bits(g) != math.Float32bits(packGuard) {
		t.Fatalf("rows=%d k=%d: Pack wrote %v (%#08x) past PackedLen", rows, k, g, math.Float32bits(g))
	}
}

// TestPackLiteral checks Pack on single rows, and one of three rows, of
// values 1..k against layouts written out by hand: every value not listed
// is +0.
func TestPackLiteral(t *testing.T) {
	t.Parallel()

	for _, c := range []struct {
		rows, k int
		want    map[int]float32 // index (step·16 + lane) → value
	}{
		{1, 4, map[int]float32{0: 1, 16: 2, 32: 3, 48: 4}},
		{1, 5, map[int]float32{0: 1, 16: 2, 32: 3, 48: 4, 7 * 16: 5}},
		{1, 6, map[int]float32{0: 1, 16: 2, 32: 3, 48: 4, 6 * 16: 5, 7 * 16: 6}},
		{1, 7, map[int]float32{0: 1, 16: 2, 32: 3, 48: 4, 5 * 16: 5, 6 * 16: 6, 7 * 16: 7}},
		{1, 9, map[int]float32{0: 1, 16: 2, 32: 3, 48: 4, 4 * 16: 5, 5 * 16: 6, 6 * 16: 7, 7 * 16: 8, 11 * 16: 9}},
		{3, 6, map[int]float32{
			0: 1, 16: 2, 32: 3, 48: 4, 6 * 16: 5, 7 * 16: 6,
			1: 7, 17: 8, 33: 9, 49: 10, 6*16 + 1: 11, 7*16 + 1: 12,
			2: 13, 18: 14, 34: 15, 50: 16, 6*16 + 2: 17, 7*16 + 2: 18,
		}},
	} {
		src := make([]float32, c.rows*c.k)
		for i := range src {
			src[i] = float32(i + 1)
		}

		want := make([]float32, PackedLen(c.rows, c.k))
		for i, v := range c.want {
			want[i] = v
		}

		dst := packDst(len(want))
		Pack(dst, src, c.rows, c.k)
		checkPack(t, dst, want, c.rows, c.k)
	}
}

// TestPack checks Pack against packWant for every k%4 tail and for rows
// that fill, or leave lanes of, one to three panels, into a destination
// full of NaNs with a guard after PackedLen.
func TestPack(t *testing.T) {
	t.Parallel()

	for _, k := range []int{4, 5, 6, 7, 8, 9, 10, 11, 13, 297} {
		for _, rows := range []int{1, 2, 3, 15, 16, 17, 33} {
			t.Run(fmt.Sprintf("rows%d/k%d", rows, k), func(t *testing.T) {
				t.Parallel()

				src := packSource(rows, k)
				want := packWant(src, rows, k)

				if len(want) != PackedLen(rows, k) {
					t.Fatalf("PackedLen(%d, %d) = %d, want %d", rows, k, PackedLen(rows, k), len(want))
				}

				dst := packDst(len(want))
				Pack(dst, src, rows, k)
				checkPack(t, dst, want, rows, k)

				// Packing again over the result changes nothing.
				Pack(dst, src, rows, k)
				checkPack(t, dst, want, rows, k)
			})
		}
	}
}
