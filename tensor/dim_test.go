// algo-tensor's own, not in go-whistle or go-pocket-tts.

package tensor

import "testing"

// TestDim checks that Dim(i) is Shape()[i] on every rank, the scalar
// included, without allocating, and that it panics outside [0, Rank()).
func TestDim(t *testing.T) {
	for _, shape := range [][]int64{{}, {5}, {0}, {2, 3}, {3, 0, 4}, {2, 3, 4, 5}} {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}

		x, err := New(make([]float32, n), shape)
		if err != nil {
			t.Fatalf("New(%v): %v", shape, err)
		}

		want := x.Shape()
		if x.Rank() != len(want) {
			t.Fatalf("%v: Rank() = %d, want %d", shape, x.Rank(), len(want))
		}

		for i, d := range want {
			if got := x.Dim(i); got != d {
				t.Fatalf("%v: Dim(%d) = %d, want %d", shape, i, got, d)
			}
		}

		if x.Rank() > 0 {
			if allocs := testing.AllocsPerRun(10, func() { _ = x.Dim(x.Rank() - 1) }); allocs != 0 {
				t.Fatalf("%v: Dim allocates %v times", shape, allocs)
			}
		}

		for _, i := range []int{-1, x.Rank()} {
			func() {
				defer func() {
					if recover() == nil {
						t.Fatalf("%v: Dim(%d) did not panic", shape, i)
					}
				}()

				_ = x.Dim(i)
			}()
		}
	}
}
