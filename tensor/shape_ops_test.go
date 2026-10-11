// From github.com/cwbudde/go-whistle internal/tensor/shape_ops_test.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/tensor/shape_ops_test.go @ 9faba38.
// algo-tensor change: TestNarrowRejectsOverflowingRange, from github.com/MeKo-Christian/go-laya
// internal/runtime/tensor/robustness_test.go @ 3ad632e.

package tensor

import (
	"math"
	"testing"
)

func TestNarrow(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})

	out, err := x.Narrow(1, 1, 2)
	if err != nil {
		t.Fatalf("narrow: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}

	want := []float32{2, 3, 5, 6}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestGather(t *testing.T) {
	x, _ := New([]float32{10, 20, 30, 40, 50, 60}, []int64{2, 3})

	out, err := x.Gather(1, []int64{2, 0})
	if err != nil {
		t.Fatalf("gather: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}

	want := []float32{30, 10, 60, 40}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestTranspose2D(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})

	y, err := x.Transpose(0, 1)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}

	if got := y.Shape(); !equalI64(got, []int64{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got)
	}

	want := []float32{1, 4, 2, 5, 3, 6}
	if got := y.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestTranspose4DSwap12(t *testing.T) {
	// [B=1,T=2,H=3,D=2] -> [1,3,2,2]
	x, _ := New([]float32{
		1, 2, 3, 4, 5, 6, // t=0, h=0..2
		7, 8, 9, 10, 11, 12, // t=1, h=0..2
	}, []int64{1, 2, 3, 2})

	y, err := x.Transpose(1, 2)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}

	if got := y.Shape(); !equalI64(got, []int64{1, 3, 2, 2}) {
		t.Fatalf("shape = %v, want [1 3 2 2]", got)
	}

	// h-major then t-major
	want := []float32{
		1, 2, 7, 8, // h=0, t=0..1
		3, 4, 9, 10, // h=1, t=0..1
		5, 6, 11, 12, // h=2, t=0..1
	}
	if got := y.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestConcatDim1(t *testing.T) {
	a, _ := New([]float32{1, 2, 3, 4}, []int64{1, 2, 2})
	b, _ := New([]float32{5, 6, 7, 8}, []int64{1, 2, 2})

	out, err := Concat([]*Tensor{a, b}, 1)
	if err != nil {
		t.Fatalf("concat: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{1, 4, 2}) {
		t.Fatalf("shape = %v, want [1 4 2]", got)
	}

	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

// TestNarrowRejectsOverflowingRange: Narrow must bound start before it adds
// length, since start+length can wrap negative and pass a combined check.
func TestNarrowRejectsOverflowingRange(t *testing.T) {
	x, err := New([]float32{1}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct{ start, length int64 }{
		{math.MaxInt64, 2},
		{1, math.MaxInt64},
		{2, 0},
	}

	for _, tc := range cases {
		_, err := x.Narrow(0, tc.start, tc.length)
		if err == nil {
			t.Errorf("Narrow(0, %d, %d) on size 1 succeeded; want an error", tc.start, tc.length)
		}
	}

	// The full and the empty range at the end stay valid.
	for _, tc := range []struct{ start, length int64 }{{0, 1}, {1, 0}} {
		_, err := x.Narrow(0, tc.start, tc.length)
		if err != nil {
			t.Errorf("Narrow(0, %d, %d) on size 1: %v", tc.start, tc.length, err)
		}
	}
}
