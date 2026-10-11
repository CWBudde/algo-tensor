package demo

import (
	"math"
	"testing"
)

func TestAttentionWeights(t *testing.T) {
	tests := []struct {
		n      int
		causal bool
	}{
		{1, true}, {2, true}, {8, true}, {1, false}, {8, false},
	}
	for _, tt := range tests {
		n := tt.n

		w, err := AttentionWeights(n, tt.causal)
		if err != nil {
			t.Fatal(err)
		}

		if len(w) != n*n {
			t.Fatalf("n = %d: %d weights, want %d", n, len(w), n*n)
		}

		for i := range n {
			var sum float64

			for j := range n {
				v := w[i*n+j]
				if tt.causal && j > i && v != 0 {
					t.Fatalf("n = %d: weight [%d, %d] = %v on a future key, want 0", n, i, j, v)
				}

				if !tt.causal && v <= 0 {
					t.Fatalf("n = %d, not causal: weight [%d, %d] = %v, want > 0", n, i, j, v)
				}

				sum += float64(v)
			}

			if math.Abs(sum-1) > 1e-6 {
				t.Fatalf("n = %d: row %d sums to %v, want 1", n, i, sum)
			}
		}
	}

	for _, n := range []int{0, 257} {
		_, err := AttentionWeights(n, true)
		if err == nil {
			t.Fatalf("n = %d: want an error", n)
		}
	}
}
