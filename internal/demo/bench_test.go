package demo

import "testing"

func TestBenchDotProduct(t *testing.T) {
	tests := []struct {
		n, reps  int
		checksum float32
		wantErr  bool
	}{
		// a = i%7 − 3, b = i%5 − 2: a·b over i = 0…34 is 0 (35 = lcm(7, 5)).
		{n: 35, reps: 3, checksum: 0},
		// i = 0, 1: (−3)(−2) + (−2)(−1) = 8 per call.
		{n: 2, reps: 4, checksum: 32},
		{n: 0, reps: 1, wantErr: true},
		{n: 1, reps: 0, wantErr: true},
		{n: 1 << 25, reps: 1, wantErr: true},
		{n: 1 << 20, reps: 1 << 11, wantErr: true},
	}
	for _, tt := range tests {
		r, err := BenchDotProduct(tt.n, tt.reps)
		if tt.wantErr {
			if err == nil {
				t.Fatalf("n = %d, reps = %d: want an error", tt.n, tt.reps)
			}

			continue
		}

		if err != nil {
			t.Fatal(err)
		}

		if r.N != tt.n || r.Reps != tt.reps || r.Checksum != tt.checksum || r.Seconds < 0 || r.GFLOPS < 0 {
			t.Fatalf("n = %d, reps = %d: %+v, want checksum %v", tt.n, tt.reps, r, tt.checksum)
		}
	}
}
