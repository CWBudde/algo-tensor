package demo

import (
	"slices"
	"testing"
)

func TestSignalsMovingAverage(t *testing.T) {
	tests := []struct {
		name    string
		signal  []float32
		taps    int
		want    []float32
		wantErr bool
	}{
		{name: "one tap is the identity", signal: []float32{1, -2, 3}, taps: 1, want: []float32{1, -2, 3}},
		{name: "three taps, zero padded", signal: []float32{3, 6, 9, 3}, taps: 3, want: []float32{3, 6, 6, 4}},
		{name: "even taps", signal: []float32{1, 2}, taps: 2, wantErr: true},
		{name: "longer than the signal", signal: []float32{1}, taps: 3, wantErr: true},
		{name: "empty signal", signal: []float32{}, taps: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := SignalsMovingAverage(tt.signal, tt.taps)
			if tt.wantErr {
				if err == nil {
					t.Fatal("want an error")
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(got, tt.want) {
				t.Fatalf("SignalsMovingAverage(%v, %d) = %v, want %v", tt.signal, tt.taps, got, tt.want)
			}
		})
	}
}
