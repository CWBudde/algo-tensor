package demo

import (
	"bytes"
	"math"
	"slices"
	"testing"
)

func TestFloat32sBytesRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values []float32
		bytes  []byte
	}{
		{"empty", []float32{}, []byte{}},
		{"one", []float32{1}, []byte{0x00, 0x00, 0x80, 0x3f}},
		{"negative zero and two", []float32{float32(math.Copysign(0, -1)), 2}, []byte{0, 0, 0, 0x80, 0, 0, 0, 0x40}},
		{"infinity", []float32{float32(math.Inf(-1))}, []byte{0x00, 0x00, 0x80, 0xff}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BytesFromFloat32s(tt.values)
			if !bytes.Equal(got, tt.bytes) {
				t.Fatalf("BytesFromFloat32s(%v) = %x, want %x", tt.values, got, tt.bytes)
			}

			back, err := Float32sFromBytes(got)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.EqualFunc(back, tt.values, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
				t.Fatalf("Float32sFromBytes(%x) = %v, want %v", got, back, tt.values)
			}
		})
	}
}

func TestFloat32sFromBytesPartialValue(t *testing.T) {
	_, err := Float32sFromBytes([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("Float32sFromBytes of 3 bytes: want an error")
	}
}

func TestExactInt(t *testing.T) {
	tests := []struct {
		in   float64
		want int
		ok   bool
	}{
		{0, 0, true},
		{-7, -7, true},
		{1 << 53, 0, false},
		{1<<53 - 1, 1<<53 - 1, true},
		{2.5, 0, false},
		{math.NaN(), 0, false},
		{math.Inf(1), 0, false},
	}
	for _, tt := range tests {
		got, ok := ExactInt(tt.in)
		if got != tt.want || ok != tt.ok {
			t.Errorf("ExactInt(%v) = %d, %v; want %d, %v", tt.in, got, ok, tt.want, tt.ok)
		}
	}
}
