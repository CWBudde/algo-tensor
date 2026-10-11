package demo

import (
	"bytes"
	"math"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

func TestRGBAToNHWC(t *testing.T) {
	tests := []struct {
		name          string
		pix           []byte
		width, height int
		want          []float32
		wantErr       bool
	}{
		{
			name:  "2×1",
			pix:   []byte{0, 51, 255, 7, 255, 0, 102, 255},
			width: 2, height: 1,
			want: []float32{0, 0.2, 1, 1, 0, 0.4},
		},
		{name: "short", pix: []byte{0, 0, 0}, width: 1, height: 1, wantErr: true},
		{name: "zero width", pix: []byte{}, width: 0, height: 1, wantErr: true},
		{name: "overflowing size", pix: []byte{}, width: 1 << 31, height: 1 << 31, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RGBAToNHWC(tt.pix, tt.width, tt.height)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("want an error, got shape %v", got.Shape())
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if want := []int64{1, int64(tt.height), int64(tt.width), 3}; !slices.Equal(got.Shape(), want) {
				t.Fatalf("shape %v, want %v", got.Shape(), want)
			}

			if !slices.Equal(got.RawData(), tt.want) {
				t.Fatalf("data %v, want %v", got.RawData(), tt.want)
			}
		})
	}
}

func TestNHWCToRGBA(t *testing.T) {
	tests := []struct {
		name    string
		data    []float32
		shape   []int64
		want    []byte
		wantErr bool
	}{
		{
			name:  "rgb, clamped and rounded",
			data:  []float32{0, 0.5, 1, -1, 2, float32(math.NaN())},
			shape: []int64{1, 1, 2, 3},
			want:  []byte{0, 128, 255, 255, 0, 255, 0, 255},
		},
		{
			name:  "gray",
			data:  []float32{0.2, 1},
			shape: []int64{1, 2, 1, 1},
			want:  []byte{51, 51, 51, 255, 255, 255, 255, 255},
		},
		{name: "two channels", data: []float32{0, 0}, shape: []int64{1, 1, 1, 2}, wantErr: true},
		{name: "batch of two", data: []float32{0, 0}, shape: []int64{2, 1, 1, 1}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in, err := tensor.New(tt.data, tt.shape)
			if err != nil {
				t.Fatal(err)
			}

			got, w, h, err := NHWCToRGBA(in)
			if tt.wantErr {
				if err == nil {
					t.Fatal("want an error")
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if w != int(tt.shape[2]) || h != int(tt.shape[1]) {
				t.Fatalf("size %d×%d, want %d×%d", w, h, tt.shape[2], tt.shape[1])
			}

			if !bytes.Equal(got, tt.want) {
				t.Fatalf("pixels %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRGBARoundTrip(t *testing.T) {
	pix := make([]byte, 4*3*2)
	for i := range pix {
		pix[i] = byte(i * 11)
		if i%4 == 3 {
			pix[i] = 255
		}
	}

	img, err := RGBAToNHWC(pix, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	back, w, h, err := NHWCToRGBA(img)
	if err != nil {
		t.Fatal(err)
	}

	if w != 3 || h != 2 || !bytes.Equal(back, pix) {
		t.Fatalf("round trip %d×%d %v, want 3×2 %v", w, h, back, pix)
	}
}
