package demo

import (
	"bytes"
	"testing"
)

func TestConv2DFilter(t *testing.T) {
	// A 3×3 black image with one white pixel in the middle.
	pix := make([]byte, 4*9)
	for i := 3; i < len(pix); i += 4 {
		pix[i] = 255
	}

	pix[16], pix[17], pix[18] = 255, 255, 255

	tests := []struct {
		name          string
		filter        string
		width, height int
		want          []byte
		wantErr       bool
	}{
		{name: "identity", filter: "identity", width: 3, height: 3, want: pix},
		// The blur spreads the white pixel to 255/9 ≈ 28 on every pixel.
		{name: "box", filter: "box", width: 3, height: 3, want: bytes.Repeat([]byte{28, 28, 28, 255}, 9)},
		{name: "unknown filter", filter: "sharpen", width: 3, height: 3, wantErr: true},
		{name: "wrong size", filter: "box", width: 2, height: 3, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Conv2DFilter(pix, tt.width, tt.height, tt.filter)
			if tt.wantErr {
				if err == nil {
					t.Fatal("want an error")
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if !bytes.Equal(got, tt.want) {
				t.Fatalf("Conv2DFilter = %v, want %v", got, tt.want)
			}
		})
	}
}
