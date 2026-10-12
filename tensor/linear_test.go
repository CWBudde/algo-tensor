// algo-tensor's own, not in go-whistle or go-pocket-tts.

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
	"unsafe"
)

// Linear's contract is the arithmetic it always had: every output is
// dotF32(x row, weight row), then + bias in float32, on every
// configuration and for any worker count. The tests in this file compare
// Linear with that loop bit for bit (math.Float32bits), so a faster loop
// order or kernel may change the speed, never a bit.

// linearReference is Linear as it was before the blocked loop: one dotF32
// per output, plus the bias.
func linearReference(x, weight, bias *Tensor) []float32 {
	in, out := int(weight.shape[1]), int(weight.shape[0])
	batch, _ := shapeElemCount(x.shape[:x.Rank()-1])
	y := make([]float32, batch*out)

	for b := range batch {
		xRow := x.data[b*in : (b+1)*in]

		for o := range out {
			sum := dotF32(xRow, weight.data[o*in:(o+1)*in])
			if bias != nil {
				sum += bias.data[o]
			}

			y[b*out+o] = sum
		}
	}

	return y
}

// linearSweepIns covers both sides of the SIMD cutovers (8 on amd64, 48 on
// arm64), the tails of the 8- and 32-wide loops, and go-laya's widths.
var linearSweepIns = []int{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 63, 64, 65,
	255, 768, 1024, 1152, 2624,
}

// linearSweepBatches are the rows of x.
var linearSweepBatches = []int{0, 1, 2, 3, 4, 5, 7, 8, 17}

// linearSweepPanel is the number of weight rows Linear runs every token
// over at once, for in values per row.
func linearSweepPanel(in int) int {
	return min(510, max(3, (1<<16)/max(in, 1)/3*3))
}

// linearSweepOuts are 1–9 (around the multi-output kernel's group of
// three) and the outputs just below, at and above one and two panels.
func linearSweepOuts(in int) []int {
	p := linearSweepPanel(in)

	return []int{1, 2, 3, 4, 5, 6, 7, 8, 9, p - 1, p, p + 1, 2*p - 1, 2*p + 2}
}

// misaligned returns n floats from rng whose first element lies 4·shift
// bytes past a 32-byte boundary (shift 1–7), so no row starts where an
// aligned vector load could hide a wrong offset.
func misaligned(rng *rand.Rand, n, shift int) []float32 {
	buf := make([]float32, n+8)
	addr := uintptr(unsafe.Pointer(unsafe.SliceData(buf)))
	off := (shift - int(addr%32)/4 + 8) % 8
	s := buf[off : off+n]

	if a := uintptr(unsafe.Pointer(unsafe.SliceData(s))); a%32 != uintptr(4*shift) {
		panic(fmt.Sprintf("misaligned: start %#x, want %d past 32", a, 4*shift))
	}

	for i := range s {
		s[i] = rng.Float32()*2 - 1
	}

	return s
}

// linearSpecials writes NaNs with different payloads, ±Inf, subnormals and
// signed zeros into one row of x, one weight row and the bias, at the start,
// in the 32-wide part, in the 8-wide part and in the scalar tail. The other
// rows keep finite values, so they still show a changed summation order.
func linearSpecials(x, w, bias []float32, batch, in, out int) {
	nanX := math.Float32frombits(0x7fc00001)
	nanW := math.Float32frombits(0xffc00002)
	vals := []float32{
		nanX, float32(math.Inf(1)), math.Float32frombits(1), float32(math.Copysign(0, -1)),
		math.Float32frombits(0x807fffff), float32(math.Inf(-1)),
	}

	if in > 0 && batch > 0 && out > 0 {
		xr := x[min(1, batch-1)*in:][:in]
		wr := w[min(2, out-1)*in:][:in]

		for i, p := range []int{0, 33, in - 1, 40, 9, in - 3} {
			if p < 0 || p >= in {
				continue
			}

			xr[p] = vals[i%len(vals)]
			wr[p] = vals[(i+2)%len(vals)]
		}

		// NaN against NaN at one position: the result's payload shows which
		// operand the FMA took first.
		xr[in/2] = nanX
		wr[in/2] = nanW
	}

	if bias != nil {
		for i, o := range []int{0, out - 1, 4} {
			if o >= 0 && o < out {
				bias[o] = vals[(i+3)%len(vals)]
			}
		}
	}
}

// linearSweepCase builds x [batch, in] (rank 3 if rank3), weight
// [out, in] on misaligned storage and, if withBias, the bias.
func linearSweepCase(t *testing.T, rng *rand.Rand, batch, in, out int, rank3, withBias, specials bool) (x, w, bias *Tensor) {
	t.Helper()

	xs := []int64{int64(batch), int64(in)}
	if rank3 {
		xs = []int64{1, int64(batch), int64(in)}
		if batch%2 == 0 && batch > 0 {
			xs = []int64{2, int64(batch / 2), int64(in)}
		}
	}

	xd := misaligned(rng, batch*in, 1+(batch+in)%7)
	wd := misaligned(rng, out*in, 1+(out+in)%7)

	var bd []float32
	if withBias {
		bd = randGemmData(rng, out)
	}

	if specials {
		linearSpecials(xd, wd, bd, batch, in, out)
	}

	var err error

	x, err = FromOwned(xd, xs)
	if err != nil {
		t.Fatal(err)
	}

	w, err = FromOwned(wd, []int64{int64(out), int64(in)})
	if err != nil {
		t.Fatal(err)
	}

	if withBias {
		bias, err = FromOwned(bd, []int64{int64(out)})
		if err != nil {
			t.Fatal(err)
		}
	}

	return x, w, bias
}

// checkLinearBits runs Linear and compares every output with
// linearReference bit for bit.
func checkLinearBits(t *testing.T, name string, x, w, bias *Tensor) {
	t.Helper()

	got, err := Linear(x, w, bias)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}

	wantShape := x.Shape()

	wantShape[len(wantShape)-1] = w.shape[0]
	if !equalI64(got.shape, wantShape) {
		t.Fatalf("%s: shape %v, want %v", name, got.shape, wantShape)
	}

	want := linearReference(x, w, bias)
	if len(got.data) != len(want) {
		t.Fatalf("%s: %d outputs, want %d", name, len(got.data), len(want))
	}

	out := int(w.shape[0])

	for i := range want {
		if math.Float32bits(got.data[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: y[%d][%d] = %v (%#08x), dotF32 + bias gives %v (%#08x)",
				name, i/out, i%out, got.data[i], math.Float32bits(got.data[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// linearSweepWorkers are the worker counts of the sweep.
func linearSweepWorkers() []int {
	return []int{1, 2, 3, 7, runtime.GOMAXPROCS(0)}
}

// TestLinearBitIdentical sweeps widths, batches and output counts around
// every loop boundary, with and without bias, rank-2 and rank-3 x,
// special values and misaligned rows, at several worker counts: every
// output must have dotF32's bits plus the bias.
func TestLinearBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))

	variants := []struct {
		name                     string
		rank3, withBias, special bool
	}{
		{"plain", false, false, false},
		{"bias/rank3", true, true, false},
		{"bias/specials", false, true, true},
		{"specials/rank3", true, false, true},
	}

	for _, in := range linearSweepIns {
		for _, batch := range linearSweepBatches {
			for _, out := range linearSweepOuts(in) {
				for _, v := range variants {
					x, w, bias := linearSweepCase(t, rng, batch, in, out, v.rank3, v.withBias, v.special)

					for _, n := range linearSweepWorkers() {
						withWorkers(t, n, func() {
							checkLinearBits(t, fmt.Sprintf("in %d batch %d out %d %s workers %d", in, batch, out, v.name, n), x, w, bias)
						})
					}
				}
			}
		}
	}
}

// TestLinearWorkers runs products above the parallel threshold, split over
// tokens and over outputs, and checks every worker count against the
// reference, and so against each other, bit for bit.
func TestLinearWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))

	shapes := [][3]int{ // batch, in, out
		{512, 1024, 301},
		{1, 2624, 1024},
		{1, 1024, 3073},
		{5, 1152, 768},
		{17, 768, 2305},
		{300, 64, 7},
		{64, 1024, 65},
		{2, 4096, 257},
	}

	for _, s := range shapes {
		batch, in, out := s[0], s[1], s[2]
		x, w, bias := linearSweepCase(t, rng, batch, in, out, false, true, true)

		for _, n := range []int{1, 2, 3, 7, 16, runtime.GOMAXPROCS(0)} {
			withWorkers(t, n, func() {
				checkLinearBits(t, fmt.Sprintf("batch %d in %d out %d workers %d", batch, in, out, n), x, w, bias)
			})
		}
	}
}
