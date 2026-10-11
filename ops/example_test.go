// From github.com/cwbudde/algo-transcribe nn/example_test.go @ 21ad164.
// algo-tensor change: only ExampleNewConv2D, ExampleConv2D_ForwardRows and
// ExampleConv2D_MACsPerRow, as ExampleNewConv2DPlan, ExampleConv2DPlan_ForwardRows and
// ExampleConv2DPlan_MACsPerRow on tensor.Tensor images with SamePadding.

package ops_test

import (
	"fmt"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)

// samePad returns TensorFlow "same" padding for an inH × inW input and a
// kh × kw kernel at stride 1.
func samePad(inH, inW, kh, kw int) ops.Conv2DParams {
	top, bottom := ops.SamePadding(inH, kh, 1)
	left, right := ops.SamePadding(inW, kw, 1)

	return ops.Conv2DParams{StrideH: 1, StrideW: 1, PadTop: top, PadBottom: bottom, PadLeft: left, PadRight: right, Groups: 1}
}

func ExampleNewConv2DPlan() {
	// A 3×3 box filter of ones over a 3×3 single-channel input. "same"
	// padding keeps the output 3×3; corners see 4 inputs, edges 6, the
	// centre 9.
	kernel, err := tensor.Full([]int64{3, 3, 1, 1}, 1) // HWIO
	if err != nil {
		fmt.Println(err)

		return
	}

	plan, err := ops.NewConv2DPlan(kernel, nil, 3, 3, samePad(3, 3, 3, 3))
	if err != nil {
		fmt.Println(err)

		return
	}

	x, err := tensor.Full([]int64{1, 3, 3, 1}, 1) // NHWC, one image
	if err != nil {
		fmt.Println(err)

		return
	}

	h, w, c := plan.OutShape()

	dst, err := tensor.Zeros([]int64{1, int64(h), int64(w), int64(c)})
	if err != nil {
		fmt.Println(err)

		return
	}

	err = plan.Forward(dst, x)
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(h, w, c)

	out := dst.RawData()
	for y := range h {
		fmt.Println(out[y*w*c : (y+1)*w*c])
	}
	// Output:
	// 3 3 1
	// [4 6 4]
	// [6 9 6]
	// [4 6 4]
}

func ExampleConv2DPlan_ForwardRows() {
	// Sum over three neighbouring rows, computed two output rows per step
	// as a budgeted caller would.
	kernel, err := tensor.Full([]int64{3, 1, 1, 1}, 1)
	if err != nil {
		fmt.Println(err)

		return
	}

	plan, err := ops.NewConv2DPlan(kernel, nil, 5, 2, samePad(5, 2, 3, 1))
	if err != nil {
		fmt.Println(err)

		return
	}

	data := make([]float32, 5*2)
	for i := range data {
		data[i] = float32(i)
	}

	x, err := tensor.New(data, []int64{1, 5, 2, 1})
	if err != nil {
		fmt.Println(err)

		return
	}

	dst, err := tensor.Zeros([]int64{1, 5, 2, 1})
	if err != nil {
		fmt.Println(err)

		return
	}

	const step = 2

	h, w, c := plan.OutShape()
	for r0 := 0; r0 < h; r0 += step {
		r1 := min(r0+step, h)

		err := plan.ForwardRows(dst, x, r0, r1)
		if err != nil {
			fmt.Println(err)

			return
		}

		fmt.Printf("rows [%d,%d): %.4g\n", r0, r1, dst.RawData()[r0*w*c:r1*w*c])
	}
	// Output:
	// rows [0,2): [2 4 6 9]
	// rows [2,4): [12 15 18 21]
	// rows [4,5): [14 16]
}

func ExampleConv2DPlan_MACsPerRow() {
	// conv2d_1 of basic-pitch: a [1, 172, 264, 8] input, a 3×39 kernel and
	// 8 filters.
	kernel, err := tensor.Zeros([]int64{3, 39, 8, 8})
	if err != nil {
		fmt.Println(err)

		return
	}

	plan, err := ops.NewConv2DPlan(kernel, nil, 172, 264, samePad(172, 264, 3, 39))
	if err != nil {
		fmt.Println(err)

		return
	}

	h, _, _ := plan.OutShape()
	fmt.Println(plan.MACsPerRow(), plan.MACsPerRow()*h)
	// Output:
	// 1976832 340015104
}
