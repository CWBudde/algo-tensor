// From github.com/cwbudde/algo-transcribe nn/conv.go @ 21ad164.
// algo-tensor change: Conv2D, NewConv2D, row and im2col renamed to Conv2DPlan, NewConv2DPlan,
// rows and im2col on a scratch argument; blockM is the local conv2DPlanBlockM.
// algo-tensor change: one NHWC image as a *tensor.Tensor of shape [1, H, W, C], checked through
// Rank and Dim, instead of nn.Tensor and nn.Shape; the HWIO kernel and the bias are tensors
// ([kh, kw, inC, outC] and [outC] or nil) instead of a kernel size, a channel count and slices.
// algo-tensor change: explicit padding and strides from Conv2DParams (SamePadding gives the
// TensorFlow "same" padding nn computed) instead of WithStride and samePadding; Groups must be 1.
// algo-tensor change: OutShape returns (h, w, c); no InShape, Padding, ConvOption or WithStride.
// algo-tensor change: errors are prefixed "ops: conv2d plan:" and also reject a nil kernel,
// negative padding, a padded input smaller than the kernel and a batch other than 1.
// algo-tensor change: ForwardRows may split its rows over the SetConvWorkers goroutines, each
// with its own im2col scratch; the default stays sequential and allocation-free.
// algo-tensor change: the GEMM is tensor.MatMulTransBBias with ldc = outC instead of nn's gemm.
// algo-tensor change: ForwardRows checks its arguments, then calls run, which works on raw slices.
// algo-tensor change: where the CPU has SME2 and the shape gains, the GEMM runs on tensor's SME2
// kernel with the weights packed once, in chunks of many positions (conv2d_plan_sme.go, PLAN.md 2.2).

package ops

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/cwbudde/algo-tensor/tensor"
)

// conv2DPlanBlockM is the number of output positions unrolled into the
// im2col scratch per GEMM call. It is nn's blockM and must stay equal to
// tensor's row block (gemmBiasBlockM, 8), or at least a multiple of 8:
// the GEMM tiles a call's rows from its first row on, so another block
// size would group the rows into other tiles and, for n = 1, sum them in
// another order, and the outputs would no longer equal nn's bit for bit.
const conv2DPlanBlockM = 8

// Conv2DPlan is a 2-D convolution with groups = 1 set up for one input
// size: im2col plus a blocked float32 GEMM (tensor.MatMulTransBBias), as in
// algo-transcribe's nn.Conv2D. For each block of output positions of one
// row, the receptive fields are unrolled into contiguous rows of length
// kh·kw·inC in kernel-row, kernel-column, input-channel order, and the GEMM
// takes their dot products with the weights, transposed once to
// [outC][kh·kw·inC] by NewConv2DPlan.
//
// On darwin/arm64 CPUs with SME2 (Apple M4 and later), a plan whose GEMM
// is deep and wide enough runs it on the SME2 kernel of
// tensor.MatMulTransBBias instead, with the weights packed once for it and
// the im2col rows of many output positions per call
// (conv2d_plan_sme.go). The kernel gives every output the bits of the NEON
// kernel it replaces, so the results do not depend on the CPU, the
// chunking of ForwardRows or the worker count.
//
// The plan owns the packed weights, a copy of the bias and the im2col
// scratch (up to 4 MiB per goroutine where it runs on SME2), so Forward and
// ForwardRows do not allocate (on the sequential default of
// SetConvWorkers). Its results differ from Conv2D's in the last bits,
// Conv2D summing the products in another order.
//
// A Conv2DPlan is not safe for concurrent use: calls share the scratch.
type Conv2DPlan struct {
	inH, inW, inC    int
	kh, kw           int
	strideH, strideW int
	padTop, padLeft  int
	outH, outW, outC int

	// k is the GEMM depth, kh·kw·inC.
	k int
	// wt holds the weights transposed to [outC][k], one contiguous
	// k-vector per output channel.
	wt []float32
	// bias has outC values, zeros for a nil bias.
	bias []float32
	// cols holds one im2col scratch of colLen values per goroutine of
	// ForwardRows; cols[0] is the sequential path's, the others are
	// allocated when SetConvWorkers first asks for them.
	cols [][]float32
	// colLen is conv2DPlanBlockM·k, or for the SME2 kernel the rows of
	// smeChunk positions plus the kernel's panel scratch if that is more.
	colLen int
	// pb holds the weights packed for sme.GEMM, nil where the plan runs
	// MatMulTransBBias only.
	pb []float32
	// smeChunk is the number of output positions per sme.GEMM call.
	smeChunk int
}

// NewConv2DPlan sets up the convolution of a [1, inH, inW, inC] NHWC image
// with an HWIO kernel [kh, kw, inC, outC] and an optional bias [outC], with
// the strides and zero padding of p (SamePadding gives TensorFlow's "same").
// p.Groups must be 1. The kernel and the bias are copied, so the caller may
// reuse them.
func NewConv2DPlan(kernel, bias *tensor.Tensor, inH, inW int, p Conv2DParams) (*Conv2DPlan, error) {
	if kernel == nil {
		return nil, errors.New("ops: conv2d plan: nil kernel")
	}

	err := checkConv2DPlanParams(inH, inW, p)
	if err != nil {
		return nil, err
	}

	c, err := newConv2DPlanGeometry(kernel, bias, inH, inW, p)
	if err != nil {
		return nil, err
	}

	w := kernel.RawData()
	c.wt = make([]float32, c.k*c.outC)

	for i := range c.k {
		for co := range c.outC {
			c.wt[co*c.k+i] = w[i*c.outC+co]
		}
	}

	c.bias = make([]float32, c.outC)
	if bias != nil {
		copy(c.bias, bias.RawData())
	}

	c.colLen = conv2DPlanBlockM * c.k
	c.setupSME()
	c.cols = [][]float32{make([]float32, c.colLen)}

	return c, nil
}

// checkConv2DPlanParams checks the input size and p, without the kernel.
func checkConv2DPlanParams(inH, inW int, p Conv2DParams) error {
	if p.Groups != 1 {
		return fmt.Errorf("ops: conv2d plan: groups must be 1, got %d", p.Groups)
	}

	if p.StrideH <= 0 || p.StrideW <= 0 {
		return fmt.Errorf("ops: conv2d plan: strides (%d, %d) must be > 0", p.StrideH, p.StrideW)
	}

	if p.PadTop < 0 || p.PadBottom < 0 || p.PadLeft < 0 || p.PadRight < 0 {
		return fmt.Errorf("ops: conv2d plan: padding (%d, %d, %d, %d) must be >= 0",
			p.PadTop, p.PadBottom, p.PadLeft, p.PadRight)
	}

	if inH <= 0 || inW <= 0 {
		return fmt.Errorf("ops: conv2d plan: input size %d×%d must be > 0", inH, inW)
	}

	if !paddedFits(inH, p.PadTop, p.PadBottom) || !paddedFits(inW, p.PadLeft, p.PadRight) {
		return fmt.Errorf("ops: conv2d plan: input size %d×%d or padding too large", inH, inW)
	}

	return nil
}

// newConv2DPlanGeometry checks the kernel and the bias against each other
// and the checked input size and p, and returns a plan with the sizes set
// and nothing allocated.
func newConv2DPlanGeometry(kernel, bias *tensor.Tensor, inH, inW int, p Conv2DParams) (*Conv2DPlan, error) {
	if kernel.Rank() != 4 || kernel.Dim(0) <= 0 || kernel.Dim(1) <= 0 || kernel.Dim(2) <= 0 || kernel.Dim(3) <= 0 {
		return nil, fmt.Errorf("ops: conv2d plan: kernel shape %v, want [kh, kw, inC, outC] with positive sizes", kernel.Shape())
	}

	c := &Conv2DPlan{
		inH: inH, inW: inW, inC: int(kernel.Dim(2)),
		kh: int(kernel.Dim(0)), kw: int(kernel.Dim(1)),
		strideH: p.StrideH, strideW: p.StrideW,
		padTop: p.PadTop, padLeft: p.PadLeft,
		outC: int(kernel.Dim(3)),
	}

	if bias != nil && (bias.Rank() != 1 || bias.Dim(0) != int64(c.outC)) {
		return nil, fmt.Errorf("ops: conv2d plan: bias shape %v, want [%d]", bias.Shape(), c.outC)
	}

	padH, padW := inH+p.PadTop+p.PadBottom, inW+p.PadLeft+p.PadRight
	if padH < c.kh || padW < c.kw {
		return nil, fmt.Errorf("ops: conv2d plan: padded input %d×%d is smaller than kernel %d×%d", padH, padW, c.kh, c.kw)
	}

	c.outH = (padH-c.kh)/p.StrideH + 1
	c.outW = (padW-c.kw)/p.StrideW + 1
	c.k = c.kh * c.kw * c.inC

	if !fitsInt(c.inH, c.inW, c.inC) || !fitsInt(c.outH, c.outW, c.outC) || !fitsInt(c.outW, c.k, c.outC) {
		return nil, fmt.Errorf("ops: conv2d plan: input %d×%d×%d with output %d×%d×%d is too large",
			c.inH, c.inW, c.inC, c.outH, c.outW, c.outC)
	}

	return c, nil
}

// paddedFits reports whether a size n and its paddings lo and hi are at
// most MaxInt/4, which keeps n + lo + hi from overflowing an int.
func paddedFits(n, lo, hi int) bool {
	const limit = math.MaxInt / 4

	return n <= limit && lo <= limit && hi <= limit
}

// fitsInt reports whether the product of the positive values a, b and c
// fits in an int.
func fitsInt(a, b, c int) bool {
	return a <= math.MaxInt/b && a*b <= math.MaxInt/c
}

// OutShape returns the output's height, width and channel count; the
// output tensor is [1, h, w, c].
func (c *Conv2DPlan) OutShape() (h, w, ch int) { return c.outH, c.outW, c.outC }

// MACsPerRow returns the number of multiply-accumulates for one output row,
// outW·kh·kw·inC·outC.
func (c *Conv2DPlan) MACsPerRow() int { return c.outW * c.k * c.outC }

// Forward computes all output rows. It is ForwardRows(dst, x, 0, h) with h
// from OutShape.
func (c *Conv2DPlan) Forward(dst, x *tensor.Tensor) error {
	return c.ForwardRows(dst, x, 0, c.outH)
}

// ForwardRows computes output rows [r0, r1) of dst from x. x must be
// [1, inH, inW, inC] and dst [1, h, w, c] from OutShape, exactly; dst must
// not share memory with x. The rows are independent, so a caller can split
// a full pass into steps of any size; every row costs MACsPerRow
// multiply-accumulates. Other rows of dst are left untouched. With
// SetConvWorkers(n > 1) the rows split over up to n goroutines, with the
// same bits as sequentially.
func (c *Conv2DPlan) ForwardRows(dst, x *tensor.Tensor, r0, r1 int) error {
	err := c.check(dst, x, r0, r1)
	if err != nil {
		return err
	}

	c.run(dst.RawData(), x.RawData(), r0, r1)

	return nil
}

// run computes output rows [r0, r1) into out, one whole output image
// (outH·outW·outC values), from in, one input image (inH·inW·inC values),
// split over the SetConvWorkers goroutines as ForwardRows describes. The
// caller checks the sizes and the rows.
func (c *Conv2DPlan) run(out, in []float32, r0, r1 int) {
	n := r1 - r0

	workers := min(getConvWorkers(), n)
	if workers <= 1 {
		c.rows(out, in, c.cols[0], r0, r1)

		return
	}

	for len(c.cols) < workers {
		c.cols = append(c.cols, make([]float32, c.colLen))
	}

	// Chunk i has n/workers rows, one more for i < n%workers.
	base, rem := n/workers, n%workers
	start := func(i int) int { return r0 + i*base + min(i, rem) }

	var wg sync.WaitGroup

	for i := 1; i < workers; i++ {
		wg.Add(1)

		go func(col []float32, lo, hi int) {
			defer wg.Done()

			c.rows(out, in, col, lo, hi)
		}(c.cols[i], start(i), start(i+1))
	}

	c.rows(out, in, c.cols[0], r0, start(1))
	wg.Wait()
}

// check validates the arguments of ForwardRows without allocating unless
// it reports an error.
func (c *Conv2DPlan) check(dst, x *tensor.Tensor, r0, r1 int) error {
	if x == nil || dst == nil {
		return errors.New("ops: conv2d plan: nil x or dst")
	}

	if !isImage(x, c.inH, c.inW, c.inC) {
		return fmt.Errorf("ops: conv2d plan: x shape %v, want [1 %d %d %d]", x.Shape(), c.inH, c.inW, c.inC)
	}

	if !isImage(dst, c.outH, c.outW, c.outC) {
		return fmt.Errorf("ops: conv2d plan: dst shape %v, want [1 %d %d %d]", dst.Shape(), c.outH, c.outW, c.outC)
	}

	if r0 < 0 || r1 > c.outH || r0 > r1 {
		return fmt.Errorf("ops: conv2d plan: rows [%d, %d) outside [0, %d)", r0, r1, c.outH)
	}

	return nil
}

// isImage reports whether t has the shape [1, h, w, c], through Rank and
// Dim, which do not allocate.
func isImage(t *tensor.Tensor, h, w, c int) bool {
	return t.Rank() == 4 && t.Dim(0) == 1 && t.Dim(1) == int64(h) && t.Dim(2) == int64(w) && t.Dim(3) == int64(c)
}

// rows computes output rows [r0, r1) into out, the whole output image,
// with col as the im2col scratch: on the SME2 kernel where the plan packed
// its weights for it and the rows hold enough work (rowsSME), else in
// blocks of conv2DPlanBlockM positions through MatMulTransBBias.
func (c *Conv2DPlan) rows(out, x, col []float32, r0, r1 int) {
	if perRow := c.MACsPerRow(); c.pb != nil && r1-r0 >= (conv2DPlanSMEMinFMAs+perRow-1)/perRow {
		c.rowsSME(out, x, col, r0, r1)

		return
	}

	rowLen := c.outW * c.outC

	for y := r0; y < r1; y++ {
		row := out[y*rowLen : (y+1)*rowLen]

		for x0 := 0; x0 < c.outW; x0 += conv2DPlanBlockM {
			m := min(conv2DPlanBlockM, c.outW-x0)
			c.im2col(col, x, y, x0, m)
			tensor.MatMulTransBBias(row[x0*c.outC:], c.outC, col[:m*c.k], c.wt, c.bias, m, c.outC, c.k)
		}
	}
}

// im2col unrolls the receptive fields of the m output positions
// (y, x0..x0+m-1) of the image x into the first m rows of col. Row i holds
// the input patch in (ky, kx, ci) order, with zeros where the patch reaches
// into the padding.
func (c *Conv2DPlan) im2col(col, x []float32, y, x0, m int) {
	cin := c.inC
	seg := c.kw * cin // one kernel row of a patch

	for i := range m {
		dst := col[i*c.k : (i+1)*c.k]
		ix0 := (x0+i)*c.strideW - c.padLeft
		// Valid kernel columns [kx0, kx1) stay inside [0, inW).
		kx0 := max(0, -ix0)
		kx1 := min(c.kw, c.inW-ix0)

		for ky := range c.kh {
			d := dst[ky*seg : (ky+1)*seg]

			iy := y*c.strideH - c.padTop + ky
			if iy < 0 || iy >= c.inH || kx0 >= kx1 {
				clear(d)

				continue
			}

			clear(d[:kx0*cin])
			srcRow := x[iy*c.inW*cin:]
			copy(d[kx0*cin:kx1*cin], srcRow[(ix0+kx0)*cin:(ix0+kx1)*cin])
			clear(d[kx1*cin:])
		}
	}
}
