//go:build arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: MatMulTransBBias on
// arm64 CPUs with SME2 through an SME2 kernel that gives gemmNEON4x4's bits
// (PLAN.md 2.1).

package tensor

import (
	"sync"
	"sync/atomic"
)

// gemmSME is the SME2 kernel (gemm_bias_sme_arm64.s): it computes the
// product of gemmBias for m >= 1, n a positive multiple of 4, k >= 4 and
// ldc >= n with every output bit for bit as gemmNEON4x4 does, in one
// streaming-mode section. With bOuter = 0 it packs b whole into pi and runs
// a panel by panel through po, with bOuter = 1 the other way round; po
// holds 16·kp floats (kp = 4⌈k/4⌉) and pi 16·⌈rows/16⌉·kp for the rows of
// the inner operand; with innerPacked = 1, pi holds the inner operand
// packed by the caller (gemmBiasSMEPacked). clobberK and clobberR are test
// hooks (see
// smeClobberK). It returns the number of failed signal checks (each
// redoes a block or its stores), the outer rows done (rows of a for bOuter
// = 0, of b for 1): all, or fewer when one block failed its check nine
// times, in which case the outputs from that row (column) on are left to
// the caller, and nan = 1 if it stored a NaN or an infinity. After a failed
// check, nan may miss one. Its NaNs are the default NaN (FMOPA behaves as
// if FPCR.DN were set), where gemmNEON4x4 passes on a NaN operand's
// payload and sign, so their bits may differ; all other outputs have
// gemmNEON4x4's bits.
//
//go:noescape
func gemmSME(c *float32, ldc int, a, b, bias *float32, m, n, k int, po, pi *float32, bOuter, innerPacked, clobberK, clobberR int) (redos, done, nan int)

// gemmBiasUseSME reports whether MatMulTransBBias runs the products that
// smeTakes accepts on gemmSME. It is set once at start-up (smeSupported).
var gemmBiasUseSME = smeSupported()

// The products gemmSME takes, from BenchmarkMatMulTransBBias on an Apple
// M5 Pro (ops/PERFORMANCE.md): at least smeMinRows rows, smeMinCols
// columns and smeMinFMAs multiply-adds. The smaller products stay on NEON:
// fewer rows (a GEMV in particular, where the packed copy of b costs more
// than NEON's single pass over it), fewer columns, which fill less than a
// tile, and short products, which the packing and the reduction outweigh.
// Products of 8 rows gain alone, but ops.Conv2DPlan, which runs its im2col
// blocks of 8 rows through MatMulTransBBias, lost time with them on SME, so
// the least is 16. Tests lower all three (setSMEThreshold).
const (
	smeDefaultMinRows = 16
	smeDefaultMinCols = 16
	smeDefaultMinFMAs = 1 << 15
)

var (
	smeMinRows = newAtomicInt64(smeDefaultMinRows)
	smeMinCols = newAtomicInt64(smeDefaultMinCols)
	smeMinFMAs = newAtomicInt64(smeDefaultMinFMAs)
)

// newAtomicInt64 returns an atomic.Int64 that holds v.
func newAtomicInt64(v int64) *atomic.Int64 {
	a := new(atomic.Int64)
	a.Store(v)

	return a
}

// smePieceFMAs is the least work of one Parallel piece of the SME path.
// Every piece packs its share of one operand and the whole other one, and
// all workers together reach only about twice the speed of one
// (ops/PERFORMANCE.md), so the pieces are much larger than gemmPieceFMAs:
// smaller ones ran slower than one worker.
const smePieceFMAs = 1 << 24

// Test hooks: smeClobberK (smeClobberR) clobbers the canary of gemmSME's
// check after the k loop (after the stores), as a signal would; from<<32 |
// count clobbers count checks from the one numbered from on (counted from
// 0 in each gemmSME call; the hooks also clear the NaN collector, as a
// signal does). smeRedos counts the failed signal checks, smeFallbacks the
// calls finished on NEON after a block failed nine times, smeNaNs the calls
// computed again on NEON for a NaN or infinite output.
var (
	smeClobberK  atomic.Uint64
	smeClobberR  atomic.Uint64
	smeRedos     atomic.Int64
	smeFallbacks atomic.Int64
	smeNaNs      atomic.Int64
)

// smeTakes reports whether gemmSME runs the product m×n×k.
func smeTakes(m, n, k int) bool {
	return gemmBiasUseSME && k >= 4 && n%4 == 0 &&
		int64(m) >= smeMinRows.Load() && int64(n) >= smeMinCols.Load() &&
		int64(m)*int64(n)*int64(k) >= smeMinFMAs.Load()
}

// matMulTransBBiasSME runs the checked product of MatMulTransBBias on
// gemmSME and reports whether it did; it returns false, without touching
// c, for products that smeTakes declines. len(bias) is n.
//
// On more than one worker, it splits the larger of m and n over Parallel
// in whole 16-row (16-column) blocks, one piece per worker where the work
// allows: each piece packs its share of a (b) and all of b (a). Every
// output's bits are the same for any split.
func matMulTransBBiasSME(c []float32, ldc int, a, b, bias []float32, m, n, k int) bool {
	if !smeTakes(m, n, k) {
		return false
	}

	w := getWorkers()
	if w <= 1 || m*n*k < 2*smePieceFMAs {
		gemmBiasSMERun(c, ldc, a, b, bias, m, n, k)
		return true
	}

	if n >= m {
		blocks := (n + 15) / 16
		grain := max((blocks+w-1)/w, smePieceFMAs/(16*m*k))

		Parallel(blocks, grain, func(lo, hi int) {
			j0, j1 := lo*16, min(n, hi*16)
			gemmBiasSMERun(c[j0:], ldc, a, b[j0*k:j1*k], bias[j0:j1], m, j1-j0, k)
		})

		return true
	}

	blocks := (m + 15) / 16
	grain := max((blocks+w-1)/w, smePieceFMAs/(16*n*k))

	Parallel(blocks, grain, func(lo, hi int) {
		i0, i1 := lo*16, min(m, hi*16)
		gemmBiasSMERun(c[i0*ldc:], ldc, a[i0*k:i1*k], b, bias, i1-i0, n, k)
	})

	return true
}

// smeScratch holds the packed panels of one gemmSME call.
type smeScratch struct{ buf []float32 }

// smeScratchPool keeps the panels between calls, so that MatMulTransBBias
// does not allocate in steady state.
var smeScratchPool = sync.Pool{New: func() any { return new(smeScratch) }}

// gemmBiasSMERun runs gemmSME on one product (or one Parallel piece) and
// computes with gemmNEON4x4 the outputs from the rows (columns) of a block
// the kernel gave up on, and all of them if one is NaN or infinite, so
// every output has gemmNEON4x4's bits. After a failed signal check it looks
// for those outputs itself. It needs what gemmSME needs; the slices are bounded
// as in gemmBiasSIMD before their pointers reach the assembly.
//
// The operand with more rows runs panel by panel (outer), the other one is
// packed whole and read once per outer panel: for a few rows of a against
// many of b (a Linear over a few tokens), a stays in the cache and b is
// read once.
func gemmBiasSMERun(c []float32, ldc int, a, b, bias []float32, m, n, k int) {
	a = a[0 : m*k : len(a)]
	b = b[0 : n*k : len(b)]
	bias = bias[0:n:len(bias)]
	c = c[0 : (m-1)*ldc+n : len(c)]

	bOuter, inner := 0, n
	if n > m {
		bOuter, inner = 1, m
	}

	kp := (k + 3) &^ 3
	no, ni := 16*kp, (inner+15)/16*16*kp

	s, ok := smeScratchPool.Get().(*smeScratch)
	if !ok {
		s = new(smeScratch)
	}

	if len(s.buf) < no+ni {
		s.buf = make([]float32, no+ni)
	}

	redos, done, nan := gemmSME(&c[0], ldc, &a[0], &b[0], &bias[0], m, n, k, &s.buf[0], &s.buf[no],
		bOuter, 0, int(smeClobberK.Load()), int(smeClobberR.Load()))
	smeScratchPool.Put(s)

	if redos > 0 {
		smeRedos.Add(int64(redos))
	}

	switch {
	case bOuter == 0 && done < m:
		smeFallbacks.Add(1)
		gemmNEON4x4(&c[done*ldc], &a[done*k], &b[0], &bias[0], m-done, n, k, ldc)
	case bOuter == 1 && done < n:
		smeFallbacks.Add(1)
		gemmNEON4x4(&c[done], &a[0], &b[done*k], &bias[done], m, n-done, k, ldc)
	}

	if nan != 0 || (redos > 0 && !finiteOutputs(c, ldc, m, n)) {
		smeNaNs.Add(1)
		gemmNEON4x4(&c[0], &a[0], &b[0], &bias[0], m, n, k, ldc)
	}
}

// finiteOutputs reports whether c[i*ldc+j] is finite for every i < m and
// j < n.
func finiteOutputs(c []float32, ldc, m, n int) bool {
	for i := range m {
		for _, x := range c[i*ldc : i*ldc+n] {
			if x-x != 0 { // NaN for a NaN or an infinity
				return false
			}
		}
	}

	return true
}
