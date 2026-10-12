//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: a stress test of
// the SME2 path of MatMulTransBBias under signals and GC (PLAN.md 2.1), and
// of sme.GEMM (PLAN.md 2.2).

package tensor

import (
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// smeStressDuration is how long TestGemmSMEStress runs its GEMMs.
const smeStressDuration = 3 * time.Second

// smeStressCase is one product of TestGemmSMEStress with gemmNEON4x4's
// output, the gaps between rows and one guard element past the output
// set to gemmSentinel. public runs it through MatMulTransBBias instead of
// gemmBiasSMERun; pb, when set, holds b packed by sme.Pack and runs it
// through sme.GEMM, computed again on NEON when that returns false, as
// ops.Conv2DPlan does.
type smeStressCase struct {
	m, n, k, ldc int
	a, b, bias   []float32
	want         []float32
	public       bool
	pb           []float32
}

func newSMEStressCase(rng *rand.Rand, m, n, k, ldc int, public bool) smeStressCase {
	c := smeStressCase{
		m: m, n: n, k: k, ldc: ldc, public: public,
		a: randGemmData(rng, m*k), b: randGemmData(rng, n*k), bias: randGemmData(rng, n),
	}

	c.want = make([]float32, (m-1)*ldc+n+1)
	for i := range c.want {
		c.want[i] = gemmSentinel
	}

	gemmNEON4x4(&c.want[0], &c.a[0], &c.b[0], &c.bias[0], m, n, k, ldc)

	return c
}

// run fills got with gemmSentinel, runs the product into it and returns
// the first element whose bits differ from want, or -1.
func (c *smeStressCase) run(got []float32) int {
	for i := range got {
		got[i] = gemmSentinel
	}

	switch {
	case c.pb != nil:
		po := make([]float32, sme.PanelRows*sme.Steps(c.k))
		if !sme.GEMM(got, c.ldc, c.a, po, c.pb, c.bias, c.m, c.n, c.k) {
			gemmNEON4x4(&got[0], &c.a[0], &c.b[0], &c.bias[0], c.m, c.n, c.k, c.ldc)
		}
	case c.public:
		MatMulTransBBias(got, c.ldc, c.a, c.b, c.bias, c.m, c.n, c.k)
	default:
		gemmBiasSMERun(got, c.ldc, c.a, c.b, c.bias, c.m, c.n, c.k)
	}

	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(c.want[i]) {
			return i
		}
	}

	return -1
}

// untilClosed calls fn until stop is closed.
func untilClosed(wg *sync.WaitGroup, stop <-chan struct{}, fn func()) {
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				fn()
			}
		}
	})
}

// TestGemmSMEStress runs SME GEMMs on 2×GOMAXPROCS goroutines for a few
// seconds under a CPU profile, a SIGUSR1 storm and runtime.GC and checks
// every output bit, gap and guard against gemmNEON4x4: on darwin a signal
// that arrives in streaming mode zeroes lanes 4-15 of every Z register,
// which gemmSME must detect and redo. The products run through
// gemmBiasSMERun, MatMulTransBBias and sme.GEMM with b packed by sme.Pack.
func TestGemmSMEStress(t *testing.T) {
	if !gemmBiasUseSME {
		t.Skip("no SME2")
	}

	if testing.Short() {
		t.Skip("stress test skipped in short mode")
	}

	rng := rand.New(rand.NewPCG(21, 22))

	direct := [][3]int{{21, 20, 4}, {7, 36, 5}, {40, 64, 6}, {21, 20, 7}, {7, 64, 8}, {40, 36, 9}, {1, 20, 9}}
	public := [][3]int{{40, 36, 297}, {21, 64, 672}}
	packed := [][3]int{{40, 36, 297}, {3, 20, 7}, {17, 64, 200}}
	cases := make([]smeStressCase, 0, 2*(len(direct)+len(public)+len(packed)))

	for _, s := range direct {
		cases = append(cases, newSMEStressCase(rng, s[0], s[1], s[2], s[1], false),
			newSMEStressCase(rng, s[0], s[1], s[2], s[1]+3, false))
	}

	for _, s := range public {
		if !smeTakes(s[0], s[1], s[2]) {
			t.Fatalf("smeTakes(%d, %d, %d) = false, want the public product on SME", s[0], s[1], s[2])
		}

		cases = append(cases, newSMEStressCase(rng, s[0], s[1], s[2], s[1], true),
			newSMEStressCase(rng, s[0], s[1], s[2], s[1]+3, true))
	}

	for _, s := range packed {
		for _, ldc := range []int{s[1], s[1] + 3} {
			c := newSMEStressCase(rng, s[0], s[1], s[2], ldc, false)
			c.pb = make([]float32, sme.PackedLen(c.n, c.k))
			sme.Pack(c.pb, c.b, c.n, c.k)
			cases = append(cases, c)
		}
	}

	err := pprof.StartCPUProfile(io.Discard)
	if err != nil {
		t.Logf("no CPU profile: %v", err)
	} else {
		defer pprof.StopCPUProfile()
	}

	sigs := make(chan os.Signal, 64)
	signal.Notify(sigs, syscall.SIGUSR1)

	defer signal.Stop(sigs)

	redos0, fallbacks0 := smeRedos.Load(), smeFallbacks.Load()
	stop := make(chan struct{})

	var (
		wg, drain        sync.WaitGroup
		gemms, bad, sent atomic.Int64
		firstMu          sync.Mutex
		first            string
	)

	drainStop := make(chan struct{})

	untilClosed(&drain, drainStop, func() {
		select {
		case <-sigs:
		case <-drainStop:
		}
	})

	untilClosed(&wg, stop, func() {
		if syscall.Kill(os.Getpid(), syscall.SIGUSR1) == nil {
			sent.Add(1)
		}

		runtime.Gosched()
	})

	untilClosed(&wg, stop, func() {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	})

	for g := range 2 * runtime.GOMAXPROCS(0) {
		got := make([][]float32, len(cases))
		for i := range cases {
			got[i] = make([]float32, len(cases[i].want))
		}

		iter := g

		untilClosed(&wg, stop, func() {
			ci := iter % len(cases)
			iter++

			c := &cases[ci]

			gemms.Add(1)

			if i := c.run(got[ci]); i >= 0 && bad.Add(1) == 1 {
				firstMu.Lock()
				first = fmt.Sprintf("m=%d n=%d k=%d ldc=%d public=%v packed=%v: c[%d] = %#08x, NEON %#08x",
					c.m, c.n, c.k, c.ldc, c.public, c.pb != nil, i, math.Float32bits(got[ci][i]), math.Float32bits(c.want[i]))
				firstMu.Unlock()
			}
		})
	}

	time.Sleep(smeStressDuration)
	close(stop)
	wg.Wait()
	close(drainStop)
	drain.Wait()

	t.Logf("%d GEMMs, %d SIGUSR1 sent, %d blocks redone, %d NEON fallbacks",
		gemms.Load(), sent.Load(), smeRedos.Load()-redos0, smeFallbacks.Load()-fallbacks0)

	if n := bad.Load(); n > 0 {
		firstMu.Lock()
		t.Errorf("%d of %d GEMMs differ from gemmNEON4x4, first: %s", n, gemms.Load(), first)
		firstMu.Unlock()
	}

	if gemms.Load() == 0 {
		t.Error("no GEMM ran")
	}

	// Not a failure: on a quiet machine no signal need land in streaming
	// mode. But then the run did not exercise the guard.
	if smeRedos.Load() == redos0 {
		t.Log("warning: no block was redone, so no signal hit the SME kernel")
	}
}
