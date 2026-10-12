// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of Conv2DPlan
// on the SME2 kernel (PLAN.md 2.2).

package ops

import (
	"os"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// noSMEEnv names an environment variable that turns the SME path off in
// the test binary, as in tensor's tests, for the NEON ("before") numbers of
// the Conv2DPlan benchmarks (ops/PERFORMANCE.md).
const noSMEEnv = "ALGO_TENSOR_NO_SME"

//nolint:gochecknoinits // the switch must act before the first test or benchmark
func init() {
	if os.Getenv(noSMEEnv) != "" {
		sme.GEMM = nil
	}
}
