# Migrating go-whistle and go-pocket-tts to algo-tensor

Both projects carry a copy of these packages: go-whistle as `internal/tensor` and `internal/ops` (main @ 8926ee3),
go-pocket-tts as `internal/runtime/tensor` and `internal/runtime/ops` (@ 9faba38). The module is the union of both
copies, so each project switches by changing import paths only: the package names stay `tensor` and `ops`, and every
identifier keeps its name and signature. No `gofmt -r` rewrite is needed; the commands below only replace the import
paths.

The lists below come from the code: `go list -f '{{.Imports}} {{.TestImports}}' ./...` for the importing packages, and
a grep of `tensor.X` / `ops.X` in every tracked `.go` file that imports the packages (build-tagged files included).

## go-whistle

Importers: `internal/cq`, `internal/decoder`, `internal/encoder`, `internal/needle`, `internal/weights` (tests only).

| go-whistle identifier                    | used in                             | algo-tensor identifier | behaviour                     |
| ---------------------------------------- | ----------------------------------- | ---------------------- | ----------------------------- |
| `internal/tensor`.`Axpy`                 | decoder, needle                     | `tensor.Axpy`          | same code                     |
| `internal/tensor`.`DotProduct`           | decoder, needle                     | `tensor.DotProduct`    | same code                     |
| `internal/tensor`.`MatMulTransB`         | cq, needle                          | `tensor.MatMulTransB`  | same code                     |
| `internal/tensor`.`New`                  | encoder                             | `tensor.New`           | same code                     |
| `internal/tensor`.`Parallel`             | cq, encoder, needle                 | `tensor.Parallel`      | same code                     |
| `internal/tensor`.`SetWorkers`           | cq, encoder, needle, weights (test) | `tensor.SetWorkers`    | same code                     |
| `internal/tensor`.`Tensor` (and methods) | encoder                             | `tensor.Tensor`        | same code                     |
| `internal/tensor`.`Workers`              | cq, encoder, needle, weights (test) | `tensor.Workers`       | same code, default GOMAXPROCS |
| `internal/ops`.`Conv2D`                  | encoder                             | `ops.Conv2D`           | same code                     |
| `internal/ops`.`Conv2DParams`            | encoder                             | `ops.Conv2DParams`     | same code                     |
| `internal/ops`.`SamePadding`             | encoder                             | `ops.SamePadding`      | same code                     |

Every declaration of go-whistle's two packages is in the module unchanged (compared declaration by declaration on the
comment-stripped syntax trees), except `ops.KernelTolerances`, which has go-pocket-tts' three extra entries
(`layer_norm`, `mlp`, `convtranspose1d`) again. go-whistle does not use it outside `internal/ops`.

```sh
cd go-whistle
git rm -r -q internal/tensor internal/ops
git grep -lz -e '"github.com/cwbudde/go-whistle/internal/tensor"' -e '"github.com/cwbudde/go-whistle/internal/ops"' -- '*.go' |
  xargs -0 perl -pi -e '
    s|"github.com/cwbudde/go-whistle/internal/tensor"|"github.com/cwbudde/algo-tensor/tensor"|;
    s|"github.com/cwbudde/go-whistle/internal/ops"|"github.com/cwbudde/algo-tensor/ops"|'
go get github.com/cwbudde/algo-tensor@v0.1.0
go mod tidy
just fmt # gci moves the imports from the module's own group to the third-party group
```

`perl -pi -e` is a `sed -i` that behaves the same with BSD and GNU tools. With GNU sed,
`xargs -0 sed -i -e 's|…/internal/tensor"|…/algo-tensor/tensor"|' -e 's|…/internal/ops"|…/algo-tensor/ops"|'` (paths
written out) does the same.

Afterwards, by hand:

- Comments that name the old paths: `cmd/whistle/doctor.go` (`kernels`) and `internal/cq/whistle_test.go`. `kernels`
  stays correct: the module has the same `purego` tag and CPU checks.
- `docs/FINDINGS.md` and `PLAN.md` name `internal/tensor` and `internal/ops`; the module's comments keep pointing to
  go-whistle's F-numbers.

## go-pocket-tts

Importers: the root package `pockettts` (`engine.go`), `cmd/pockettts`, `internal/bench/stageprof`, `internal/native`,
`internal/nativert`, `internal/tts`.

| go-pocket-tts identifier                                   | used in                                          | algo-tensor identifier                  | behaviour                                  |
| ---------------------------------------------------------- | ------------------------------------------------ | --------------------------------------- | ------------------------------------------ |
| `internal/runtime/tensor`.`BroadcastAdd`                   | native                                           | `tensor.BroadcastAdd`                   | same code                                  |
| `internal/runtime/tensor`.`BroadcastMul`                   | native                                           | `tensor.BroadcastMul`                   | same code                                  |
| `internal/runtime/tensor`.`Concat`                         | native, nativert                                 | `tensor.Concat`                         | same code                                  |
| `internal/runtime/tensor`.`Full`                           | native                                           | `tensor.Full`                           | overflow check, see 3                      |
| `internal/runtime/tensor`.`LayerNorm`                      | native                                           | `tensor.LayerNorm`                      | same code                                  |
| `internal/runtime/tensor`.`Linear`                         | native                                           | `tensor.Linear`                         | same bits, see 2                           |
| `internal/runtime/tensor`.`MatMulTransB`                   | native                                           | `tensor.MatMulTransB`                   | same bits, see 2                           |
| `internal/runtime/tensor`.`New`                            | native, nativert                                 | `tensor.New`                            | overflow check, see 3                      |
| `internal/runtime/tensor`.`SetWorkers`                     | pockettts, cmd/pockettts, stageprof, native, tts | `tensor.SetWorkers`                     | same code                                  |
| `internal/runtime/tensor`.`Tensor` (and methods)           | cmd/pockettts, native, nativert                  | `tensor.Tensor`                         | same code                                  |
| `internal/runtime/tensor`.`Workers`                        | cmd/pockettts, native                            | `tensor.Workers`                        | **default GOMAXPROCS instead of 1**, see 1 |
| `internal/runtime/tensor`.`Zeros`                          | native                                           | `tensor.Zeros`                          | overflow check, see 3                      |
| `internal/runtime/ops`.`Attention`                         | native                                           | `ops.Attention`                         | same code                                  |
| `internal/runtime/ops`.`AttentionNoContext`                | native                                           | `ops.AttentionNoContext`                | same value                                 |
| `internal/runtime/ops`.`AttentionWithPositions`            | native                                           | `ops.AttentionWithPositions`            | same code                                  |
| `internal/runtime/ops`.`Conv1D`                            | native                                           | `ops.Conv1D`                            | short-input error, see 4                   |
| `internal/runtime/ops`.`Conv1DLeftPad`                     | native                                           | `ops.Conv1DLeftPad`                     | short-input error, see 4                   |
| `internal/runtime/ops`.`ConvTranspose1DPrePackedRightTrim` | native                                           | `ops.ConvTranspose1DPrePackedRightTrim` | same code                                  |
| `internal/runtime/ops`.`ConvTranspose1DRightTrim`          | native                                           | `ops.ConvTranspose1DRightTrim`          | same code                                  |
| `internal/runtime/ops`.`KernelTolerance`                   | native                                           | `ops.KernelTolerance`                   | same code and map                          |
| `internal/runtime/ops`.`RepackConvTransposeKernel`         | native                                           | `ops.RepackConvTransposeKernel`         | same code                                  |
| `internal/runtime/ops`.`RoPE`                              | native                                           | `ops.RoPE`                              | same code                                  |
| `internal/runtime/ops`.`SetConvWorkers`                    | pockettts, stageprof, native, tts                | `ops.SetConvWorkers`                    | same code                                  |
| `internal/runtime/ops`.`Tolerance`                         | native                                           | `ops.Tolerance`                         | same type                                  |

Every declaration of go-pocket-tts' two packages is in the module (same check as above). The ones that differ are
go-whistle's changes, which the module keeps:

1. **Default worker count.** `Workers()` is GOMAXPROCS (read once at start-up) until `SetWorkers` is called, where it
   was 1. go-pocket-tts' CLI and server set it from the config (`tts.ApplyNativeWorkers`, from `tts.NewService` and
   `export-voice`), so they are not affected. The library is: `pockettts.SetWorkers` documents "default 1", and its
   `internal/native` kernels split their work by `tensor.Workers()` in ways that change the summation order. A program
   that never calls `pockettts.SetWorkers` now runs those kernels on GOMAXPROCS goroutines and gets other bits for the
   same seed than before. To keep the old default, go-pocket-tts can call `tensor.SetWorkers(1)` before its first
   kernel runs (in `pockettts.Load`, for example, unless the caller set a count). The pool is process-wide, so an
   `init` would also change the default for everything else in the binary that uses algo-tensor.
2. **`MatMulTransB` (and `Linear` and `MatMul` through it) with `Workers() > 1`**: products of at least 2²⁰
   multiply-adds are split over `tensor.Parallel` in pieces of whole 4-row blocks (go-whistle F-038). Every output is
   the same bit for bit as on one worker; `TestMatMulTransBWorkers` and the golden hashes check it.
3. **`New`, `Zeros`, `Full`, `Reshape` and `MatMul`'s batch shape** check the element count for overflow before
   multiplying. A shape
   such as [3e9, 4e9] used to wrap to a negative count and could pass the checks; it is an error now.
4. **`Conv1D`, `Conv1DLeftPad`**: an input shorter than the dilated kernel (a negative span) gave one output column,
   because Go's division truncates towards zero; it is an error now ("non-positive output length"). Valid inputs give
   the same bits.

```sh
cd go-pocket-tts
git rm -r -q internal/runtime/tensor internal/runtime/ops
git grep -lz -e '"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"' -e '"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"' -- '*.go' |
  xargs -0 perl -pi -e '
    s|"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"|"github.com/cwbudde/algo-tensor/tensor"|;
    s|"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"|"github.com/cwbudde/algo-tensor/ops"|'
go get github.com/cwbudde/algo-tensor@v0.1.0
go mod tidy
just fmt
```

Afterwards, by hand:

- `.golangci.yml`: the `internal/runtime/tensor/.*\.go` gosec G103 rule and the `internal/runtime/` cyclop/gocognit
  rule have nothing left to match.
- `justfile` (line 34) runs `./internal/runtime/tensor/` under `qemu-aarch64-static`; the arm64 assembly now lives in
  algo-tensor, whose `go test ./...` runs it on Apple silicon.

## Both in one binary

`tensor.SetWorkers`, `Workers`, `Parallel` and `ops.SetConvWorkers` are process-wide. A binary that links both
projects shares one worker count and one helper budget: whichever calls `SetWorkers` last sets it for both.
