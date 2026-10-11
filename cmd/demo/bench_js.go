package main

import (
	"syscall/js"

	"github.com/cwbudde/algo-tensor/internal/demo"
)

func init() {
	register("bench", map[string]Func{
		"dotProduct": benchDotProduct,
	})
}

// benchDotProduct is dotProduct(n: number, reps: number): {n, reps, seconds,
// gflops, checksum}, a timing of reps tensor.DotProduct calls on n values
// (demo.BenchDotProduct). It is the scaffold's stub.
func benchDotProduct(args []js.Value) (any, error) {
	n, err := argInt(args, 0, "n")
	if err != nil {
		return nil, err
	}

	reps, err := argInt(args, 1, "reps")
	if err != nil {
		return nil, err
	}

	r, err := demo.BenchDotProduct(n, reps)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"n":        r.N,
		"reps":     r.Reps,
		"seconds":  r.Seconds,
		"gflops":   r.GFLOPS,
		"checksum": r.Checksum,
	}, nil
}
