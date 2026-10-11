// Package demo is the logic behind the browser demo in cmd/demo: the
// computations and data conversions, in plain Go so that they are tested and
// linted on the host. The js/wasm glue in cmd/demo only parses the
// JavaScript arguments, calls this package and converts the results back.
//
// Each demo keeps its logic in its own file, internal/demo/<key>.go, and its
// tests in internal/demo/<key>_test.go, for the keys conv2d, attention, bench
// and signals. The identifiers in a demo's file start with the demo's name
// (Conv2D…, Attention…, Bench…, Signals…, or conv2d…, attention…, … when
// unexported), so that demos written in parallel do not collide. convert.go
// and image.go hold the helpers several demos share.
package demo
