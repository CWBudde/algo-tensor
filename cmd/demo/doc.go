// Command demo is the browser demo of algo-tensor: built for js/wasm by
// "just build-demo" and loaded by web/src/wasm/loader.ts, it exposes the
// demos' Go functions as globalThis.algoTensor.<key>.<name>.
//
// main_js.go builds that object from a registry that each demo's
// <key>_js.go fills in its init function, so adding a demo does not touch
// main_js.go. bridge_js.go converts arguments and results and defines how
// errors reach JavaScript. The computations live in internal/demo.
//
// The _js.go files build for GOOS=js only; on other systems the package is
// main_other.go, which exits with a hint, so that "go build ./..." works.
package main
