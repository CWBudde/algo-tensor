package main

import (
	"runtime"
	"sort"
	"syscall/js"
)

// Func is one function of a demo. It gets the JavaScript arguments and
// returns a value js.ValueOf accepts (a js.Value, a number, a string, a
// bool, nil, []any or map[string]any) or an error; see bridge_js.go.
type Func func(args []js.Value) (any, error)

// registry maps each demo key to its functions by name.
var registry = map[string]map[string]Func{}

// register adds a demo's functions as globalThis.algoTensor.<key>.<name>.
// Each demo calls it once, from the init function in its <key>_js.go.
func register(key string, funcs map[string]Func) {
	if _, ok := registry[key]; ok {
		panic("demo: key " + key + " registered twice")
	}

	registry[key] = funcs
}

// main publishes globalThis.algoTensor, sets globalThis.algoTensorReady,
// calls globalThis.onAlgoTensorReady if the loader set it, and then keeps
// the Go program alive for the callbacks.
func main() {
	object := js.Global().Get("Object")
	api := object.New()
	api.Set("goVersion", runtime.Version())

	keys := make([]string, 0, len(registry))
	for key, funcs := range registry {
		keys = append(keys, key)
		demo := object.New()

		for name, fn := range funcs {
			demo.Set(name, export(key+"."+name, fn))
		}

		api.Set(key, demo)
	}

	sort.Strings(keys)
	api.Set("demos", stringsToJS(keys))

	js.Global().Set("algoTensor", api)
	js.Global().Set("algoTensorReady", true)

	if ready := js.Global().Get("onAlgoTensorReady"); ready.Type() == js.TypeFunction {
		ready.Invoke()
	}

	select {}
}
