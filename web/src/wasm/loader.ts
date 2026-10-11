// Loads the Go wasm build of cmd/demo once per JavaScript realm (the page,
// or a worker) and returns its globalThis.algoTensor object.
//
// It runs on the main thread and inside module workers: it never touches
// `document`, and instead of a <script> tag or importScripts it fetches
// wasm_exec.js and runs it as a classic script, which defines globalThis.Go.
// Both files come from import.meta.env.BASE_URL (web/public, filled by
// `just build-demo`).

/**
 * The Go functions of all demos, by demo key. Each src/wasm/<key>.ts adds its
 * own demo through declaration merging, so this file never changes:
 *
 *     declare module "./loader" {
 *       interface AlgoTensorDemos {
 *         conv2d: Conv2DApi;
 *       }
 *     }
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- filled by declaration merging
export interface AlgoTensorDemos {}

/** globalThis.algoTensor, published by cmd/demo/main_js.go. */
export interface AlgoTensor extends AlgoTensorDemos {
  /** runtime.Version() of the Go toolchain that built demo.wasm. */
  readonly goVersion: string;
  /** The registered demo keys, sorted. */
  readonly demos: readonly string[];
}

/**
 * What a Go function returns instead of its result when it fails; see
 * cmd/demo/bridge_js.go. The wrappers pass every result through unwrap.
 */
export interface WasmError {
  readonly error: string;
}

/** A Go function's result: the value, or a WasmError. */
export type WasmResult<T> = T | WasmError;

/** Returns the result of a Go function or throws its error as an Error. */
export function unwrap<T>(result: WasmResult<T>): T {
  if (
    typeof result === "object" &&
    result !== null &&
    "error" in result &&
    typeof result.error === "string"
  ) {
    throw new Error(result.error);
  }
  return result as T;
}

/** The part of wasm_exec.js's Go class the loader uses. */
interface GoRuntime {
  readonly importObject: WebAssembly.Imports;
  run(instance: WebAssembly.Instance): Promise<void>;
}

/** The globals wasm_exec.js and cmd/demo read and write. */
type WasmGlobals = typeof globalThis & {
  Go?: new () => GoRuntime;
  algoTensor?: AlgoTensor;
  onAlgoTensorReady?: () => void;
};

const globals = globalThis as WasmGlobals;

let loading: Promise<AlgoTensor> | undefined;

/**
 * Loads demo.wasm on the first call and returns the same promise afterwards.
 * After a failure, or if the Go program ends later, the next call loads it
 * again.
 */
export function loadAlgoTensor(): Promise<AlgoTensor> {
  loading ??= load().catch((err: unknown) => {
    loading = undefined;
    throw err;
  });
  return loading;
}

async function fetchOk(file: string): Promise<Response> {
  const url = `${import.meta.env.BASE_URL}${file}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  // Vite's dev server answers a missing file with index.html.
  if (response.headers.get("content-type")?.startsWith("text/html")) {
    throw new Error(`${url} is missing; run just build-demo or just dev-demo`);
  }
  return response;
}

async function load(): Promise<AlgoTensor> {
  if (globals.Go === undefined) {
    const source = await (await fetchOk("wasm_exec.js")).text();
    // A Function body runs in the global scope, like a classic script. (A
    // Content-Security-Policy would have to allow 'unsafe-eval' for this.)
    new Function(source)();
  }
  const Go = globals.Go;
  if (Go === undefined) {
    throw new Error("wasm_exec.js did not define globalThis.Go");
  }

  const go = new Go();
  const bytes = await (await fetchOk("demo.wasm")).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, go.importObject);

  return new Promise<AlgoTensor>((resolve, reject) => {
    // cmd/demo calls this once globalThis.algoTensor is complete.
    globals.onAlgoTensorReady = () => {
      globals.onAlgoTensorReady = undefined;
      if (globals.algoTensor === undefined) {
        reject(new Error("demo.wasm did not define globalThis.algoTensor"));
      } else {
        resolve(globals.algoTensor);
      }
    };
    // run settles only when the Go program ends, which it never should.
    // Before it was ready that fails this load; afterwards the next
    // loadAlgoTensor starts a new program.
    const ended = (err: unknown) => {
      globals.onAlgoTensorReady = undefined;
      globals.algoTensor = undefined;
      loading = undefined;
      reject(err);
    };
    go.run(instance).then(() => ended(new Error("demo.wasm exited")), ended);
  });
}
