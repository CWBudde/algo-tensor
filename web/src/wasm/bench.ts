// Types and typed wrappers for the bench demo's Go functions
// (cmd/demo/bench_js.go), and the messages of its worker (bench.worker.ts).

import { loadAlgoTensor, unwrap, type WasmResult } from "./loader";

/** One timing of a kernel. */
export interface BenchResult {
  /** Vector length. */
  readonly n: number;
  /** Calls timed. */
  readonly reps: number;
  /** Wall time of all calls. */
  readonly seconds: number;
  /** 2·n·reps / seconds / 10⁹; 0 if the clock did not advance. */
  readonly gflops: number;
  /** Sum of the results. */
  readonly checksum: number;
}

/** globalThis.algoTensor.bench */
export interface BenchApi {
  /** Times reps calls of tensor.DotProduct on n values. */
  dotProduct(n: number, reps: number): WasmResult<BenchResult>;
}

declare module "./loader" {
  interface AlgoTensorDemos {
    bench: BenchApi;
  }
}

/** Times reps calls of tensor.DotProduct on n values in this thread. */
export async function benchDotProduct(
  n: number,
  reps: number,
): Promise<BenchResult> {
  const at = await loadAlgoTensor();
  return unwrap(at.bench.dotProduct(n, reps));
}

/** A message to bench.worker.ts. */
export interface BenchRequest {
  readonly n: number;
  readonly reps: number;
}

/** bench.worker.ts's answer to a BenchRequest. */
export type BenchResponse =
  | {
      readonly ok: true;
      readonly result: BenchResult;
      readonly inWorker: boolean;
    }
  | { readonly ok: false; readonly error: string };
