// The bench demo's module worker: it loads its own copy of demo.wasm and
// runs the timings off the main thread, so the page stays responsive.

import {
  benchDotProduct,
  type BenchRequest,
  type BenchResponse,
} from "./bench";

function reply(response: BenchResponse) {
  self.postMessage(response);
}

self.addEventListener("message", (event: MessageEvent<BenchRequest>) => {
  const { n, reps } = event.data;
  benchDotProduct(n, reps).then(
    (result) =>
      reply({ ok: true, result, inWorker: self instanceof WorkerGlobalScope }),
    (err: unknown) => reply({ ok: false, error: String(err) }),
  );
});
