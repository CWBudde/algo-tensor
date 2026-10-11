// The bench demo: kernel GFLOP/s over sizes, measured in a Web Worker.
// Scaffold stub: one tensor.DotProduct timing in the worker.

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { BenchRequest, BenchResponse } from "@/wasm/bench";

const REQUEST: BenchRequest = { n: 4096, reps: 2000 };

export function Benchmark() {
  const [run, setRun] = useState(0);
  const [response, setResponse] = useState<BenchResponse>();

  useEffect(() => {
    const worker = new Worker(
      new URL("../wasm/bench.worker.ts", import.meta.url),
      { type: "module", name: "bench" },
    );
    worker.addEventListener("message", (event: MessageEvent<BenchResponse>) => {
      setResponse(event.data);
      worker.terminate();
    });
    worker.addEventListener("error", (event: ErrorEvent) => {
      setResponse({ ok: false, error: event.message || "the worker failed" });
      worker.terminate();
    });
    worker.postMessage(REQUEST);
    return () => worker.terminate();
  }, [run]);

  function again() {
    setResponse(undefined);
    setRun((r) => r + 1);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Kernel benchmark <Badge variant="secondary">stub</Badge>
        </CardTitle>
        <CardDescription>
          {REQUEST.reps} calls of <code>tensor.DotProduct</code> on {REQUEST.n}{" "}
          values, timed in a Web Worker with its own copy of the wasm.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col items-start gap-4">
        <p className="text-sm" data-testid="bench-status">
          {response === undefined
            ? "Running in the worker…"
            : response.ok
              ? `${response.result.gflops.toFixed(2)} GFLOP/s (${(response.result.seconds * 1000).toFixed(1)} ms, checksum ${response.result.checksum}) ${response.inWorker ? "in a Web Worker" : "on the main thread"}.`
              : `Error: ${response.error}`}
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={again}
          disabled={response === undefined}
        >
          Run again
        </Button>
      </CardContent>
    </Card>
  );
}
