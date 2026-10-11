import { BookOpen, Code, LoaderCircle } from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Attention } from "@/demos/Attention";
import { Benchmark } from "@/demos/Benchmark";
import { Conv2D } from "@/demos/Conv2D";
import { Signals } from "@/demos/Signals";
import { loadAlgoTensor } from "@/wasm/loader";

const REPO = "https://github.com/CWBudde/algo-tensor";
const DOCS = "https://pkg.go.dev/github.com/cwbudde/algo-tensor";

/** The demos, one tab each, keyed like cmd/demo/<key>_js.go. */
const DEMOS: { key: string; label: string; Demo: ComponentType }[] = [
  { key: "conv2d", label: "Conv2D filters", Demo: Conv2D },
  { key: "attention", label: "Attention", Demo: Attention },
  { key: "bench", label: "Benchmark", Demo: Benchmark },
  { key: "signals", label: "Signals", Demo: Signals },
];

type LoadState =
  | { status: "loading" }
  | { status: "ready"; goVersion: string }
  | { status: "error"; message: string };

function Header({ goVersion }: { goVersion?: string }) {
  return (
    <header className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="font-heading text-3xl font-semibold tracking-tight">
            algo-tensor
          </h1>
          <p className="text-muted-foreground">
            Pure-Go tensor kernels and neural-network ops, compiled to
            WebAssembly and running in your browser.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href={REPO} target="_blank" rel="noreferrer">
              <Code /> GitHub
            </a>
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href={DOCS} target="_blank" rel="noreferrer">
              <BookOpen /> pkg.go.dev
            </a>
          </Button>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        In the browser the library runs as{" "}
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="cursor-help">
              js/wasm
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            {goVersion === undefined
              ? "GOOS=js GOARCH=wasm"
              : `Built with ${goVersion}, GOOS=js GOARCH=wasm`}
          </TooltipContent>
        </Tooltip>
        : the pure-Go fallbacks instead of the NEON and AVX2 assembly, on a
        single thread. Native builds are several times faster.
      </p>
    </header>
  );
}

export default function App() {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    loadAlgoTensor().then(
      (at) => {
        if (!cancelled) setState({ status: "ready", goVersion: at.goVersion });
      },
      (err: unknown) => {
        if (!cancelled) setState({ status: "error", message: String(err) });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  function retry() {
    setState({ status: "loading" });
    setAttempt((a) => a + 1);
  }

  return (
    <TooltipProvider>
      <div className="mx-auto flex min-h-svh max-w-5xl flex-col gap-6 px-4 py-8">
        <Header
          goVersion={state.status === "ready" ? state.goVersion : undefined}
        />
        <Separator />
        <main>
          {state.status === "loading" && (
            <p
              className="flex items-center gap-2 text-muted-foreground"
              role="status"
            >
              <LoaderCircle className="size-4 animate-spin" /> Loading the
              WebAssembly module…
            </p>
          )}
          {state.status === "error" && (
            <Card className="border-destructive" role="alert">
              <CardHeader>
                <CardTitle>The WebAssembly module did not load</CardTitle>
                <CardDescription>{state.message}</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" size="sm" onClick={retry}>
                  Try again
                </Button>
              </CardContent>
            </Card>
          )}
          {state.status === "ready" && (
            <Tabs defaultValue={DEMOS[0].key}>
              <TabsList className="h-auto flex-wrap">
                {DEMOS.map(({ key, label }) => (
                  <TabsTrigger key={key} value={key}>
                    {label}
                  </TabsTrigger>
                ))}
              </TabsList>
              {DEMOS.map(({ key, Demo }) => (
                <TabsContent key={key} value={key} className="pt-2">
                  <Demo />
                </TabsContent>
              ))}
            </Tabs>
          )}
        </main>
      </div>
    </TooltipProvider>
  );
}
