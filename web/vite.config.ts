import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  // Served from https://cwbudde.github.io/algo-tensor/; the dev server and
  // the preview use the same base.
  base: "/algo-tensor/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  // Module workers (the benchmark runs the wasm in one) share the ES module
  // loader with the main thread.
  worker: {
    format: "es",
  },
  build: {
    // React, Radix and recharts make one ~700 kB chunk; demo.wasm is larger.
    chunkSizeWarningLimit: 1024,
  },
});
