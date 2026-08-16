import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { injectSeoShell } from "./seoShell.js";

export default defineConfig(({ command }) => ({
  plugins: [
    react(),
    {
      name: "shinonome-seo-shell",
      transformIndexHtml: {
        order: "pre",
        handler: injectSeoShell,
      },
    },
  ],
  define:
    command === "build"
      ? { "process.env.NODE_ENV": JSON.stringify("production") }
      : undefined,
  server: {
    host: "127.0.0.1",
    port: 8081,
    strictPort: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.js",
  },
  build: {
    lib: {
      entry: resolve(import.meta.dirname, "src/main.jsx"),
      formats: ["es"],
      fileName: "homepage",
      cssFileName: "homepage",
    },
    outDir: resolve(import.meta.dirname, "../web/home-assets"),
    emptyOutDir: true,
    minify: "oxc",
  },
}));
