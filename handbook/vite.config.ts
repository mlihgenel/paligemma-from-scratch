import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  // GitHub Pages serves the site under /<repo>/: scripts/deploy.sh sets HANDBOOK_BASE for the build
  base: process.env.HANDBOOK_BASE ?? "/",
  plugins: [
    { enforce: "pre", ...mdx({ remarkPlugins: [remarkMath, remarkGfm], rehypePlugins: [rehypeKatex], providerImportSource: "@mdx-js/react" }) },
    react({ include: /\.(mdx|tsx?)$/ }),
  ],
  // ../src/*.py files are read with ?raw: the book shows the project's code live
  server: { fs: { allow: [".."] } },
  test: { include: ["src/tests/**/*.test.ts"] },
} as any);
