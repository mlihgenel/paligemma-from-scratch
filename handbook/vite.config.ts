import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  plugins: [
    { enforce: "pre", ...mdx({ remarkPlugins: [remarkMath, remarkGfm], rehypePlugins: [rehypeKatex], providerImportSource: "@mdx-js/react" }) },
    react({ include: /\.(mdx|tsx?)$/ }),
  ],
  // ../src/*.py dosyaları ?raw ile okunuyor: kitap senin kodunu canlı gösteriyor
  server: { fs: { allow: [".."] } },
  test: { include: ["src/tests/**/*.test.ts"] },
} as any);
