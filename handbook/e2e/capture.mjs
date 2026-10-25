// Records short interactions from the running handbook and turns them into GIFs for the README.
// Run (with `npm run dev` running): node e2e/capture.mjs [name ...]
// Output: ../assets/<name>.gif (needs ffmpeg on PATH)
import { chromium } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import path from "node:path";

const BASE = "http://localhost:5173/?lang=en#";
const OUT = path.resolve("../assets");
const TMP = path.resolve("test-results/capture");
const W = 1280, H = 820;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** Each scene: page, element to frame, actions to record, output width */
const SCENES = {
  // The landing animation: plays by itself once visible
  architecture: {
    path: "/", target: ".arch-full", width: 960,
    run: async () => { await wait(15500); },
  },
  // The KV cache stepper: prefill, a few cached steps, then the same step without the cache
  "kv-cache": {
    path: "/gemma/kv-cache", target: "Generating", width: 860,
    run: async (page, el) => {
      const next = el.getByRole("button", { name: "→" });
      await wait(1200);
      for (let i = 0; i < 4; i++) { await next.click(); await wait(1100); }
      await el.getByRole("button", { name: "without cache" }).click(); await wait(1800);
      await el.getByRole("button", { name: "with KV cache" }).click(); await wait(1500);
    },
  },
  // RoPE: shifting both positions rotates the arrows but keeps the dot product
  rope: {
    path: "/gemma/rope", target: "One pair rotating", width: 820,
    run: async (page, el) => {
      await wait(1000);
      const plus = el.getByRole("button", { name: "shift both +1" });
      for (let i = 0; i < 6; i++) { await plus.click(); await wait(550); }
      await wait(800);
    },
  },
  // Attention weights from the real GemmaAttention, with and without the causal mask
  attention: {
    path: "/gemma/attention", target: "Weights from the project", width: 860,
    run: async (page, el) => {
      await wait(1200);
      await el.getByRole("button", { name: "no mask" }).click(); await wait(1800);
      await el.getByRole("button", { name: "causal mask on" }).click(); await wait(1800);
    },
  },
  // The end-to-end tour in free mode: step through the stops
  "end-to-end": {
    path: "/end-to-end", target: "Free exploration", width: 900,
    run: async (page, el) => {
      const slider = el.locator("input[type=range]");
      await slider.fill("0"); await wait(1500);
      for (const s of [2, 3, 4, 6, 7, 8, 10]) { await slider.fill(String(s)); await wait(1500); }
    },
  },
};

async function record(name, scene) {
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, colorScheme: "light", recordVideo: { dir: TMP, size: { width: W, height: H } } });
  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto(BASE + scene.path, { waitUntil: "networkidle" });
  // A CSS selector, or the start of a figure caption
  const el = scene.target.startsWith(".") ? page.locator(scene.target).first() : page.locator(".fig").filter({ hasText: scene.target }).first();
  await el.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -40));
  await wait(1500);
  const box = await el.boundingBox();
  const start = (Date.now() - t0) / 1000;
  await scene.run(page, el);
  const dur = (Date.now() - t0) / 1000 - start;
  const video = page.video();
  await ctx.close();
  await browser.close();
  const webm = path.join(TMP, `${name}.webm`);
  renameSync(await video.path(), webm);

  const x = Math.max(0, Math.round(box.x)), y = Math.max(0, Math.round(box.y));
  const w = Math.min(W - x, Math.round(box.width)), h = Math.min(H - y, Math.round(box.height));
  const filter = `crop=${w}:${h}:${x}:${y},fps=12,scale=${scene.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`;
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", String(start), "-t", String(dur), "-i", webm, "-vf", filter, path.join(OUT, `${name}.gif`)]);
  console.log(`${name}: ${dur.toFixed(1)} s, crop ${w}×${h} -> assets/${name}.gif`);
}

mkdirSync(OUT, { recursive: true });
rmSync(TMP, { recursive: true, force: true });
mkdirSync(TMP, { recursive: true });
const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(SCENES);
for (const n of names) await record(n, SCENES[n]);
for (const f of readdirSync(TMP)) if (f.endsWith(".webm") && !names.includes(f.replace(".webm", ""))) rmSync(path.join(TMP, f));
