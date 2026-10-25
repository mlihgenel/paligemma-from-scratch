// Viewport screenshots at several scroll positions: node e2e/scroll.mjs <dir> <path> <y1,y2,...> [--dark] [--w=1440]
import { chromium } from "@playwright/test";
const [out, path, ys, ...rest] = process.argv.slice(2);
const dark = rest.includes("--dark");
const w = +(rest.find((r) => r.startsWith("--w="))?.slice(4) ?? 1440);
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: w, height: 900 }, colorScheme: dark ? "dark" : "light" });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
await page.goto("http://localhost:5173/#" + path, { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
for (const y of ys.split(",")) {
  await page.evaluate((y) => window.scrollTo(0, +y), y);
  await page.waitForTimeout(1200);
  const name = `${out}/s_${path.replace(/\//g, "_")}_${y}${dark ? "_d" : ""}_${w}.png`;
  await page.screenshot({ path: name });
  console.log(name);
}
console.log(errs.length ? errs.join("\n") : "(hata yok)");
await browser.close();
