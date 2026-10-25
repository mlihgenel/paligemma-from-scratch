// Screenshot + console errors: node e2e/shot.mjs <out dir> <path> [path...] [--dark] [--mobile] [--full]
import { chromium } from "@playwright/test";
const args = process.argv.slice(2);
const out = args[0];
const flags = new Set(args.filter((a) => a.startsWith("--")));
const paths = args.slice(1).filter((a) => !a.startsWith("--"));
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const ctx = await browser.newContext({
  viewport: flags.has("--mobile") ? { width: 390, height: 844 } : { width: 1440, height: 900 },
  deviceScaleFactor: 1, colorScheme: flags.has("--dark") ? "dark" : "light",
});
for (const p of paths) {
  const page = await ctx.newPage();
  const errs = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errs.push(`[${m.type()}] ${m.text()}`); });
  page.on("pageerror", (e) => errs.push("[pageerror] " + e.message));
  await page.goto("http://localhost:5173/#" + p, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  const name = (p.replace(/\//g, "_") || "_home") + (flags.has("--dark") ? "-dark" : "") + (flags.has("--mobile") ? "-m" : "");
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: flags.has("--full") });
  console.log(p, "->", `${out}/${name}.png`, errs.length ? "\n  " + errs.slice(0, 8).join("\n  ") : "(no errors)");
  await page.close();
}
await browser.close();
