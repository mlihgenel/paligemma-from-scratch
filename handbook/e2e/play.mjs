// Screenshots of chosen stops of the tour: node e2e/play.mjs <dir> 0 2 3 8 12
import { chromium } from "@playwright/test";
const [out, ...stages] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
page.on("console", (m) => m.type() === "error" && errs.push(m.text()));
await page.goto("http://localhost:5173/#/oyun-alani", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
const fig = page.locator(".fig").first();
await fig.scrollIntoViewIfNeeded();
const slider = page.locator('.fig input[type="range"]').first();
for (const s of stages) {
  await slider.fill(String(s));
  await page.waitForTimeout(2600);
  await fig.screenshot({ path: `${out}/play_${s}.png` });
}
console.log(errs.length ? errs.join("\n") : "(hata yok)");
await browser.close();
