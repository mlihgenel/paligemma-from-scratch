// Bir sayfadaki n. scrolly'nin her adımında figürün görüntüsü: node e2e/steps.mjs <klasör> <yol> [scrollyIdx=0] [--dark]
import { chromium } from "@playwright/test";
const [out, path, idx = "0", ...rest] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: rest.includes("--dark") ? "dark" : "light" });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
page.on("console", (m) => m.type() === "error" && errs.push(m.text()));
await page.goto("http://localhost:5173/#" + path, { waitUntil: "networkidle" });
await page.waitForTimeout(1200);
const sc = page.locator(".scrolly").nth(+idx);
const n = await sc.locator(".scrolly-step").count();
for (let i = 0; i < n; i++) {
  await sc.locator(".scrolly-step").nth(i).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
  await page.waitForTimeout(2200);
  await page.screenshot({ path: `${out}/st_${path.replace(/\//g, "_")}_${idx}_${i}.png` });
}
console.log(path, n, "adım", errs.length ? errs.join("\n") : "(hata yok)");
await browser.close();
