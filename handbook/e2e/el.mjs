// Bir öğenin ekran görüntüsü: node e2e/el.mjs <çıktı.png> <yol> <css seçici> [sıra] [--dark] [--click=seçici]
import { chromium } from "@playwright/test";
const [out, path, sel, nth = "0", ...rest] = process.argv.slice(2);
const dark = rest.includes("--dark");
const clicks = rest.filter((r) => r.startsWith("--click=")).map((r) => r.slice(8));
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: dark ? "dark" : "light" });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
page.on("console", (m) => m.type() === "error" && errs.push(m.text()));
await page.goto("http://localhost:5173/#" + path, { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
for (const c of clicks) { await page.locator(c).first().click(); await page.waitForTimeout(800); }
const el = page.locator(sel).nth(+nth);
await el.scrollIntoViewIfNeeded();
await page.waitForTimeout(1800);
await el.screenshot({ path: out });
console.log(out, errs.length ? errs.join("\n") : "(hata yok)");
await browser.close();
