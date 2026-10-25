// Every page in both languages: no console errors, code anchors found, data, 3D canvases and videos load.
// Run: npm run dev (in another terminal), then npx playwright test
import { expect, test } from "@playwright/test";

const PAGES = ["/", "/end-to-end", "/big-picture", "/patches", "/encoder", "/bridge", "/gemma", "/gemma/residual", "/gemma/sqrt-d", "/gemma/rmsnorm", "/gemma/geglu", "/gemma/layer", "/gemma/model", "/gemma/tying", "/gemma/rope", "/gemma/mqa", "/gemma/attention", "/gemma/kv-cache"];

for (const lang of ["en", "tr"]) {
  for (const p of PAGES) {
    test(`page ${p} (${lang})`, async ({ page }) => {
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(e.message));
      page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
      await page.goto(`http://localhost:5173/?lang=${lang}#${p}`, { waitUntil: "networkidle" });
      await page.waitForTimeout(1500);
      expect(errors, errors.join("\n")).toEqual([]);
      await expect(page.locator("html")).toHaveAttribute("lang", lang);
      // CodeWalk symbols and note anchors must exist in src/ (this breaks when src changes)
      for (const text of ["not found", "bulunamadı", "The code may have changed", "Kodun değişmiş olabilir", "Could not load data", "Veri yüklenemedi"]) {
        await expect(page.getByText(text, { exact: false })).toHaveCount(0);
      }
      for (const v of await page.locator("video").all()) {
        await expect.poll(() => v.evaluate((el: HTMLVideoElement) => el.readyState)).toBeGreaterThan(0);
      }
    });
  }
}

// Old addresses (including the earlier Turkish slugs) redirect, keeping the section anchor
for (const [from, to] of [["/oyun-alani", "/end-to-end"], ["/hepsi", "/bridge"], ["/gemma/sina", "/gemma"], ["/kopru", "/bridge"], ["/gemma/akis", "/gemma/residual"], ["/kopru#merge", "/bridge#merge"]]) {
  test(`redirect ${from} → ${to}`, async ({ page }) => {
    await page.goto("http://localhost:5173/#" + from, { waitUntil: "networkidle" });
    await expect(page).toHaveURL(new RegExp("#" + to.replace("#", "#") + "$"));
  });
}

test("language switch: EN by default, TR from the toggle, remembered after reload", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rope", { waitUntil: "networkidle" });
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("RoPE: position by rotation");
  await page.locator(".lang-switch").getByRole("button", { name: "TR" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("RoPE: konum, dönerek");
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.locator("html")).toHaveAttribute("lang", "tr");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("RoPE: konum, dönerek");
});

test("KV cache stepper: one token per step with the cache, the whole sequence without", async ({ page }) => {
  await page.goto("http://localhost:5173/?lang=en#/gemma/kv-cache", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Generating" }).first();
  const stat = (label: string) => fig.locator(".stat").filter({ hasText: label }).locator(".val");
  await expect(stat("tokens processed in this step")).toHaveText("260");
  await fig.getByRole("button", { name: "→" }).click();
  await expect(stat("tokens processed in this step")).toHaveText("1");
  await expect(stat("position of the newest token")).toHaveText("261");
  await fig.getByRole("button", { name: "without cache" }).click();
  await expect(stat("tokens processed in this step")).toHaveText("261");
  await expect(stat("SigLIP runs so far")).toHaveText("2");
});
