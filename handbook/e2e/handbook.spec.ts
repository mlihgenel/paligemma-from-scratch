// Her hazır sayfa: konsol hatası yok, kod parçaları bulunuyor, 3D canvas'lar ve videolar yükleniyor.
// Çalıştır: npm run dev (ayrı terminalde), sonra npx playwright test
import { expect, test } from "@playwright/test";

const PAGES = ["/", "/uctan-uca", "/buyuk-resim", "/goruntu", "/encoder", "/kopru", "/gemma", "/gemma/akis", "/gemma/sqrt-d", "/gemma/rmsnorm", "/gemma/geglu", "/gemma/katman", "/gemma/model", "/gemma/tying", "/gemma/rope", "/gemma/mqa", "/gemma/attention"];

for (const p of PAGES) {
  test(`sayfa ${p}`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    await page.goto("http://localhost:5173/#" + p, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    expect(errors, errors.join("\n")).toEqual([]);
    // CodeWalk sembolleri src/'de bulunmalı (src değişirse burada kırılır)
    await expect(page.getByText("bulunamadı", { exact: false }).filter({ hasText: "satır" })).toHaveCount(0);
    await expect(page.getByText("Kodun değişmiş olabilir")).toHaveCount(0);
    await expect(page.getByText("Veri yüklenemedi")).toHaveCount(0);
    for (const v of await page.locator("video").all()) {
      await expect.poll(() => v.evaluate((el: HTMLVideoElement) => el.readyState)).toBeGreaterThan(0);
    }
  });
}

// Eski adresler yeni sayfalara yönleniyor
for (const [from, to] of [["/oyun-alani", "/uctan-uca"], ["/hepsi", "/kopru"], ["/gemma/sina", "/gemma"]]) {
  test(`yönlendirme ${from} → ${to}`, async ({ page }) => {
    await page.goto("http://localhost:5173/#" + from, { waitUntil: "networkidle" });
    await expect(page).toHaveURL(new RegExp("#" + to + "$"));
  });
}
