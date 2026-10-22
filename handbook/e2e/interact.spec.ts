// Etkileşim testleri: scrolly adımları, hücre hover'ı, 3D hover.
import { expect, test } from "@playwright/test";

test("5.3 scrolly: kaydırınca adım ve figür değişiyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rmsnorm", { waitUntil: "networkidle" });
  const steps = page.locator(".scrolly").first().locator(".scrolly-step");
  await steps.nth(0).scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  await expect(page.locator(".scrolly").first().locator(".scrolly-fig code").first()).toHaveText("x");
  await steps.nth(2).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
  await page.waitForTimeout(800);
  await expect(steps.nth(2)).toHaveClass(/on/);
  await expect(page.locator(".scrolly").first().locator(".scrolly-fig code").first()).toHaveText(".mean(-1, keepdim=True)");
});

test("5.3 laboratuvar: hücreye gelince hesap görünüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rmsnorm", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "RMSNorm laboratuvarı" });
  await fig.scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  const cells = fig.locator("div[style*='position: absolute']").filter({ hasText: /^-?\d/ });
  await cells.nth(1).hover();
  await expect(page.locator(".tip").filter({ hasText: "rsqrt" }).first()).toBeVisible();
});

test("5.3 3D: küpün üstüne gelince değer görünüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rmsnorm", { waitUntil: "networkidle" });
  const sc = page.locator(".scrolly").nth(1);
  await sc.locator(".scrolly-step").nth(0).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
  await page.waitForTimeout(2500);
  const canvas = sc.locator("canvas");
  const box = (await canvas.boundingBox())!;
  let found = false;
  for (let dx = -60; dx <= 60 && !found; dx += 20) {
    await page.mouse.move(box.x + box.width / 2 + dx, box.y + box.height / 2);
    await page.waitForTimeout(250);
    found = (await page.locator(".tip").filter({ hasText: "x[b=" }).count()) > 0;
  }
  expect(found).toBe(true);
});

for (const p of ["/goruntu", "/encoder", "/uctan-uca", "/kopru", "/gemma", "/gemma/akis", "/gemma/sqrt-d", "/gemma/geglu", "/gemma/katman", "/gemma/tying", "/gemma/attention"]) {
  test(`${p} scrolly: son adıma kadar ilerliyor`, async ({ page }) => {
    const errs: string[] = [];
    page.on("pageerror", (e) => errs.push(e.message));
    await page.goto("http://localhost:5173/#" + p, { waitUntil: "networkidle" });
    const steps = page.locator(".scrolly").first().locator(".scrolly-step");
    const n = await steps.count();
    expect(n).toBeGreaterThan(2);
    await steps.nth(n - 1).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
    await page.waitForTimeout(900);
    await expect(steps.nth(n - 1)).toHaveClass(/on/);
    expect(errs).toEqual([]);
  });
}

test("3: masked_scatter adımında görsel hücre hover'ı hedefini söylüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/kopru", { waitUntil: "networkidle" });
  const sc = page.locator(".scrolly").filter({ hasText: "_merge_input_ids_with_image_features" }).first();
  await sc.locator(".scrolly-step").nth(5).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
  await page.waitForTimeout(2500);
  const cell = sc.locator(".scrolly-fig div[style*='position: absolute']").filter({ hasText: /^-?\d/ });
  const n = await cell.count();
  let ok = false;
  for (let i = n - 1; i >= 0 && !ok; i--) {
    await cell.nth(i).hover({ force: true });
    ok = (await page.locator(".tip").filter({ hasText: "scaled_image_features" }).count()) > 0;
  }
  expect(ok).toBe(true);
});

test("2: yama gezgini hover ile yama ve token değişiyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/goruntu", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Yama gezgini" });
  await fig.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1500);
  const img = fig.locator("div[style*='crosshair']").first();
  const b = (await img.boundingBox())!;
  await page.mouse.move(b.x + b.width * 0.1, b.y + b.height * 0.1);
  await expect(fig.getByText("yama (1, 1) → token 17")).toBeVisible();
});

test("3: kutu çizince <loc> token'ları güncelleniyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/kopru", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Bir kutu çiz" });
  await fig.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1000);
  const img = fig.locator("div[style*='crosshair']").first();
  const b = (await img.boundingBox())!;
  await page.mouse.move(b.x + 1, b.y + 1);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 5 });
  // sürüklerken kutunun köşelerinde token'lar görünüyor
  await expect(fig.getByTestId("loc-tl")).toHaveText(/<loc000\d><loc000\d>/);
  await expect(fig.getByTestId("loc-br")).toHaveText(/<loc05\d\d><loc05\d\d>/);
  await page.mouse.up();
  await expect(fig.locator(".mono").filter({ hasText: "tabela" }).first()).toHaveText(/<loc000\d><loc000\d><loc05\d\d><loc05\d\d> tabela/);
});

test("açılış: mimari animasyonu oynuyor, adıma tıklanıyor, kutu bölüme götürüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/", { waitUntil: "networkidle" });
  const arch = page.locator(".arch-full");
  await expect(arch.locator(".arch-step-n")).toHaveText("1");
  await page.waitForTimeout(3000);
  await expect(arch.locator(".arch-step-n")).not.toHaveText("1");
  await arch.getByRole("tab", { name: "Gemma" }).click();
  await expect(arch.locator(".arch-step-n")).toHaveText("5");
  await arch.locator("[aria-label=siglip]").click();
  await expect(page).toHaveURL(/#\/buyuk-resim$/);
  await expect(page.locator(".arch-mini")).toBeVisible();
});

test("2.2: filtre seçimi hover'da değil tıklamada değişiyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/goruntu", { waitUntil: "networkidle" });
  const gal = page.getByRole("listbox", { name: "filtre galerisi" });
  await gal.scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  const opts = gal.getByRole("option");
  await opts.nth(5).click();
  await opts.nth(20).hover();
  await expect(opts.nth(5)).toHaveAttribute("aria-selected", "true");
  await expect(opts.nth(20)).toHaveAttribute("aria-selected", "false");
  await opts.nth(20).click();
  await expect(opts.nth(20)).toHaveAttribute("aria-selected", "true");
});

test("4.4: laboratuvarda x × 10 çıktıyı değiştirmiyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rmsnorm", { waitUntil: "networkidle" });
  const btn = page.getByRole("button", { name: "x × 10" });
  await btn.scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  const root = btn.locator("xpath=../../..");
  // çıktı bloğu "=" işaretinin sağında: en sağdaki 32 sayı
  const outs = () => root.evaluate((el) => {
    const cells = [...el.querySelectorAll("div")].filter((d) => d.style.position === "absolute" && /^-?\d/.test(d.textContent ?? "") && d.children.length === 0);
    const xs = [...new Set(cells.map((d) => parseFloat(d.style.left)))].sort((a, b) => a - b);
    const right = xs.slice(-8);
    return cells.filter((d) => right.includes(parseFloat(d.style.left))).map((d) => d.textContent).join(",");
  });
  const before = await outs();
  await btn.click();
  await page.waitForTimeout(900);
  expect(await outs()).toBe(before);
});

// --- 4.9–4.11: attention sayfaları ---
const stat = (root: import("@playwright/test").Locator, label: string) =>
  root.locator(".stat").filter({ hasText: label }).locator(".val");

test("4.9 RoPE: iki konumu birlikte kaydırınca q·k değişmiyor, ayrı kaydırınca değişiyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rope", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Bir çiftin dönüşü" });
  await fig.scrollIntoViewIfNeeded();
  await expect(stat(fig, "q · k")).toBeVisible();
  const before = await stat(fig, "q · k").textContent();
  await fig.getByRole("button", { name: "ikisini +1 kaydır" }).click();
  await fig.getByRole("button", { name: "ikisini +1 kaydır" }).click();
  await expect(stat(fig, "m − n")).toHaveText("3");
  expect(await stat(fig, "q · k").textContent()).toBe(before);
  await fig.getByLabel("m", { exact: true }).fill("20");
  await expect(stat(fig, "m − n")).not.toHaveText("3");
  expect(await stat(fig, "q · k").textContent()).not.toBe(before);
});

test("4.9 RoPE: boyut düzeni, üstüne gelince eşi söyleniyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/rope", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Aynı döndürme, iki düzen" });
  await fig.scrollIntoViewIfNeeded();
  await fig.getByText("0", { exact: true }).first().hover();
  await expect(fig.getByText(/Boyut 0 ile boyut 8 aynı 2B düzlemde/)).toBeVisible();
  await fig.getByRole("button", { name: /komşu çiftler/ }).click();
  await fig.getByText("0", { exact: true }).first().hover();
  await expect(fig.getByText(/Boyut 0 ile boyut 1 aynı 2B düzlemde/)).toBeVisible();
});

test("4.10 MQA: düğmeler head eşlemesini değiştiriyor, KV cache boyutu T ile büyüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/mqa", { waitUntil: "networkidle" });
  const map = page.locator(".fig").filter({ hasText: "Hangi sorgu head'i" });
  await map.scrollIntoViewIfNeeded();
  await expect(map.getByText("n_rep = 8 / 1 = 8")).toBeVisible();
  await map.getByRole("button", { name: /GQA · 2 KV/ }).click();
  await expect(map.getByText("n_rep = 8 / 2 = 4")).toBeVisible();
  const cache = page.locator(".fig").filter({ hasText: "KV cache, bf16" });
  await cache.scrollIntoViewIfNeeded();
  await expect(cache.getByText("4.6 MiB").first()).toBeVisible();
  await cache.getByRole("button", { name: /8192/ }).click();
  await expect(cache.getByText("144.0 MiB").first()).toBeVisible();
});

test("4.10 MQA: yanlış eksen sırası Q head'lerini dönüşümlü bağlıyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/mqa", { waitUntil: "networkidle" });
  const box = page.locator(".breakit");
  await box.scrollIntoViewIfNeeded();
  await expect(box.getByText("Q0–Q3 → KV0, Q4–Q7 → KV1")).toBeVisible();
  await box.getByRole("switch").click();
  await expect(box.getByText("dönüşümlü")).toBeVisible();
});

test("4.11 attention: maske açıkken köşegen üstü 0, maskesizken gelecek sızıyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/attention", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Senin attention'ından çıkan ağırlıklar" });
  await fig.scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  expect(Number(await stat(fig, "köşegen üstü en büyük ağırlık").textContent())).toBe(0);
  expect(Number(await stat(fig, "aynı deney, maskesiz").textContent())).toBeGreaterThan(1);
  await expect(stat(fig, "son token'ı değiştirince önceki çıktılar, maskeli")).toHaveText(/^0\.0e\+0$|^0e\+0$/);
  await fig.getByRole("button", { name: "maske yok" }).click();
  await expect(fig.getByRole("button", { name: "maske yok" })).toHaveAttribute("aria-pressed", "true");
});

test("4.11 attention: maske ızgarasında satıra gelince görülebilenler söyleniyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/attention", { waitUntil: "networkidle" });
  const grid = page.getByTestId("mask-grid");
  await grid.scrollIntoViewIfNeeded();
  await grid.getByText("q3", { exact: true }).hover();
  await expect(page.getByText("yalnızca k0 … k3'ı görebiliyor (4 token)")).toBeVisible();
});

test("4.11 attention: ölçeksiz softmax head_dim 256'da tek token'a sıkışıyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/attention", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Aynı vektörler, ölçekli ve ölçeksiz" });
  await fig.scrollIntoViewIfNeeded();
  const max = async () => Number((await stat(fig, "en büyük ağırlık").textContent())!.replace("%", ""));
  const scaled = await max();
  await fig.getByRole("button", { name: "ölçeksiz" }).click();
  const raw = await max();
  expect(raw).toBeGreaterThan(scaled);
  expect(raw).toBeGreaterThan(80);
});

test("4.3 √d: eski yol adımı etiketli ve bf16 gidiş-dönüşte değişen değer oranını gösteriyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/sqrt-d", { waitUntil: "networkidle" });
  const sc = page.locator(".scrolly").first();
  const steps = sc.locator(".scrolly-step");
  await steps.nth(2).evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.4));
  await page.waitForTimeout(800);
  await expect(steps.nth(2)).toHaveClass(/on/);
  const fig = sc.locator(".scrolly-fig");
  await expect(fig).toContainText("Eski yol");
  const val = fig.locator(".stat").filter({ hasText: "gidiş-dönüşte değişen değer" }).locator(".val");
  await expect(val).toHaveText(/^%\d/);
  // float32 satırında fark ~0, bf16 satırı kaydırıcıyla değişiyor
  await fig.locator("input[type=range]").fill("5");
  await expect(fig.getByText("× 45.25 → bf16")).toBeVisible();
});

test("4.3 √d: sayfada artık görsel token'ların ÷√d'si anlatılmıyor, yalnızca metin ×√d", async ({ page }) => {
  await page.goto("http://localhost:5173/#/gemma/sqrt-d", { waitUntil: "networkidle" });
  const body = page.locator("main, article, body").first();
  await expect(body).toContainText("Görsel token'lar: neden çarpılmıyor?");
  await expect(body).toContainText("scale_embeddings=False");
  await expect(body).not.toContainText("neden önce bölüyoruz");
  // Düzeltmeden sonraki gerçek model ölçümü sayfada yazıyor
  await expect(body).toContainText("0.195");
  await expect(body).toContainText("%100");
  await expect(body).not.toContainText("Colab'da yeniden koşu bekliyor");
});
