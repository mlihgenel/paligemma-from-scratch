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

for (const p of ["/goruntu", "/encoder", "/oyun-alani", "/hepsi", "/gemma", "/gemma/akis", "/gemma/sqrt-d", "/gemma/geglu", "/gemma/katman", "/gemma/tying", "/gemma/dogrulama"]) {
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

test("6: masked_scatter adımında görsel hücre hover'ı hedefini söylüyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/hepsi", { waitUntil: "networkidle" });
  const sc = page.locator(".scrolly").first();
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

test("4: kutu çizince <loc> token'ları güncelleniyor", async ({ page }) => {
  await page.goto("http://localhost:5173/#/kopru", { waitUntil: "networkidle" });
  const fig = page.locator(".fig").filter({ hasText: "Bir kutu çiz" });
  await fig.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1000);
  const img = fig.locator("div[style*='crosshair']").first();
  const b = (await img.boundingBox())!;
  await page.mouse.move(b.x + 1, b.y + 1);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 5 });
  await page.mouse.up();
  await expect(fig.getByText(/<loc000\d><loc000\d><loc05\d\d><loc05\d\d>/)).toBeVisible();
});
