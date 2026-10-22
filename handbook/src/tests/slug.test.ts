import { describe, expect, it } from "vitest";
import { slug, uniqueId } from "../lib/useSections";

describe("slug", () => {
  it("Türkçe harfleri sadeleştirir", () => {
    expect(slug("Adım adım, gerçek boyutlarla")).toBe("adim-adim-gercek-boyutlarla");
    expect(slug("İçindekiler: şüphe, öğüt, çığlık")).toBe("icindekiler-suphe-ogut-ciglik");
    expect(slug("IŞIK")).toBe("isik");   // büyük I → ı → i
  });
  it("sembolleri ve baştaki/sondaki tireleri atar", () => {
    expect(slug("Neden 1/√head_dim?")).toBe("neden-1-head-dim");
    expect(slug("  ?? Kodda nerede? ")).toBe("kodda-nerede");
  });
  it("sadece sembolden oluşan başlık boş döner (çağıran yedek ad verir)", () => {
    expect(slug("√ ?")).toBe("");
  });
});

describe("uniqueId", () => {
  it("çakışmazsa aynen döner", () => {
    expect(uniqueId("tek-bakista", () => false)).toBe("tek-bakista");
  });
  it("çakışırsa -2, -3 ekler", () => {
    const taken = new Set(["bolum", "bolum-2"]);
    expect(uniqueId("bolum", (id) => taken.has(id))).toBe("bolum-3");
  });
});
