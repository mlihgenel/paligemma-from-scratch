// "Bu sayfada" listesinin verisi: sayfadaki h2 başlıkları ve kaydırırken hangisinde olduğumuz.
// Durum küçük bir store'da duruyor: PageView (ve içindeki ağır widget'lar) kaydırırken yeniden çizilmesin.
import { useEffect } from "react";
import { create } from "zustand";

export interface Section { id: string; text: string }

interface SectionsState {
  sections: Section[];
  active: string;     // üst çubuğun altından geçmiş son başlığın id'si, hiçbiri geçmediyse ""
}

export const useSectionsStore = create<SectionsState>(() => ({ sections: [], active: "" }));

/** Başlık metninden id: Türkçe harfler sadeleşir ("Neden 1/√head_dim?" → "neden-1-head-dim"). Boş kalabilir. */
export function slug(text: string): string {
  return text
    .toLocaleLowerCase("tr")                 // İ → i, I → ı
    .normalize("NFD").replace(/[̀-ͯ]/g, "")   // ş → s, ğ → g, ü → u, ö → o, ç → c
    .replace(/ı/g, "i")                      // ı'nın ayrışımı yok
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** base çakışıyorsa -2, -3 ... ekler */
export function uniqueId(base: string, taken: (id: string) => boolean): string {
  let id = base, n = 2;
  while (taken(id)) id = `${base}-${n++}`;
  return id;
}

/** Üst çubuk 52 px. Başlığın üstü bu çizgiyi geçince "o bölümdeyiz" sayıyoruz */
const ACTIVE_LINE = 120;

const HEADINGS = "article.prose h2";

/** Sayfa değişince başlıkları toplar, id atar ve kaydırmayı izler. PageView'da bir kez çağrılır, state tutmaz. */
export function useSectionsTracker(path: string) {
  useEffect(() => {
    const set = useSectionsStore.setState;
    set({ sections: [], active: "" });
    let last = "", timer = 0, raf = 0;

    const headings = () => Array.from(document.querySelectorAll<HTMLElement>(HEADINGS)).filter((h) => !h.closest(".fig"));

    // MDX tembel yükleniyor ve widget'lar DOM'u değiştiriyor: başlıkları her değişiklikte yeniden topluyoruz (ucuz)
    const collect = () => {
      const hs = headings();
      const taken = new Set(hs.map((h) => h.id).filter(Boolean));
      const list = hs.map((h) => {
        const text = (h.textContent || "").trim();
        if (!h.id) {
          h.id = uniqueId(slug(text) || "bolum", (id) => taken.has(id) || !!document.getElementById(id));
          taken.add(h.id);
        }
        return { id: h.id, text };
      });
      const key = list.map((s) => `${s.id}\u0000${s.text}`).join("\u0001");
      if (key !== last) { last = key; set({ sections: list }); }
    };

    const tick = () => {
      raf = 0;
      let active = "";
      for (const h of headings()) { if (h.getBoundingClientRect().top <= ACTIVE_LINE) active = h.id; else break; }

      if (useSectionsStore.getState().active !== active) set({ active });
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(tick); };

    const article = document.querySelector("article.prose");
    const mo = new MutationObserver(() => {
      clearTimeout(timer);
      timer = window.setTimeout(() => { collect(); schedule(); }, 120);
    });
    if (article) mo.observe(article, { childList: true, subtree: true });
    collect(); schedule();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      mo.disconnect(); clearTimeout(timer); if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("scroll", schedule); window.removeEventListener("resize", schedule);
    };
  }, [path]);
}

/** Bölüme kaydır ve adresi paylaşılabilir yap: #/yol#bolum (router bu biçimi zaten anlıyor) */
export function goToSection(id: string, pathname: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  history.replaceState(null, "", `#${pathname}#${id}`);
}
