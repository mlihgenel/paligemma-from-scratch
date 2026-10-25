// Data for the "On this page" list: the page's h2 headings and which one we are in while scrolling.
// The state lives in a small store so PageView (and its heavy widgets) do not re-render on scroll.
import { useEffect } from "react";
import { create } from "zustand";

export interface Section { id: string; text: string }

interface SectionsState {
  sections: Section[];
  active: string;     // id of the last heading that passed under the top bar, "" if none has
}

export const useSectionsStore = create<SectionsState>(() => ({ sections: [], active: "" }));

/** Id from heading text: Turkish letters are simplified ("Neden 1/√head_dim?" → "neden-1-head-dim"). May be empty. */
export function slug(text: string): string {
  return text
    .toLocaleLowerCase("tr")                 // İ → i, I → ı
    .normalize("NFD").replace(/[̀-ͯ]/g, "")   // ş → s, ğ → g, ü → u, ö → o, ç → c
    .replace(/ı/g, "i")                      // ı has no decomposition
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Appends -2, -3 ... if base is taken */
export function uniqueId(base: string, taken: (id: string) => boolean): string {
  let id = base, n = 2;
  while (taken(id)) id = `${base}-${n++}`;
  return id;
}

/** The top bar is 52 px. Once a heading's top passes this line we count it as "in that section" */
const ACTIVE_LINE = 120;

const HEADINGS = "article.prose h2";

/** Collects the headings when the page changes, assigns ids and tracks scrolling. Called once in PageView, holds no state. */
export function useSectionsTracker(path: string, lang = "") {
  useEffect(() => {
    const set = useSectionsStore.setState;
    set({ sections: [], active: "" });
    let last = "", timer = 0, raf = 0;

    const headings = () => Array.from(document.querySelectorAll<HTMLElement>(HEADINGS)).filter((h) => !h.closest(".fig"));

    // MDX loads lazily and widgets change the DOM: collect the headings again on every change (cheap)
    const collect = () => {
      const hs = headings();
      const taken = new Set(hs.map((h) => h.id).filter(Boolean));
      const list = hs.map((h) => {
        const text = (h.textContent || "").trim();
        if (!h.id) {
          h.id = uniqueId(slug(text) || "section", (id) => taken.has(id) || !!document.getElementById(id));
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
  }, [path, lang]);
}

/** Scroll to a section and make the address shareable: #/path#section (the router already understands this form) */
export function goToSection(id: string, pathname: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  history.replaceState(null, "", `#${pathname}#${id}`);
}
