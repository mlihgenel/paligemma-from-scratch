import { create } from "zustand";
import type { Lang } from "./i18n";

/** Language at startup: `?lang=` in the URL, then the saved choice, then English */
function initialLang(): Lang {
  try {
    const q = new URLSearchParams(window.location.search).get("lang");
    if (q === "en" || q === "tr") return q;
    const saved = localStorage.getItem("lang");
    if (saved === "en" || saved === "tr") return saved;
  } catch { /* no window or storage: fall through */ }
  return "en";
}

type Theme = "auto" | "light" | "dark";

interface State {
  theme: Theme;
  setTheme: (t: Theme) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
  imageIdx: number;
  setImageIdx: (i: number) => void;
  promptIdx: number;
  setPromptIdx: (i: number) => void;
  realDims: boolean;               // real size / toy size
  setRealDims: (b: boolean) => void;
}

const saved = (() => { try { return (localStorage.getItem("theme") as Theme) || "auto"; } catch { return "auto"; } })();

export const useStore = create<State>((set) => ({
  theme: saved,
  setTheme: (theme) => {
    try { localStorage.setItem("theme", theme); } catch { /* ignore */ }
    set({ theme });
  },
  lang: initialLang(),
  setLang: (lang) => {
    try { localStorage.setItem("lang", lang); } catch { /* ignore */ }
    set({ lang });
  },
  imageIdx: 0, setImageIdx: (imageIdx) => set({ imageIdx }),
  promptIdx: 0, setPromptIdx: (promptIdx) => set({ promptIdx }),
  realDims: true, setRealDims: (realDims) => set({ realDims }),
}));
