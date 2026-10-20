import { create } from "zustand";

type Theme = "auto" | "light" | "dark";

interface State {
  theme: Theme;
  setTheme: (t: Theme) => void;
  imageIdx: number;
  setImageIdx: (i: number) => void;
  promptIdx: number;
  setPromptIdx: (i: number) => void;
  realDims: boolean;               // gerçek boyut / oyuncak boyut
  setRealDims: (b: boolean) => void;
}

const saved = (() => { try { return (localStorage.getItem("theme") as Theme) || "auto"; } catch { return "auto"; } })();

export const useStore = create<State>((set) => ({
  theme: saved,
  setTheme: (theme) => {
    try { localStorage.setItem("theme", theme); } catch { /* yok say */ }
    set({ theme });
  },
  imageIdx: 0, setImageIdx: (imageIdx) => set({ imageIdx }),
  promptIdx: 0, setPromptIdx: (promptIdx) => set({ promptIdx }),
  realDims: true, setRealDims: (realDims) => set({ realDims }),
}));
