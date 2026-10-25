// Two languages: English (default) and Turkish. The choice lives in the store and in localStorage;
// `?lang=tr` in the URL (before the #) overrides it, which makes a language easy to link to and to test.
import { ReactNode, useEffect } from "react";
import { useStore } from "./store";

export type Lang = "en" | "tr";
export const LANGS: Lang[] = ["en", "tr"];

/** A value in both languages */
export type L<T = ReactNode> = { en: T; tr: T };

export const useLang = () => useStore((s) => s.lang);

/** t("English", "Türkçe"): picks the string (or node) for the current language */
export function useT() {
  const lang = useLang();
  return <T extends ReactNode>(en: T, tr: T): T => (lang === "tr" ? tr : en);
}

export const pick = <T,>(l: L<T>, lang: Lang): T => l[lang];

/** Keeps <html lang> and the document title in sync with the chosen language */
export function useDocumentLang() {
  const lang = useLang();
  useEffect(() => {
    document.documentElement.lang = lang;
    document.title = lang === "tr" ? "PaliGemma El Kitabı" : "PaliGemma Handbook";
  }, [lang]);
}
