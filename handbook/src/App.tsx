import { Suspense, useEffect, useState } from "react";
import { HashRouter, Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { MDXProvider } from "@mdx-js/react";
import { PAGES, PART_TITLE, Page, Part, REDIRECTS, START_TITLE, partVars } from "./registry";
import { ArchDiagram } from "./interact/ArchDiagram";
import { useStore } from "./store";
import { LANGS, useDocumentLang, useLang, useT } from "./i18n";
import { goToSection, useSectionsStore, useSectionsTracker } from "./lib/useSections";

function useEffectiveTheme() {
  const theme = useStore((s) => s.theme);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => { document.documentElement.dataset.theme = theme === "auto" ? (mq.matches ? "dark" : "light") : theme; };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);
}

function useScrollProgress() {
  const [p, setP] = useState(0);
  useEffect(() => {
    const on = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      setP(h > 0 ? Math.min(1, window.scrollY / h) : 0);
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => { window.removeEventListener("scroll", on); window.removeEventListener("resize", on); };
  }, []);
  return p;
}

/** Sections of the open page. The same items in the right-hand list and in the dropdown on narrow screens */
function SectionList({ onPick }: { onPick?: () => void }) {
  const { pathname } = useLocation();
  const sections = useSectionsStore((s) => s.sections);
  const active = useSectionsStore((s) => s.active);
  return (
    <>
      {sections.map((sec) => (
        <a key={sec.id} href={`#${pathname}#${sec.id}`} className={sec.id === active ? "on" : ""} aria-current={sec.id === active ? "location" : undefined}
           onClick={(e) => { e.preventDefault(); goToSection(sec.id, pathname); onPick?.(); }}>
          {sec.text}
        </a>
      ))}
    </>
  );
}

/** On very wide screens (≥ 1680 px), outside the area of wide figures: a quiet list without a box */
function PageToc() {
  const count = useSectionsStore((s) => s.sections.length);
  const t = useT();
  if (count < 2) return null;
  return (
    <nav className="pagetoc" aria-label={t("On this page", "Bu sayfada")}>
      <div className="pt-label">{t("On this page", "Bu sayfada")}</div>
      <div className="pt-track"><SectionList /></div>
    </nav>
  );
}

/** Where the right-hand list does not fit (< 1680 px): a dropdown in the top bar */
function SectionsMenu() {
  const count = useSectionsStore((s) => s.sections.length);
  const [open, setOpen] = useState(false);
  const t = useT();
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const away = (e: MouseEvent) => { if (!(e.target as HTMLElement).closest(".sections-menu")) setOpen(false); };
    window.addEventListener("keydown", key);
    window.addEventListener("mousedown", away);
    return () => { window.removeEventListener("keydown", key); window.removeEventListener("mousedown", away); };
  }, [open]);
  if (count < 2) return null;
  return (
    <div className="sections-menu">
      <button className="icon-btn" onClick={() => setOpen(!open)} aria-expanded={open}>{t("On this page", "Bu sayfada")} <span className="caret" style={{ fontSize: 11 }}>▾</span></button>
      {open && <nav className="sections-pop" aria-label={t("On this page", "Bu sayfada")}><SectionList onPick={() => setOpen(false)} /></nav>}
    </div>
  );
}

const THEMES = ["auto", "light", "dark"] as const;
const THEME_LABEL = {
  en: { auto: "◐ Auto", light: "☀ Light", dark: "☾ Dark" },
  tr: { auto: "◐ Oto", light: "☀ Açık", dark: "☾ Koyu" },
};

/** EN / TR switch: the chosen language is bold */
function LangSwitch() {
  const lang = useLang();
  const setLang = useStore((s) => s.setLang);
  return (
    <div className="seg lang-switch" role="group" aria-label="Language">
      {LANGS.map((l) => (
        <button key={l} aria-pressed={l === lang} onClick={() => setLang(l)}>{l.toUpperCase()}</button>
      ))}
    </div>
  );
}

function TopBar({ page, onToc }: { page: Page; onToc: () => void }) {
  const p = useScrollProgress();
  const { theme, setTheme } = useStore();
  const lang = useLang();
  const t = useT();
  const cur = useSectionsStore((s) => s.sections.find((x) => x.id === s.active)?.text);
  return (
    <header className="topbar">
      <Link to="/" className="brand">PaliGemma <span>{t("handbook", "el kitabı")}</span></Link>
      <div className="where">{page.num && <span className="mono" style={{ marginRight: 8 }}>{page.num}</span>}<b>{page.title[lang]}</b>{cur && <span className="cur"> · {cur}</span>}</div>
      <SectionsMenu />
      <button className="icon-btn" onClick={() => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % 3])} title={t("Theme", "Tema")}>{THEME_LABEL[lang][theme]}</button>
      <LangSwitch />
      <button className="icon-btn" onClick={onToc} style={{ color: "var(--fg)" }}>{t("Contents", "İçindekiler")} <span style={{ fontSize: 16, lineHeight: 1 }}>☰</span></button>
      <div className="progress" style={{ width: `${p * 100}%` }} />
    </header>
  );
}

function Toc({ current, onClose }: { current: Page; onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const lang = useLang();
  const t = useT();
  const groups: { part: Part | null; title: string; pages: Page[] }[] = [
    { part: null, title: START_TITLE[lang], pages: PAGES.filter((p) => p.part === null) },
    ...(["vision", "bridge", "lang"] as Part[]).map((part) => ({ part, title: PART_TITLE[part][lang], pages: PAGES.filter((p) => p.part === part) })),
  ];
  return (
    <>
      <div className="toc-backdrop" onClick={onClose} />
      <nav className="toc" aria-label={t("Contents", "İçindekiler")}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontFamily: "var(--display)", fontWeight: 700, fontSize: 20 }}>{t("Contents", "İçindekiler")}</div>
          <button className="icon-btn" onClick={onClose}>{t("Close", "Kapat")} ✕</button>
        </div>
        {groups.map((g) => (
          <div key={g.title} className="toc-part" style={partVars(g.part)}>
            <h2>{g.title}</h2>
            {g.pages.map((p) => (
              <Link key={p.path} to={p.path} onClick={onClose} className={"item" + (p === current ? " active" : "")}>
                <span className="n">{p.num || "·"}</span><span className="t">{p.title[lang]}</span><span className="b">{p.blurb[lang]}</span>
              </Link>
            ))}
          </div>
        ))}
      </nav>
    </>
  );
}

function PageView({ page }: { page: Page }) {
  const { pathname, hash } = useLocation();
  const [toc, setToc] = useState(false);
  const lang = useLang();
  const t = useT();
  useSectionsTracker(pathname, lang);
  useEffect(() => {
    if (!hash) { window.scrollTo(0, 0); return; }
    // MDX loads lazily: retry until the element exists (up to ~3 s), then wait briefly for the layout to settle
    const id = decodeURIComponent(hash.slice(1));
    let tries = 0, go = 0;
    const poll = window.setInterval(() => {
      const el = document.getElementById(id);
      if (el) { window.clearInterval(poll); go = window.setTimeout(() => el.scrollIntoView({ behavior: "smooth" }), 250); }
      else if (++tries > 30) window.clearInterval(poll);
    }, 100);
    return () => { window.clearInterval(poll); window.clearTimeout(go); };
  }, [pathname, hash]);
  const ri = PAGES.indexOf(page);
  const prev = ri > 0 ? PAGES[ri - 1] : null;
  const next = ri < PAGES.length - 1 ? PAGES[ri + 1] : null;
  return (
    <>
      <TopBar page={page} onToc={() => setToc(true)} />
      {toc && <Toc current={page} onClose={() => setToc(false)} />}
      <main className="page" style={partVars(page.part)}>
        <PageToc />
        {page.num && (
          <div className="page-head">
            <div className="num">{page.num}</div>
            <div className="kicker">{page.part ? PART_TITLE[page.part][lang] : START_TITLE[lang]}</div>
          </div>
        )}
        {page.arch.length > 0 && <ArchDiagram variant="mini" active={page.arch} />}
        <article className="prose" style={{ ["--chap" as any]: page.num ? `"${page.num}."` : '""' }}>
          <Suspense fallback={<div className="muted">{t("Loading…", "Yükleniyor…")}</div>}><PageBody page={page} /></Suspense>
        </article>
        <nav className="pager">
          {prev ? <Link to={prev.path}><span className="lbl">← {t("Previous", "Önceki")}</span><span className="t">{prev.num} {prev.title[lang]}</span></Link> : <span />}
          {next ? <Link to={next.path} className="next"><span className="lbl">{t("Next", "Sonraki")} →</span><span className="t">{next.num} {next.title[lang]}</span></Link> : <span />}
        </nav>
        <footer className="credits small muted">
          {t("Code: MIT. Weight slices from ", "Kod: MIT. Ağırlık parçaları ")}<a href="https://huggingface.co/google/paligemma-3b-pt-224">google/paligemma-3b-pt-224</a>
          {t(", provided under and subject to the ", " modelinden; ")}<a href="https://ai.google.dev/gemma/terms">Gemma Terms of Use</a>
          {t(". Example photos from Wikimedia Commons: Vikiçizer (CC BY-SA 4.0), Wiki leylek (CC BY 4.0), Ibrahim Husain Meraj (CC BY-SA 3.0).",
             " kapsamında. Örnek fotoğraflar Wikimedia Commons'tan: Vikiçizer (CC BY-SA 4.0), Wiki leylek (CC BY 4.0), Ibrahim Husain Meraj (CC BY-SA 3.0).")}
          {" "}<a href="https://github.com/mlihgenel/paligemma-from-scratch">GitHub</a>
        </footer>
      </main>
    </>
  );
}

/** The page's MDX in the current language. Keyed by language so switching remounts it cleanly */
function PageBody({ page }: { page: Page }) {
  const lang = useLang();
  const C = page.Component[lang];
  return <C key={lang} />;
}

/** Redirect that keeps the #section part of the address */
function Redirect({ to }: { to: string }) {
  const { hash } = useLocation();
  return <Navigate to={to + hash} replace />;
}

const mdxComponents = {
  table: (p: any) => <div style={{ overflowX: "auto" }}><table {...p} /></div>,
};

export default function App() {
  useEffectiveTheme();
  useDocumentLang();
  return (
    <HashRouter>
      <MDXProvider components={mdxComponents}>
        <Routes>
          {PAGES.map((p) => <Route key={p.path} path={p.path} element={<PageView page={p} />} />)}
          {Object.entries(REDIRECTS).map(([from, to]) => <Route key={from} path={from} element={<Redirect to={to} />} />)}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MDXProvider>
    </HashRouter>
  );
}
