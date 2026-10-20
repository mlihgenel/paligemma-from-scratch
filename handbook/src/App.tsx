import { Suspense, useEffect, useState } from "react";
import { HashRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { MDXProvider } from "@mdx-js/react";
import { PAGES, Page } from "./registry";
import { PipelineMinimap } from "./components/PipelineMinimap";
import { useStore } from "./store";

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

const THEMES = ["auto", "light", "dark"] as const;
const THEME_LABEL = { auto: "◐ Oto", light: "☀ Açık", dark: "☾ Koyu" };

function TopBar({ page, onToc }: { page: Page; onToc: () => void }) {
  const p = useScrollProgress();
  const { theme, setTheme } = useStore();
  return (
    <header className="topbar">
      <Link to="/" className="brand">PaliGemma <span>el kitabı</span></Link>
      <div className="where">{page.num && <span className="mono" style={{ marginRight: 8 }}>{page.num}</span>}<b>{page.title}</b></div>
      <button className="icon-btn" onClick={() => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % 3])} title="Tema">{THEME_LABEL[theme]}</button>
      <button className="icon-btn" onClick={onToc} style={{ color: "var(--fg)" }}>İçindekiler <span style={{ fontSize: 16, lineHeight: 1 }}>☰</span></button>
      <div className="progress" style={{ width: `${p * 100}%` }} />
    </header>
  );
}

function Toc({ current, onClose }: { current: Page; onClose: () => void }) {
  const chapters = [...new Set(PAGES.map((p) => p.chapter))];
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <>
      <div className="toc-backdrop" onClick={onClose} />
      <nav className="toc" aria-label="İçindekiler">
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontFamily: "var(--display)", fontWeight: 700, fontSize: 20 }}>İçindekiler</div>
          <button className="icon-btn" onClick={onClose}>Kapat ✕</button>
        </div>
        <h2>Mimaride konum</h2>
        <PipelineMinimap active={current.active} vertical onNavigate={onClose} />
        {chapters.map((c) => (
          <div key={c}>
            <h2>{c}</h2>
            {PAGES.filter((p) => p.chapter === c).map((p) => (
              p.Component ? (
                <Link key={p.path} to={p.path} onClick={onClose} className={"item" + (p === current ? " active" : "")}>
                  <span className="n">{p.num}</span><span>{p.title}</span><span className="soon">{p.stage ?? ""}</span>
                </Link>
              ) : (
                <span key={p.path} className="item disabled" style={{ display: "grid", gridTemplateColumns: "44px 1fr auto", gap: 8, padding: "7px 0", color: "var(--fg-faint)" }}>
                  <span className="mono" style={{ fontSize: 12.5 }}>{p.num}</span><span>{p.title}</span><span style={{ fontSize: 12 }}>yakında</span>
                </span>
              )
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
  useEffect(() => {
    if (hash) setTimeout(() => document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth" }), 300);
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  const ready = PAGES.filter((p) => p.Component);
  const ri = ready.indexOf(page);
  const prev = ri > 0 ? ready[ri - 1] : null;
  const next = ri >= 0 && ri < ready.length - 1 ? ready[ri + 1] : null;
  return (
    <>
      <TopBar page={page} onToc={() => setToc(true)} />
      {toc && <Toc current={page} onClose={() => setToc(false)} />}
      <main className="page">
        {page.num && (
          <div className="page-head">
            <div className="num">{page.num}</div>
            <div className="kicker">{page.chapter}{page.stage ? ` · ${page.stage}` : ""}</div>
          </div>
        )}
        <article className="prose" style={{ ["--chap" as any]: page.num ? `"${page.num}."` : '""' }}>
          {page.Component ? (
            <Suspense fallback={<div className="muted">Yükleniyor…</div>}><page.Component /></Suspense>
          ) : (
            <>
              <h1>{page.title}</h1>
              <p className="lead">Bu bölüm sonraki fazda yazılacak.</p>
            </>
          )}
        </article>
        <nav className="pager">
          {prev ? <Link to={prev.path}><span className="lbl">← Önceki</span><span className="t">{prev.num} {prev.title}</span></Link> : <span />}
          {next ? <Link to={next.path} className="next"><span className="lbl">Sonraki →</span><span className="t">{next.num} {next.title}</span></Link> : <span />}
        </nav>
      </main>
    </>
  );
}

const mdxComponents = {
  table: (p: any) => <div style={{ overflowX: "auto" }}><table {...p} /></div>,
};

export default function App() {
  useEffectiveTheme();
  return (
    <HashRouter>
      <MDXProvider components={mdxComponents}>
        <Routes>
          {PAGES.map((p) => <Route key={p.path} path={p.path} element={<PageView page={p} />} />)}
        </Routes>
      </MDXProvider>
    </HashRouter>
  );
}
