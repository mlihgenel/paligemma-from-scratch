// Small building blocks used throughout the book: shapes, figures, asides, questions.
import { ReactNode, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLang, useT, type L } from "../i18n";

/** Axis name -> color. The same axis has the same color throughout the book */
export const AXIS_COLOR: Record<string, string> = {
  B: "var(--ax-B)", T: "var(--ax-T)", D: "var(--ax-D)", I: "var(--ax-I)", V: "var(--ax-V)", H: "var(--ax-H)",
  hd: "var(--ax-hd)", C: "var(--ax-C)", P: "var(--ax-P)", N: "var(--ax-P)", "1": "var(--fg-faint)",
};
const axisKey = (s: string) => {
  const t = s.trim();
  if (AXIS_COLOR[t]) return t;
  if (/^(2048|1152|768|256|64|hidden|d)$/i.test(t)) return /^(256)$/.test(t) ? "T" : "D";
  return "";
};

/** Colored shape: <S s="B, T, D" /> or <S s="1, 256, 1152" ax="B,T,D" /> */
export function S({ s, ax }: { s: string; ax?: string }) {
  const parts = s.split(",").map((p) => p.trim());
  const axes = ax ? ax.split(",").map((p) => p.trim()) : parts;
  return (
    <span className="shape mono" style={{ whiteSpace: "nowrap", fontSize: "0.88em" }}>
      [{parts.map((p, i) => (
        <span key={i}>
          <span style={{ color: AXIS_COLOR[axes[i]] ?? AXIS_COLOR[axisKey(axes[i] ?? "")] ?? "inherit", fontWeight: 600 }}>{p}</span>
          {i < parts.length - 1 ? ", " : ""}
        </span>
      ))}]
    </span>
  );
}

export function AxisLegend({ axes }: { axes: [string, string][] }) {
  return (
    <div className="row small" style={{ gap: 12, fontFamily: "var(--sans)" }}>
      {axes.map(([a, label]) => (
        <span key={a} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 3, background: AXIS_COLOR[a] }} />
          <b className="mono" style={{ color: AXIS_COLOR[a] }}>{a}</b>
          <span className="muted">{label}</span>
        </span>
      ))}
    </div>
  );
}

export function Fig({ children, caption, wide = false }: { children: ReactNode; caption?: ReactNode; wide?: boolean; pad?: boolean }) {
  return (
    <figure className={"fig" + (wide ? " wide" : "")} style={{ marginLeft: wide ? undefined : 0, marginRight: wide ? undefined : 0 }}>
      <div className="fig-body">{children}</div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

const CALLOUT: Record<string, { label: L<string>; color: string }> = {
  not: { label: { en: "Note", tr: "Not" }, color: "var(--border-strong)" },
  dikkat: { label: { en: "Careful", tr: "Dikkat" }, color: "var(--warn)" },
  neden: { label: { en: "Why?", tr: "Neden?" }, color: "var(--accent)" },
  bulgu: { label: { en: "From the real weights", tr: "Gerçek ağırlıklardan" }, color: "var(--ax-V)" },
  ozet: { label: { en: "Summary", tr: "Özet" }, color: "var(--fg)" },
};

/** Plain aside: a left rule + a small label. No box */
export function Callout({ kind = "not", title, children }: { kind?: keyof typeof CALLOUT; title?: ReactNode; children: ReactNode }) {
  const c = CALLOUT[kind];
  const lang = useLang();
  return (
    <aside className="aside" style={{ ["--aside-c" as any]: c.color }}>
      <span className="aside-label">{title ?? c.label[lang]}</span>
      <div>{children}</div>
    </aside>
  );
}

/** Side note: in the right margin on wide screens, inline on narrow ones */
export function Note({ label, children }: { label?: ReactNode; children: ReactNode }) {
  return (
    <span className="sidenote" role="note">
      {label && <span className="sn-label">{label}</span>}
      {children}
    </span>
  );
}

/** "Guess first": the answer is hidden */
export function Predict({ q, children, hint }: { q: ReactNode; children: ReactNode; hint?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [guess, setGuess] = useState("");
  const t = useT();
  return (
    <div className="aside" style={{ ["--aside-c" as any]: "var(--ax-I)" }}>
      <span className="aside-label">{t("Guess first", "Önce tahmin et")}</span>
      <div style={{ fontWeight: 500, marginBottom: 10 }}>{q}</div>
      {!open && (
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          <input value={guess} onChange={(e) => setGuess(e.target.value)} placeholder={t("your guess…", "tahminin…")}
            style={{ flex: 1, minWidth: 160, padding: "6px 0", border: 0, borderBottom: "1px solid var(--border-strong)", background: "transparent", outline: "none" }} />
          <button className="btn primary" onClick={() => setOpen(true)}>{t("Show answer", "Cevabı göster")}</button>
        </div>
      )}
      {hint && !open && <div className="small muted" style={{ marginTop: 8 }}>{t("Hint", "İpucu")}: {hint}</div>}
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}>
            {guess && <div className="small muted" style={{ marginBottom: 6 }}>{t("Your guess", "Senin tahminin")}: <b>{guess}</b></div>}
            <div>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Concept question, the answer in a collapsible panel */
export function ConceptQ({ n, q, children }: { n?: number | string; q: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderTop: "1px solid var(--border)", padding: "14px 0", fontFamily: "var(--sans)", fontSize: 15.5 }}>
      <button onClick={() => setOpen(!open)} style={{ all: "unset", cursor: "pointer", display: "flex", gap: 14, width: "100%" }}>
        <span className="mono" style={{ color: "var(--fg-faint)", fontSize: 13, paddingTop: 2 }}>{n ?? "?"}</span>
        <span style={{ flex: 1, fontWeight: 500 }}>{q}</span>
        <span className="muted" style={{ fontSize: 18, lineHeight: 1 }}>{open ? "−" : "+"}</span>
      </button>
      {open && <div style={{ padding: "10px 0 0 32px" }}>{children}</div>}
    </div>
  );
}

/** "What does this line affect?" */
export function Impact({ from, to, children }: { from: ReactNode; to: ReactNode; children: ReactNode }) {
  const t = useT();
  return (
    <div className="aside" style={{ ["--aside-c" as any]: "var(--accent)" }}>
      <span className="aside-label">{t("What does it affect?", "Neyi etkiliyor?")}</span>
      <div className="mono" style={{ fontSize: 13, margin: "2px 0 8px", color: "var(--fg)" }}>
        {from} <span style={{ color: "var(--accent)", margin: "0 6px" }}>⟶</span> {to}
      </div>
      <div>{children}</div>
    </div>
  );
}

/** Segmented control */
export function Seg<T extends string | number>({ value, options, onChange }: { value: T; options: [T, ReactNode][]; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="group">
      {options.map(([v, label]) => (
        <button key={String(v)} aria-pressed={v === value} onClick={() => onChange(v)}>{label}</button>
      ))}
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: "ok" | "bad" | "warn" }) {
  return (
    <div className="stat">
      <div className="lbl">{label}</div>
      <div className="val" style={{ color: tone ? `var(--${tone})` : undefined }}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Loading({ what, error }: { what: string; error?: string | null }) {
  const t = useT();
  return (
    <div className="small" style={{ padding: 20, color: error ? "var(--bad)" : "var(--fg-muted)" }}>
      {error ? `${t("Could not load data", "Veri yüklenemedi")} (${what}): ${error}` : `${what} ${t("loading…", "yükleniyor…")}`}
    </div>
  );
}
