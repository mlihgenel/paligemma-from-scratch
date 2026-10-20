// Kitap boyunca kullanılan küçük yapı taşları: şekil, figür, kutu, sorular.
import { ReactNode, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/** Eksen adı -> renk. Kitap boyunca aynı eksen hep aynı renk */
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

/** Renkli şekil: <S s="B, T, D" /> ya da <S s="1, 256, 1152" ax="B,T,D" /> */
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

const CALLOUT: Record<string, { label: string; color: string }> = {
  not: { label: "Not", color: "var(--border-strong)" },
  dikkat: { label: "Dikkat", color: "var(--warn)" },
  neden: { label: "Neden?", color: "var(--accent)" },
  bulgu: { label: "Gerçek ağırlıklardan", color: "var(--ax-V)" },
  ozet: { label: "Özet", color: "var(--fg)" },
};

/** Sade aside: sol kenar çizgisi + küçük etiket. Kutu yok */
export function Callout({ kind = "not", title, children }: { kind?: keyof typeof CALLOUT; title?: ReactNode; children: ReactNode }) {
  const c = CALLOUT[kind];
  return (
    <aside className="aside" style={{ ["--aside-c" as any]: c.color }}>
      <span className="aside-label">{title ?? c.label}</span>
      <div>{children}</div>
    </aside>
  );
}

/** Kenar notu: geniş ekranda sağ boşlukta, dar ekranda satır içi */
export function Note({ label, children }: { label?: ReactNode; children: ReactNode }) {
  return (
    <span className="sidenote" role="note">
      {label && <span className="sn-label">{label}</span>}
      {children}
    </span>
  );
}

/** "Önce tahmin et": cevap gizli */
export function Predict({ q, children, hint }: { q: ReactNode; children: ReactNode; hint?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [guess, setGuess] = useState("");
  return (
    <div className="aside" style={{ ["--aside-c" as any]: "var(--ax-I)" }}>
      <span className="aside-label">Önce tahmin et</span>
      <div style={{ fontWeight: 500, marginBottom: 10 }}>{q}</div>
      {!open && (
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          <input value={guess} onChange={(e) => setGuess(e.target.value)} placeholder="tahminin…"
            style={{ flex: 1, minWidth: 160, padding: "6px 0", border: 0, borderBottom: "1px solid var(--border-strong)", background: "transparent", outline: "none" }} />
          <button className="btn primary" onClick={() => setOpen(true)}>Cevabı göster</button>
        </div>
      )}
      {hint && !open && <div className="small muted" style={{ marginTop: 8 }}>İpucu: {hint}</div>}
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}>
            {guess && <div className="small muted" style={{ marginBottom: 6 }}>Senin tahminin: <b>{guess}</b></div>}
            <div>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** ROADMAP kavram sorusu, cevap açılır panelde */
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

/** "Bu satır neyi etkiliyor?" */
export function Impact({ from, to, children }: { from: ReactNode; to: ReactNode; children: ReactNode }) {
  return (
    <div className="aside" style={{ ["--aside-c" as any]: "var(--accent)" }}>
      <span className="aside-label">Neyi etkiliyor?</span>
      <div className="mono" style={{ fontSize: 13, margin: "2px 0 8px", color: "var(--fg)" }}>
        {from} <span style={{ color: "var(--accent)", margin: "0 6px" }}>⟶</span> {to}
      </div>
      <div>{children}</div>
    </div>
  );
}

/** Segment seçici */
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
  return (
    <div className="small" style={{ padding: 20, color: error ? "var(--bad)" : "var(--fg-muted)" }}>
      {error ? `Veri yüklenemedi (${what}): ${error}` : `${what} yükleniyor…`}
    </div>
  );
}
