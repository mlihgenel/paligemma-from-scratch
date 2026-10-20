import { ReactNode, useState } from "react";
import { motion } from "framer-motion";
import { S } from "./ui";

/** Adım adım şekil izi. Her adım: işlem, kod, çıkış şekli */
export interface TraceStep { op: ReactNode; code?: string; shape: string; ax?: string; note?: ReactNode; changed?: number[] }

export function ShapeTrace({ steps, start }: { steps: TraceStep[]; start?: { shape: string; ax?: string; label?: ReactNode } }) {
  const [i, setI] = useState(steps.length);
  return (
    <div style={{ fontFamily: "var(--sans)", fontSize: 14.5 }}>
      <div className="row" style={{ gap: 8, marginBottom: 10, alignItems: "center" }}>
        <button className="btn" onClick={() => setI(0)}>⟲</button>
        <button className="btn" disabled={i <= 0} onClick={() => setI(i - 1)}>←</button>
        <button className="btn primary" disabled={i >= steps.length} onClick={() => setI(i + 1)}>Sonraki adım →</button>
        <span className="small muted">{i}/{steps.length}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: "6px 14px", alignItems: "center" }}>
        {start && (
          <div style={{ display: "contents" }}>
            <span className="pill">giriş</span>
            <span className="muted">{start.label ?? "girdi"}</span>
            <S s={start.shape} ax={start.ax} />
          </div>
        )}
        {steps.map((s, k) => (
          <motion.div key={k} style={{ display: "contents" }} initial={false} animate={{ opacity: k < i ? 1 : 0.18 }}>
            <motion.span className="pill" animate={{ opacity: k < i ? 1 : 0.25 }} style={{ background: k === i - 1 ? "var(--accent-soft)" : undefined, color: k === i - 1 ? "var(--accent)" : undefined }}>{k + 1}</motion.span>
            <motion.div animate={{ opacity: k < i ? 1 : 0.25 }}>
              <div>{s.op} {s.code && <code>{s.code}</code>}</div>
              {s.note && k === i - 1 && <div className="small muted" style={{ marginTop: 2 }}>{s.note}</div>}
            </motion.div>
            <motion.div animate={{ opacity: k < i ? 1 : 0.25 }}><S s={s.shape} ax={s.ax} /></motion.div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

/** "Ne bozulur?" anahtarı: doğru/hatalı kod ve sonuç */
export function BreakIt({ title, good, bad, children, result }: {
  title: ReactNode; good: string; bad: string; children?: (broken: boolean) => ReactNode; result?: (broken: boolean) => ReactNode;
}) {
  const [broken, setBroken] = useState(false);
  return (
    <div className="breakit" style={{ margin: "40px 0", fontFamily: "var(--sans)", fontSize: 15 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
        <div><span className="eyebrow" style={{ color: "var(--bad)" }}>Ne bozulur?</span><div style={{ fontWeight: 600, fontSize: 16, marginTop: 2 }}>{title}</div></div>
        <button role="switch" aria-checked={broken} onClick={() => setBroken(!broken)} className="switch" data-on={broken}>
          <span className="knob" />
          <span>{broken ? "bozuk sürüm" : "senin kodun"}</span>
        </button>
      </div>
      <pre className="mono" style={{ margin: 0, padding: "12px 14px", fontSize: 12.8, background: "var(--code-bg)", whiteSpace: "pre-wrap", borderRadius: 6, borderLeft: `2px solid ${broken ? "var(--bad)" : "var(--ok)"}`, transition: "border-color .2s" }}>
        {(broken ? bad : good).split("\n").map((l, i) => (
          <div key={i} style={{ color: l.includes("# ←") ? (broken ? "var(--bad)" : "var(--ok)") : undefined }}>{l}</div>
        ))}
      </pre>
      <div style={{ paddingTop: 14 }}>
        {children?.(broken)}
        {result?.(broken)}
      </div>
    </div>
  );
}

export function ManimVideo({ scene, caption }: { scene: string; caption?: ReactNode; steps?: ReactNode[] }) {
  const [err, setErr] = useState(false);
  const [open, setOpen] = useState(false);
  return (
    <div style={{ margin: "36px 0", fontFamily: "var(--sans)" }}>
      <button className="btn" onClick={() => setOpen(!open)} style={{ paddingLeft: 0, color: "var(--fg)" }}>
        <span style={{ display: "inline-flex", width: 26, height: 26, borderRadius: "50%", background: "var(--fg)", color: "var(--bg)", alignItems: "center", justifyContent: "center", fontSize: 10, marginRight: 10 }}>{open ? "■" : "▶"}</span>
        Animasyon: {caption}
      </button>
      {open && (
        <div style={{ marginTop: 12, overflow: "hidden", borderRadius: 6, background: "var(--stage-bg)" }}>
          {err ? (
            <div className="small muted" style={{ padding: 30, textAlign: "center" }}>Video henüz üretilmedi: <code>{scene}.mp4</code></div>
          ) : (
            <video src={`${import.meta.env.BASE_URL}videos/${scene}.mp4`} controls autoPlay loop muted playsInline onError={() => setErr(true)} style={{ width: "100%", display: "block", aspectRatio: "16 / 9" }} />
          )}
        </div>
      )}
    </div>
  );
}
