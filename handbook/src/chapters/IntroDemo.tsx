// Giriş sayfası: kitabın etkileşim dilini gösteren küçük, canlı bir örnek.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useWidth } from "../components/charts";
import { CellCanvas, Cell } from "../interact/cells";
import { rmsnormSteps, fmtNum } from "../lib/tensor";
import { PAGES } from "../registry";

export function IntroDemo() {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [x, setX] = useState([2.4, -1.2, 0.6, 3.1, -2.2, 0.9]);
  const st = rmsnormSteps(x, x.map(() => 0));
  const cs = Math.min(52, (W - 160) / 13);
  const cells: Cell[] = [];
  x.forEach((v, i) => cells.push({ id: "x" + i, x: 40 + i * (cs + 4), y: 34, w: cs, h: cs, v, text: v.toFixed(1), editable: true,
    tip: () => <>x[{i}] = <b>{v.toFixed(2)}</b><div className="muted">yukarı/aşağı sürükle</div></> }));
  st.out.forEach((v, i) => cells.push({ id: "o" + i, x: 40 + (6 + 1) * (cs + 4) + i * (cs + 4), y: 34, w: cs, h: cs, v, text: v.toFixed(2),
    tip: () => <>x[{i}] / rms(x) = {x[i].toFixed(2)} / {fmtNum(Math.sqrt(st.meanSq), 3)} = <b>{v.toFixed(3)}</b></> }));
  return (
    <div ref={ref}>
      <CellCanvas cells={cells} width={W} height={cs + 64} max={3}
        onDrag={(id, dy) => { const i = +id.slice(1); const n = x.slice(); n[i] = Math.round((n[i] + dy / 25) * 10) / 10; setX(n); }}
        labels={<>
          <div style={{ position: "absolute", left: 40, top: 8, fontSize: 12.5, fontWeight: 600 }}>x <span className="muted" style={{ fontWeight: 400 }}>(sürükle)</span></div>
          <div style={{ position: "absolute", left: 40 + 7 * (cs + 4), top: 8, fontSize: 12.5, fontWeight: 600 }}>RMSNorm(x) <span className="muted" style={{ fontWeight: 400 }}>(üstüne gel)</span></div>
          <div style={{ position: "absolute", left: 40 + 6 * (cs + 4) + cs * 0.2, top: 34, lineHeight: `${cs}px`, fontSize: 20, color: "var(--fg-faint)" }}>→</div>
        </>} />
    </div>
  );
}

export function ChapterList() {
  const groups = [...new Set(PAGES.filter((p) => p.num).map((p) => p.chapter))];
  return (
    <div className="grid2" style={{ gap: "8px 40px", fontFamily: "var(--sans)" }}>
      {groups.map((g) => (
        <div key={g} style={{ marginBottom: 18 }}>
          <div className="eyebrow" style={{ marginBottom: 6 }}>{g}</div>
          {PAGES.filter((p) => p.chapter === g && p.num).map((p) => p.Component ? (
            <Link key={p.path} to={p.path} style={{ display: "grid", gridTemplateColumns: "38px 1fr", padding: "4px 0", background: "none", color: "var(--fg)", fontSize: 15 }}>
              <span className="mono" style={{ color: "var(--fg-faint)", fontSize: 12.5 }}>{p.num}</span><span>{p.title}</span>
            </Link>
          ) : (
            <div key={p.path} style={{ display: "grid", gridTemplateColumns: "38px 1fr", padding: "4px 0", color: "var(--fg-faint)", fontSize: 15 }}>
              <span className="mono" style={{ fontSize: 12.5 }}>{p.num}</span><span>{p.title} · yakında</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
