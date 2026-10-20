// d3 tabanlı küçük grafikler: çizgi, histogram, ısı haritası (canvas), çubuk.
import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const M = { t: 14, r: 16, b: 34, l: 48 };

export interface Series { name: string; values: number[]; color: string; dash?: boolean; x?: number[] }

export function LineChart({ series, height = 240, xLabel, yLabel, xTicks, yDomain, fmtX, marks, legend = true }: {
  series: Series[]; height?: number; xLabel?: string; yLabel?: string; xTicks?: (number | string)[]; yDomain?: [number, number];
  fmtX?: (x: number) => string; marks?: { x: number; label: string }[]; legend?: boolean;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const xs = series[0]?.x ?? series[0]?.values.map((_, i) => i) ?? [];
  const x = d3.scaleLinear().domain(d3.extent(series.flatMap((s) => s.x ?? s.values.map((_, i) => i))) as [number, number]).range([M.l, w - M.r]);
  const all = series.flatMap((s) => s.values);
  const y = d3.scaleLinear().domain(yDomain ?? [Math.min(0, d3.min(all)!), d3.max(all)! * 1.05]).nice().range([height - M.b, M.t]);
  return (
    <div ref={ref} style={{ position: "relative", fontFamily: "var(--sans)" }}>
      {legend && series.length > 1 && (
        <div className="row small" style={{ gap: 14, marginBottom: 6 }}>
          {series.map((s) => (
            <span key={s.name} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <svg width="18" height="8"><line x1="0" x2="18" y1="4" y2="4" stroke={s.color} strokeWidth="2.5" strokeDasharray={s.dash ? "4 3" : undefined} /></svg>
              {s.name}
            </span>
          ))}
        </div>
      )}
      <svg width={w} height={height} onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const xv = x.invert(e.clientX - r.left);
          let best = 0;
          xs.forEach((v, i) => { if (Math.abs(v - xv) < Math.abs(xs[best] - xv)) best = i; });
          setHover(best);
        }}>
        <g>
          {y.ticks(5).map((t) => (
            <g key={t}>
              <line x1={M.l} x2={w - M.r} y1={y(t)} y2={y(t)} stroke="var(--border)" />
              <text x={M.l - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize="11" fill="var(--fg-faint)">{fmtTick(t)}</text>
            </g>
          ))}
          {(xTicks ?? x.ticks(Math.min(10, xs.length))).map((t, i) => {
            const v = typeof t === "number" ? t : xs[i];
            return <text key={i} x={x(v)} y={height - M.b + 16} textAnchor="middle" fontSize="11" fill="var(--fg-faint)">{typeof t === "string" ? t : fmtX ? fmtX(t) : t}</text>;
          })}
          {xLabel && <text x={(M.l + w - M.r) / 2} y={height - 2} textAnchor="middle" fontSize="11.5" fill="var(--fg-muted)">{xLabel}</text>}
          {yLabel && <text transform={`translate(12,${(M.t + height - M.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize="11.5" fill="var(--fg-muted)">{yLabel}</text>}
          {marks?.map((m) => (
            <g key={m.label}>
              <line x1={x(m.x)} x2={x(m.x)} y1={M.t} y2={height - M.b} stroke="var(--fg-faint)" strokeDasharray="3 3" />
              <text x={x(m.x) + 4} y={M.t + 10} fontSize="11" fill="var(--fg-muted)">{m.label}</text>
            </g>
          ))}
          {series.map((s) => {
            const sx = s.x ?? s.values.map((_, i) => i);
            const line = d3.line<number>().x((_, i) => x(sx[i])).y((v) => y(v)).curve(d3.curveMonotoneX);
            return (
              <g key={s.name}>
                <path d={line(s.values) ?? ""} fill="none" stroke={s.color} strokeWidth="2.2" strokeDasharray={s.dash ? "5 4" : undefined} />
                {s.values.length <= 30 && s.values.map((v, i) => <circle key={i} cx={x(sx[i])} cy={y(v)} r={hover === i ? 4.5 : 2.6} fill={s.color} />)}
              </g>
            );
          })}
          {hover !== null && <line x1={x(xs[hover])} x2={x(xs[hover])} y1={M.t} y2={height - M.b} stroke="var(--fg-faint)" />}
        </g>
      </svg>
      {hover !== null && (
        <div className="tip small" style={{ position: "absolute", top: 24, left: Math.min(x(xs[hover]) + 10, w - 190), padding: "6px 10px", pointerEvents: "none", minWidth: 150 }}>
          <div className="muted">{fmtX ? fmtX(xs[hover]) : xs[hover]}</div>
          {series.map((s) => (
            <div key={s.name} style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
              <span style={{ color: s.color }}>{s.name}</span><b className="mono">{fmtTick(s.values[hover])}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function fmtTick(v: number) {
  if (v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(a < 0.1 ? 3 : 2);
}

export function Histogram({ values, bins = 50, height = 180, color = "var(--accent)", domain, marks, xLabel, overlay }: {
  values: ArrayLike<number>; bins?: number; height?: number; color?: string; domain?: [number, number];
  marks?: { x: number; label: string; color?: string }[]; xLabel?: string; overlay?: { values: ArrayLike<number>; color: string; label: string };
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const ext = domain ?? (d3.extent(Array.from(values)) as [number, number]);
  const x = d3.scaleLinear().domain(ext).nice().range([M.l, w - M.r]);
  const gen = d3.bin().domain(x.domain() as [number, number]).thresholds(x.ticks(bins));
  const b1 = gen(Array.from(values));
  const b2 = overlay ? gen(Array.from(overlay.values)) : [];
  const y = d3.scaleLinear().domain([0, d3.max([...b1, ...b2], (b) => b.length)!]).nice().range([height - M.b, M.t]);
  return (
    <div ref={ref} style={{ fontFamily: "var(--sans)" }}>
      <svg width={w} height={height}>
        {y.ticks(4).map((t) => <line key={t} x1={M.l} x2={w - M.r} y1={y(t)} y2={y(t)} stroke="var(--border)" />)}
        {b1.map((b, i) => <rect key={i} x={x(b.x0!) + 0.5} width={Math.max(0, x(b.x1!) - x(b.x0!) - 1)} y={y(b.length)} height={y(0) - y(b.length)} fill={color} opacity={overlay ? 0.75 : 0.9} />)}
        {b2.map((b, i) => <rect key={"o" + i} x={x(b.x0!) + 0.5} width={Math.max(0, x(b.x1!) - x(b.x0!) - 1)} y={y(b.length)} height={y(0) - y(b.length)} fill={overlay!.color} opacity={0.55} />)}
        {x.ticks(8).map((t) => <text key={t} x={x(t)} y={height - M.b + 16} textAnchor="middle" fontSize="11" fill="var(--fg-faint)">{fmtTick(t)}</text>)}
        {xLabel && <text x={(M.l + w - M.r) / 2} y={height - 2} textAnchor="middle" fontSize="11.5" fill="var(--fg-muted)">{xLabel}</text>}
        {marks?.map((m) => (
          <g key={m.label}>
            <line x1={x(m.x)} x2={x(m.x)} y1={M.t} y2={height - M.b} stroke={m.color ?? "var(--fg)"} strokeWidth="1.5" strokeDasharray="4 3" />
            <text x={x(m.x) + 4} y={M.t + 10} fontSize="11" fill={m.color ?? "var(--fg)"}>{m.label}</text>
          </g>
        ))}
      </svg>
      {overlay && (
        <div className="row small" style={{ gap: 14 }}>
          <span><span style={{ color }}>■</span> önce</span><span><span style={{ color: overlay.color }}>■</span> {overlay.label}</span>
        </div>
      )}
    </div>
  );
}

/** Canvas ısı haritası. values: rows*cols, satır öncelikli */
export function Heatmap({ values, rows, cols, height, diverging = false, domain, onCell, highlight, cellLabel, rowLabel, colLabel, legend = true }: {
  values: ArrayLike<number>; rows: number; cols: number; height?: number; diverging?: boolean; domain?: [number, number];
  onCell?: (r: number, c: number) => void; highlight?: { r?: number; c?: number }; cellLabel?: (r: number, c: number, v: number) => ReactNode;
  rowLabel?: string; colLabel?: string; legend?: boolean;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const H = height ?? Math.min(420, Math.max(80, (w * rows) / cols));
  const dom = useMemo<[number, number]>(() => {
    if (domain) return domain;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) { lo = Math.min(lo, values[i]); hi = Math.max(hi, values[i]); }
    if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); return [-m, m]; }
    return [lo, hi];
  }, [values, domain, diverging]);
  const color = useMemo(() => diverging ? d3.scaleDiverging(d3.interpolateRdBu).domain([dom[1], 0, dom[0]]) : d3.scaleSequential(d3.interpolateInferno).domain(dom), [dom, diverging]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = cols; c.height = rows;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const col = d3.rgb(color(values[i]) as string);
      img.data[i * 4] = col.r; img.data[i * 4 + 1] = col.g; img.data[i * 4 + 2] = col.b; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, rows, cols, color]);

  const pick = (e: React.MouseEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const c = Math.floor(((e.clientX - r.left) / r.width) * cols), rr = Math.floor(((e.clientY - r.top) / r.height) * rows);
    return [Math.min(rows - 1, Math.max(0, rr)), Math.min(cols - 1, Math.max(0, c))] as [number, number];
  };
  return (
    <div ref={ref} style={{ position: "relative", fontFamily: "var(--sans)" }}>
      <div style={{ position: "relative", width: "100%", height: H, cursor: onCell ? "crosshair" : "default" }}
        onMouseMove={(e) => setHover(pick(e))} onMouseLeave={() => setHover(null)} onClick={(e) => { const [r, c] = pick(e); onCell?.(r, c); }}>
        <canvas ref={canvas} style={{ width: "100%", height: "100%", imageRendering: "pixelated", borderRadius: 4, display: "block" }} />
        {highlight?.r !== undefined && <div style={{ position: "absolute", left: 0, right: 0, top: `${(highlight.r / rows) * 100}%`, height: `${100 / rows}%`, outline: "2px solid var(--accent)", pointerEvents: "none" }} />}
        {highlight?.c !== undefined && <div style={{ position: "absolute", top: 0, bottom: 0, left: `${(highlight.c / cols) * 100}%`, width: `${100 / cols}%`, outline: "2px solid var(--accent)", pointerEvents: "none" }} />}
      </div>
      <div className="row small muted" style={{ justifyContent: "space-between", marginTop: 4 }}>
        <span>{rowLabel && <>↓ {rowLabel}</>} {colLabel && <>· → {colLabel}</>}</span>
        {legend && <span className="mono">{fmtTick(dom[0])} <span style={{ display: "inline-block", width: 80, height: 8, verticalAlign: "middle", background: `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((t) => color(dom[0] + t * (dom[1] - dom[0]))).join(",")})`, borderRadius: 2 }} /> {fmtTick(dom[1])}</span>}
      </div>
      {hover && (
        <div className="tip small mono" style={{ position: "absolute", top: 4, right: 4, padding: "4px 8px", pointerEvents: "none" }}>
          {cellLabel ? cellLabel(hover[0], hover[1], values[hover[0] * cols + hover[1]]) : `[${hover[0]}, ${hover[1]}] = ${fmtTick(values[hover[0] * cols + hover[1]])}`}
        </div>
      )}
    </div>
  );
}

/** Yatay çubuklar: {label, value, color} */
export function Bars({ items, fmt = fmtTick, max }: { items: { label: ReactNode; value: number; color?: string; note?: ReactNode }[]; fmt?: (v: number) => string; max?: number }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(90px, max-content) 1fr auto", gap: "6px 12px", alignItems: "center", fontFamily: "var(--sans)", fontSize: 13.5 }}>
      {items.map((it, i) => (
        <div key={i} style={{ display: "contents" }}>
          <div>{it.label}</div>
          <div style={{ background: "var(--bg-sunk)", borderRadius: 4, height: 14 }}>
            <div style={{ width: `${(Math.abs(it.value) / m) * 100}%`, height: "100%", background: it.color ?? "var(--accent)", borderRadius: 4, transition: "width .4s" }} />
          </div>
          <div className="mono" style={{ textAlign: "right" }}>{fmt(it.value)} {it.note}</div>
        </div>
      ))}
    </div>
  );
}
