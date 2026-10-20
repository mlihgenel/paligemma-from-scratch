// 5.3 pilot: scrolly RMSNorm dizisi, sürüklenebilir laboratuvar, Mafs geometrisi, 3D keepdim.
import { ReactNode, useMemo, useState } from "react";
import { Mafs, Coordinates, Circle, Ellipse, Line, Point, Text, useMovablePoint } from "mafs";
import { useData, rowOf } from "../../../lib/data";
import { rmsnormSteps, layernorm, rng, fmtNum } from "../../../lib/tensor";
import { useWidth } from "../../../components/charts";
import { Cell, CellCanvas } from "../../../interact/cells";
import { Stage3D, Cubes, Cube, Tag, View } from "../../../interact/three";
import { Loading, S, Seg } from "../../../components/ui";
import type { PgSlices } from "./types";

const D = 8;
const TOK = ["<bos>", "caption", " tr", "\n"];
const SHOW = (p: string) => p.replace("▁", "␣").replace("\n", "\\n");

/** Gerçek veri: 4 prompt token'ının embedding'i (×√2048, ilk 8 boyut) ve katman 0 input_layernorm ağırlığı */
function useRealX() {
  const { data, error } = useData<PgSlices>("pg_slices");
  const v = useMemo(() => {
    if (!data) return null;
    const rows = TOK.map((t) => {
      const i = data.embed_named.findIndex((n) => n.text === t);
      return { piece: data.embed_named[i].piece, x: Array.from(rowOf(data.embed_named_rows, i).slice(0, D), (a) => Math.fround(a * 45.25)) };
    });
    const w = Array.from(rowOf(data.norms.input_layernorm, 0).slice(0, D));
    return { rows, w };
  }, [data]);
  return { v, error };
}

const f = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2));

const STEP_CODE = [
  ["x", "B, T, D", "girdi"],
  ["x.pow(2)", "B, T, D", "eleman eleman kare"],
  [".mean(-1, keepdim=True)", "B, T, 1", "son eksen boyunca ortalama"],
  ["torch.rsqrt(… + self.eps)", "B, T, 1", "1 / √(ortalama + ε)"],
  ["x * rsqrt  (broadcast)", "B, T, D", "[T, 1] sütunu D boyunca yayılıyor"],
  ["_norm(x)", "B, T, D", "her satırın RMS'i artık 1"],
  ["output * (1.0 + self.weight)", "B, T, D", "her boyut kendi katsayısıyla"],
];

export function RmsSequence({ step }: { step: number }) {
  const { v, error } = useRealX();
  const [ref, W] = useWidth<HTMLDivElement>();
  if (!v) return <div ref={ref}><Loading what="gerçek embedding'ler" error={error} /></div>;
  const T = v.rows.length;
  const st = v.rows.map((r) => rmsnormSteps(r.x, v.w));
  const gap = 4;
  const cs = Math.max(26, Math.min(56, (W - 130 - 90) / (D + 2)));
  const x0 = 78, y0 = 56;
  const colX = x0 + D * (cs + gap) + 34;
  const gx = (d: number) => x0 + d * (cs + gap);
  const gy = (t: number) => y0 + t * (cs + gap);
  const H = y0 + T * (cs + gap) + 50;

  const cells: Cell[] = [];
  const maxOf = (a: number[]) => Math.max(...a.map(Math.abs));
  let max = 1;
  // x hücreleri (aynı id'ler bütün adımlarda yaşıyor)
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const s = st[t];
    const xv = s.x[d], sq = s.sq[d], nv = s.normed[d], out = s.out[d];
    let val = xv, text = f(xv), x = gx(d), y = gy(t), opacity = 1, tip: () => ReactNode;
    if (step === 1) { val = sq; text = f(sq); }
    if (step === 2 || step === 3) { x = colX; opacity = 0; val = sq; text = ""; }
    if (step === 4) { opacity = 0.0; }
    if (step === 5) { val = nv; text = f(nv); }
    if (step >= 6) { val = out; text = f(out); }
    tip = () => {
      const head = <div className="muted" style={{ marginBottom: 2 }}>[t={t} "{SHOW(v.rows[t].piece)}", d={d}]</div>;
      if (step === 0) return <>{head}x = <b>{fmtNum(xv, 4)}</b></>;
      if (step === 1) return <>{head}x² = {fmtNum(xv, 3)}² = <b>{fmtNum(sq, 3)}</b></>;
      if (step === 5) return <>{head}x · rsqrt = {fmtNum(xv, 3)} × {fmtNum(s.rsqrt, 4)} = <b>{fmtNum(nv, 4)}</b></>;
      return <>{head}normed · (1 + w[{d}]) = {fmtNum(nv, 3)} × {fmtNum(s.scale[d], 3)} = <b>{fmtNum(out, 4)}</b></>;
    };
    cells.push({ id: `x-${t}-${d}`, x, y, w: cs, h: cs, v: val, text: cs > 34 ? text : "", opacity, tip, group: `r${t}` });
  }
  max = step === 1 ? maxOf(st.flatMap((s) => s.sq)) : step === 5 ? 2.5 : step >= 6 ? maxOf(st.flatMap((s) => s.out)) : maxOf(st.flatMap((s) => s.x));
  // ortalama / rsqrt sütunu [T, 1]
  for (let t = 0; t < T; t++) {
    const s = st[t];
    const on = step >= 2 && step <= 4;
    const isR = step >= 3;
    cells.push({
      id: `m-${t}`, x: colX, y: gy(t), w: cs * 1.5, h: cs, opacity: on ? 1 : 0,
      fill: isR ? "var(--ax-T)" : "var(--fg-muted)", text: on ? (isR ? f(s.rsqrt) : f(s.meanSq)) : "", group: `r${t}`, z: 3,
      tip: () => isR ? <>rsqrt[t={t}] = 1/√({fmtNum(s.meanSq, 3)} + 1e-6) = <b>{fmtNum(s.rsqrt, 5)}</b></> : <>mean(x²) = ({s.sq.map((q) => f(q)).join(" + ")}) / {D} = <b>{fmtNum(s.meanSq, 3)}</b></>,
    });
    // broadcast kopyaları: sütundan grid'e uçuyor
    for (let d = 0; d < D; d++) {
      const bc = step === 4;
      cells.push({ id: `b-${t}-${d}`, x: bc ? gx(d) : colX, y: gy(t), w: cs, h: cs, fill: "var(--ax-T)", opacity: bc ? 0.85 : 0, text: bc && cs > 34 ? f(s.rsqrt) : "", group: `r${t}`, z: 2,
        tip: () => <>yayılmış kopya: rsqrt[t={t}] = {fmtNum(s.rsqrt, 4)}<div className="muted">bellekte kopya yok, sadece görünüm</div></> });
    }
  }
  // w satırı [D] → [1, 1, D]
  for (let d = 0; d < D; d++) {
    const on = step >= 6;
    cells.push({ id: `w-${d}`, x: gx(d), y: on ? y0 - cs - 14 : y0 - cs - 60, w: cs, h: cs * 0.7, fill: "var(--ax-hd)", opacity: on ? 1 : 0, text: on && cs > 34 ? f(1 + v.w[d]) : "", z: 3,
      tip: () => <>1 + w[{d}] = 1 + {fmtNum(v.w[d], 4)} = <b>{fmtNum(1 + v.w[d], 4)}</b><div className="muted">gerçek katman 0 input_layernorm</div></> });
  }

  const [code, shape, what] = STEP_CODE[Math.min(step, STEP_CODE.length - 1)];
  const rowRms = (t: number) => {
    const a = step === 5 ? st[t].normed : step >= 6 ? st[t].out : step === 1 ? st[t].sq : st[t].x;
    return Math.sqrt(a.reduce((s, q) => s + q * q, 0) / D);
  };
  const labels = (
    <>
      {v.rows.map((r, t) => (
        <div key={t} style={{ position: "absolute", left: 0, top: gy(t), height: cs, width: x0 - 10, display: "flex", alignItems: "center", justifyContent: "flex-end", fontFamily: "var(--mono)", fontSize: 12, color: "var(--ax-T)" }}>{SHOW(r.piece)}</div>
      ))}
      {(step === 0 || step === 5 || step >= 6) && v.rows.map((_, t) => (
        <div key={"r" + t} style={{ position: "absolute", left: colX, top: gy(t), height: cs, display: "flex", alignItems: "center", fontSize: 12, color: step === 5 ? "var(--ok)" : "var(--fg-muted)", fontFamily: "var(--mono)" }}>
          rms = {rowRms(t).toFixed(2)}
        </div>
      ))}
      <div style={{ position: "absolute", left: x0, top: y0 + T * (cs + gap) + 6, fontSize: 12, color: "var(--ax-D)" }}>D → (gerçekte 2048, burada ilk 8)</div>
      {step >= 2 && step <= 5 && <div style={{ position: "absolute", left: colX, top: y0 - 22, fontSize: 12, color: "var(--fg-muted)" }}>[T, 1]</div>}
      <div style={{ position: "absolute", left: 0, top: y0 - 22, width: x0 - 10, textAlign: "right", fontSize: 12, color: "var(--ax-T)" }}>T ↓</div>
    </>
  );
  return (
    <div ref={ref}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
        <code style={{ fontSize: 14 }}>{code}</code>
        <S s={shape} />
        <span className="small muted">{what}</span>
      </div>
      <CellCanvas cells={cells} width={W} height={H} max={max} labels={labels} />
      <div className="small muted">Hücrelerin üstüne gel: her sayının nasıl hesaplandığı görünür.</div>
    </div>
  );
}

/** Serbest laboratuvar: x ve w'yi sürükleyerek değiştir, bf16 ve Llama sırasıyla karşılaştır */
export function RmsPlayground() {
  const { v, error } = useRealX();
  const [ref, W] = useWidth<HTMLDivElement>();
  const [edit, setEdit] = useState<{ x?: number[][]; w?: number[] }>({});
  const [dtype, setDtype] = useState<"f32" | "bf16">("f32");
  const [order, setOrder] = useState<"gemma" | "llama">("gemma");
  if (!v) return <div ref={ref}><Loading what="gerçek embedding'ler" error={error} /></div>;
  const X = edit.x ?? v.rows.map((r) => r.x);
  const w = edit.w ?? v.w;
  const T = X.length;
  const res = X.map((x) => rmsnormSteps(x, w, 1e-6, dtype, order));
  const ref2 = X.map((x) => rmsnormSteps(x, w, 1e-6, dtype, order === "gemma" ? "llama" : "gemma"));
  const gap = 3;
  const cs = Math.max(22, Math.min(40, (W - 140) / (2 * D + 3)));
  const xL = 70, yT = 52, xR = xL + D * (cs + gap) + 30 + cs * 1.4 + 30;
  const gy = (t: number) => yT + t * (cs + gap);
  const H = yT + T * (cs + gap) + 30;
  const cells: Cell[] = [];
  const maxX = Math.max(...X.flat().map(Math.abs), 1);
  const maxO = Math.max(...res.flatMap((r) => r.out).map(Math.abs), 1);
  let ndiff = 0;
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const r = res[t];
    const diff = r.out[d] !== ref2[t].out[d];
    if (diff) ndiff++;
    cells.push({ id: `x${t}${d}`, x: xL + d * (cs + gap), y: gy(t), w: cs, h: cs, v: X[t][d] / maxX * 2.5, text: cs > 30 ? f(X[t][d]) : "", editable: true, group: `r${t}`,
      tip: () => <>x[{t},{d}] = <b>{fmtNum(X[t][d], 4)}</b><div className="muted">yukarı/aşağı sürükle</div></> });
    cells.push({ id: `o${t}${d}`, x: xR + d * (cs + gap), y: gy(t), w: cs, h: cs, v: r.out[d] / maxO * 2.5, text: cs > 30 ? f(r.out[d]) : "", ring: dtype === "bf16" && diff, group: `r${t}`,
      tip: () => <>
        <div>x · rsqrt · (1+w) = {fmtNum(r.x[d], 3)} × {fmtNum(r.rsqrt, 4)} × {fmtNum(r.scale[d], 3)}</div>
        <div>= <b>{r.out[d]}</b> <span className="muted">({dtype}, {order} sırası)</span></div>
        {dtype === "bf16" && <div className="muted">öbür sıra: {ref2[t].out[d]} {diff ? "· FARKLI" : "· aynı"}</div>}
      </> });
  }
  for (let t = 0; t < T; t++) cells.push({ id: `r${t}`, x: xL + D * (cs + gap) + 30, y: gy(t), w: cs * 1.4, h: cs, fill: "var(--ax-T)", text: f(res[t].rsqrt), group: `r${t}`,
    tip: () => <>rms(x) = {fmtNum(Math.sqrt(res[t].meanSq), 4)} → rsqrt = <b>{fmtNum(res[t].rsqrt, 5)}</b></> });
  for (let d = 0; d < D; d++) cells.push({ id: `w${d}`, x: xR + d * (cs + gap), y: yT - cs - 8, w: cs, h: cs * 0.7, fill: "var(--ax-hd)", text: cs > 30 ? f(1 + w[d]) : "", editable: true,
    tip: () => <>1 + w[{d}] = <b>{fmtNum(1 + w[d], 4)}</b><div className="muted">sürükle</div></> });
  const onDrag = (id: string, dy: number) => {
    if (id.startsWith("x")) {
      const t = +id[1], d = +id.slice(2);
      const nx = X.map((r) => r.slice());
      nx[t][d] = Math.round((nx[t][d] + dy * maxX / 120) * 100) / 100;
      setEdit({ ...edit, x: nx });
    } else if (id.startsWith("w")) {
      const d = +id.slice(1);
      const nw = w.slice();
      nw[d] = Math.round((nw[d] + dy / 60) * 100) / 100;
      setEdit({ ...edit, w: nw });
    }
  };
  const labels = <>
    <div style={{ position: "absolute", left: xL, top: yT - 26, fontSize: 12.5, fontWeight: 600 }}>x <span className="muted" style={{ fontWeight: 400 }}>(sürükle)</span></div>
    <div style={{ position: "absolute", left: xL + D * (cs + gap) + 30, top: yT - 26, fontSize: 12.5, fontWeight: 600 }}>rsqrt</div>
    <div style={{ position: "absolute", left: xR + D * (cs + gap) + 8, top: yT - cs - 4, fontSize: 12.5, color: "var(--ax-hd)" }}>1 + w</div>
    <div style={{ position: "absolute", left: xR, top: H - 22, fontSize: 12.5, fontWeight: 600 }}>çıktı</div>
    {X.map((_, t) => <div key={t} style={{ position: "absolute", left: 0, top: gy(t), height: cs, width: xL - 10, textAlign: "right", lineHeight: `${cs}px`, fontFamily: "var(--mono)", fontSize: 12, color: "var(--ax-T)" }}>{SHOW(v.rows[t].piece)}</div>)}
  </>;
  return (
    <div ref={ref}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-end", marginBottom: 10 }}>
        <div className="row" style={{ gap: 18 }}>
          <Seg value={dtype} onChange={setDtype} options={[["f32", "float32"], ["bf16", "bfloat16"]]} />
          <Seg value={order} onChange={setOrder} options={[["gemma", "Gemma sırası"], ["llama", "Llama sırası"]]} />
        </div>
        <div className="row" style={{ gap: 4 }}>
          <button className="btn" onClick={() => setEdit({ ...edit, x: X.map((r) => r.map((a) => a * 10)) })}>x × 10</button>
          <button className="btn" onClick={() => setEdit({ ...edit, w: w.map(() => 0) })}>w = 0</button>
          <button className="btn" onClick={() => setEdit({})}>gerçek değerler</button>
        </div>
      </div>
      <CellCanvas cells={cells} width={W} height={H} max={2.5} onDrag={onDrag} labels={labels} />
      <div className="small muted" style={{ marginTop: 6 }}>
        {dtype === "bf16"
          ? <>Halkalı hücreler: Gemma ve Llama sırasının farklı sonuç verdiği yerler — <b style={{ color: "var(--fg)" }}>{ndiff} / {T * D}</b>. Fark 1 ulp, ama <code>exact=True</code> testi bunu yakalıyor.</>
          : <>"x × 10" dene: rsqrt 10 kat küçülür, çıktı hiç değişmez. bfloat16'ya geçince iki işlem sırasının farkı görünür.</>}
      </div>
    </div>
  );
}

/** Mafs: d = 2'de RMSNorm (çember), (1+w) (elips), LayerNorm (iki nokta) */
export function NormPlane() {
  const [mode, setMode] = useState<"rms" | "w" | "ln">("rms");
  const [w0, setW0] = useState(0.6);
  const [w1, setW1] = useState(-0.4);
  const p = useMovablePoint([2.2, 0.8], { color: "var(--accent)" });
  const samples = useMemo(() => { const r = rng(4); return Array.from({ length: 36 }, () => [r.normal() * 1.4 + 0.4, r.normal() * 1.1 + 0.3] as [number, number]); }, []);
  const R = Math.SQRT2;
  const map = (q: [number, number]): [number, number] => {
    if (mode === "ln") { const l = layernorm(q); return [l[0], l[1]]; }
    const o = rmsnormSteps(q, mode === "w" ? [w0, w1] : [0, 0]).out;
    return [o[0], o[1]];
  };
  const img = map(p.point as [number, number]);
  const rms = Math.sqrt((p.x * p.x + p.y * p.y) / 2);
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-end", marginBottom: 10 }}>
        <Seg value={mode} onChange={setMode} options={[["rms", "RMSNorm"], ["w", "RMSNorm · (1+w)"], ["ln", "LayerNorm"]]} />
        {mode === "w" && (
          <div className="row small" style={{ gap: 16, alignItems: "center" }}>
            <label>1+w₀ = <b className="mono">{(1 + w0).toFixed(2)}</b> <input type="range" min={-1.5} max={1.5} step={0.05} value={w0} onChange={(e) => setW0(+e.target.value)} /></label>
            <label>1+w₁ = <b className="mono">{(1 + w1).toFixed(2)}</b> <input type="range" min={-1.5} max={1.5} step={0.05} value={w1} onChange={(e) => setW1(+e.target.value)} /></label>
          </div>
        )}
      </div>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
      <Mafs height={470} viewBox={{ x: [-4, 4], y: [-2.6, 2.6] }} preserveAspectRatio="contain">
        <Coordinates.Cartesian subdivisions={false} xAxis={{ lines: 1, labels: (n) => (n % 2 === 0 ? n : "") }} yAxis={{ lines: 1, labels: (n) => (n % 2 === 0 ? n : "") }} />
        {mode === "rms" && <Circle center={[0, 0]} radius={R} color="var(--ax-T)" fillOpacity={0.04} strokeStyle="dashed" />}
        {mode === "w" && <Ellipse center={[0, 0]} radius={[R * Math.abs(1 + w0), R * Math.abs(1 + w1)]} color="var(--ax-hd)" fillOpacity={0.04} strokeStyle="dashed" />}
        {mode === "ln" && <Line.ThroughPoints point1={[-2, 2]} point2={[2, -2]} color="var(--ax-D)" style="dashed" />}
        {samples.map((q, i) => { const m = map(q); return (
          <g key={i}>
            <Line.Segment point1={q} point2={m} color="var(--fg-faint)" opacity={0.25} weight={1} />
            <Point x={q[0]} y={q[1]} color="var(--fg-faint)" opacity={0.5} />
            <Point x={m[0]} y={m[1]} color={mode === "ln" ? "var(--ax-D)" : mode === "w" ? "var(--ax-hd)" : "var(--ax-T)"} opacity={0.7} />
          </g>
        ); })}
        <Line.Segment point1={p.point} point2={img} color="var(--accent)" weight={2} />
        <Point x={img[0]} y={img[1]} color="var(--accent)" />
        <Text x={img[0]} y={img[1]} attach="se" size={13} color="var(--accent)">norm(x)</Text>
        {p.element}
        <Text x={p.x} y={p.y} attach="ne" size={13} color="var(--accent)">x</Text>
      </Mafs>
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>
        x noktasını sürükle. x = ({p.x.toFixed(2)}, {p.y.toFixed(2)}), rms(x) = {rms.toFixed(3)} → norm(x) = ({img[0].toFixed(3)}, {img[1].toFixed(3)}).{" "}
        {mode === "rms" && "Nereye sürüklersen sürükle, görüntüsü √2 yarıçaplı çemberden ayrılmıyor: yön korunuyor, büyüklük atılıyor."}
        {mode === "w" && "(1+w) her ekseni ayrı ölçekliyor: çember elipse dönüşüyor. Bir katsayıyı 0'ın altına çek, o eksen ters döner."}
        {mode === "ln" && "LayerNorm önce ortalamayı çıkarıyor (x₀ + x₁ = 0 doğrusu), sonra ölçekliyor. d = 2'de geriye sadece iki nokta kalıyor: (1, −1) ve (−1, 1). RMSNorm bu bilgiyi silmiyor."}
      </div>
    </div>
  );
}

/** 3D: [B, T, D] → mean(-1, keepdim) → [B, T, 1] → broadcast. Her küp kimliğini koruyor */
const B3 = 2, T3 = 3, D3 = 6;
export function Keepdim3DScene({ step }: { step: number }) {
  const vals = useMemo(() => { const r = rng(9); return Array.from({ length: B3 * T3 * D3 }, () => r.normal()); }, []);
  const c = 0.5, g = 0.56;
  const pos = (b: number, t: number, d: number): [number, number, number] => [(d - D3 / 2) * g, (T3 / 2 - t) * g, (b - B3 / 2) * g * 1.7];
  const items: Cube[] = [];
  const col = (v: number) => (v >= 0 ? `hsl(18, ${40 + Math.min(1, v) * 40}%, ${70 - Math.min(1, v) * 22}%)` : `hsl(232, ${40 + Math.min(1, -v) * 40}%, ${72 - Math.min(1, -v) * 22}%)`);
  for (let b = 0; b < B3; b++) for (let t = 0; t < T3; t++) {
    const row = Array.from({ length: D3 }, (_, d) => vals[(b * T3 + t) * D3 + d]);
    const ms = row.reduce((a, q) => a + q * q, 0) / D3;
    for (let d = 0; d < D3; d++) {
      const v = row[d];
      const collapsed = step === 1;
      items.push({ id: `x${b}${t}${d}`, pos: collapsed ? pos(b, t, D3 + 0.8) : pos(b, t, d), color: col(v), hidden: collapsed, dim: step === 2,
        tip: () => <>x[b={b}, t={t}, d={d}] = <b>{v.toFixed(3)}</b></> });
    }
    // ortalama küpü
    items.push({ id: `m${b}${t}`, pos: pos(b, t, D3 + 0.8), color: "#2a78d6", hidden: step === 0,
      tip: () => <>mean(x²)[b={b}, t={t}] = <b>{ms.toFixed(3)}</b><div className="muted">şekil [B, T, 1]</div></> });
    // yayılmış kopyalar
    for (let d = 0; d < D3; d++) items.push({ id: `c${b}${t}${d}`, pos: step === 2 ? [pos(b, t, d)[0], pos(b, t, d)[1] + 0.0, pos(b, t, d)[2] + 0.3] : pos(b, t, D3 + 0.8), color: "#2a78d6", hidden: step !== 2, size: [c * 0.7, c * 0.7, c * 0.25],
      tip: () => <>yayılmış: rms[b={b}, t={t}] (aynı sayı, D kez)</> });
  }
  const views: View[] = [
    { pos: [2.4, 2.6, 5.2], target: [0.3, 0, 0] },
    { pos: [4.6, 1.8, 4.2], target: [1.5, 0, 0] },
    { pos: [1.2, 2.2, 5.6], target: [0.5, 0, 0] },
  ];
  return (
    <Stage3D view={views[Math.min(step, 2)]} height={420}>
      <Cubes items={items} cell={c} />
      <Tag position={[0, (T3 / 2) * g + 0.55, -B3 * g * 0.9]} color="var(--ax-D)">D →</Tag>
      <Tag position={[-(D3 / 2) * g - 0.7, 0, 0]} color="var(--ax-T)">T ↓</Tag>
      <Tag position={[-(D3 / 2) * g - 0.5, -(T3 / 2) * g - 0.4, B3 * g]} color="var(--ax-B)">B ↗</Tag>
      {step >= 1 && <Tag position={[(D3 / 2 + 0.8) * g, (T3 / 2) * g + 0.55, 0]} color="var(--ax-T)">{step === 1 ? "[B, T, 1]" : "[B, T, 1] → [B, T, D]"}</Tag>}
    </Stage3D>
  );
}
