// 4.4: the RMSNorm scrolly, the draggable lab, Mafs geometry, 3D keepdim.
import { ReactNode, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { PivotControls } from "@react-three/drei";
import { easing } from "maath";
import * as THREE from "three";
import { Mafs, Coordinates, Circle, Ellipse, Line, Point, Text, useMovablePoint } from "mafs";
import { useData, rowOf } from "../../../lib/data";
import { rmsnormSteps, layernorm, rng, fmtNum } from "../../../lib/tensor";
import { useWidth } from "../../../components/charts";
import { Cell, CellCanvas } from "../../../interact/cells";
import { Stage3D, Cubes, Cube, Tag, View, Axes3D, cssColor } from "../../../interact/three";
import { Loading, S, Seg } from "../../../components/ui";
import type { PgSlices } from "./types";
import { useLang, useT } from "../../../i18n";
import type { L } from "../../../i18n";

const D = 8;
const TOK = ["<bos>", "caption", " tr", "\n"];
const SHOW = (p: string) => p.replace("▁", "␣").replace("\n", "\\n");

/** Real data: embeddings of 4 prompt tokens (×√2048, first 8 dims) and the layer 0 input_layernorm weight */
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

const STEP_CODE: [string, string, L<string>][] = [
  ["x", "B, T, D", { en: "input", tr: "girdi" }],
  ["x.pow(2)", "B, T, D", { en: "elementwise square", tr: "eleman eleman kare" }],
  [".mean(-1, keepdim=True)", "B, T, 1", { en: "mean over the last axis", tr: "son eksen boyunca ortalama" }],
  ["torch.rsqrt(… + self.eps)", "B, T, 1", { en: "1 / √(mean + ε)", tr: "1 / √(ortalama + ε)" }],
  ["x * rsqrt  (broadcast)", "B, T, D", { en: "the [T, 1] column is broadcast along D", tr: "[T, 1] sütunu D boyunca yayılıyor" }],
  ["_norm(x)", "B, T, D", { en: "every row now has RMS 1", tr: "her satırın RMS'i artık 1" }],
  ["output * (1.0 + self.weight)", "B, T, D", { en: "each dim with its own coefficient", tr: "her boyut kendi katsayısıyla" }],
];

export function RmsSequence({ step }: { step: number }) {
  const lang = useLang();
  const tr = useT();
  const { v, error } = useRealX();
  const [ref, W] = useWidth<HTMLDivElement>();
  if (!v) return <div ref={ref}><Loading what={tr("real embeddings", "gerçek embedding'ler")} error={error} /></div>;
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
  // x cells (the same ids live through all steps)
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
  // the mean / rsqrt column [T, 1]
  for (let t = 0; t < T; t++) {
    const s = st[t];
    const on = step >= 2 && step <= 4;
    const isR = step >= 3;
    cells.push({
      id: `m-${t}`, x: colX, y: gy(t), w: cs * 1.5, h: cs, opacity: on ? 1 : 0,
      fill: isR ? "var(--ax-T)" : "var(--fg-muted)", text: on ? (isR ? f(s.rsqrt) : f(s.meanSq)) : "", group: `r${t}`, z: 3,
      tip: () => isR ? <>rsqrt[t={t}] = 1/√({fmtNum(s.meanSq, 3)} + 1e-6) = <b>{fmtNum(s.rsqrt, 5)}</b></> : <>mean(x²) = ({s.sq.map((q) => f(q)).join(" + ")}) / {D} = <b>{fmtNum(s.meanSq, 3)}</b></>,
    });
    // broadcast copies: they fly from the column into the grid
    for (let d = 0; d < D; d++) {
      const bc = step === 4;
      cells.push({ id: `b-${t}-${d}`, x: bc ? gx(d) : colX, y: gy(t), w: cs, h: cs, fill: "var(--ax-T)", opacity: bc ? 0.85 : 0, text: bc && cs > 34 ? f(s.rsqrt) : "", group: `r${t}`, z: 2,
        tip: () => <>{tr("broadcast copy", "yayılmış kopya")}: rsqrt[t={t}] = {fmtNum(s.rsqrt, 4)}<div className="muted">{tr("no copy in memory, just a view", "bellekte kopya yok, sadece görünüm")}</div></> });
    }
  }
  // the w row [D] → [1, 1, D]
  for (let d = 0; d < D; d++) {
    const on = step >= 6;
    cells.push({ id: `w-${d}`, x: gx(d), y: on ? y0 - cs - 14 : y0 - cs - 60, w: cs, h: cs * 0.7, fill: "var(--ax-hd)", opacity: on ? 1 : 0, text: on && cs > 34 ? f(1 + v.w[d]) : "", z: 3,
      tip: () => <>1 + w[{d}] = 1 + {fmtNum(v.w[d], 4)} = <b>{fmtNum(1 + v.w[d], 4)}</b><div className="muted">{tr("real layer 0 input_layernorm", "gerçek katman 0 input_layernorm")}</div></> });
  }

  const [code, shape, whatL] = STEP_CODE[Math.min(step, STEP_CODE.length - 1)];
  const what = whatL[lang];
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
      <div style={{ position: "absolute", left: x0, top: y0 + T * (cs + gap) + 6, fontSize: 12, color: "var(--ax-D)" }}>D → ({tr("2048 in reality, the first 8 here", "gerçekte 2048, burada ilk 8")})</div>
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
      <div className="small muted">{tr("Hover the cells to see how each number is computed.", "Hücrelerin üstüne gel: her sayının nasıl hesaplandığı görünür.")}</div>
    </div>
  );
}

/** Free lab: drag x and w to change them, compare bf16 and the Llama order */
export function RmsPlayground() {
  const tr = useT();
  const { v, error } = useRealX();
  const [ref, W] = useWidth<HTMLDivElement>();
  const [edit, setEdit] = useState<{ x?: number[][]; w?: number[] }>({});
  const [dtype, setDtype] = useState<"f32" | "bf16">("f32");
  const [order, setOrder] = useState<"gemma" | "llama">("gemma");
  if (!v) return <div ref={ref}><Loading what={tr("real embeddings", "gerçek embedding'ler")} error={error} /></div>;
  const X = edit.x ?? v.rows.map((r) => r.x);
  const w = edit.w ?? v.w;
  const T = X.length;
  const res = X.map((x) => rmsnormSteps(x, w, 1e-6, dtype, order));
  const ref2 = X.map((x) => rmsnormSteps(x, w, 1e-6, dtype, order === "gemma" ? "llama" : "gemma"));
  // The layout reads as an equation from left to right: x × rsqrt × (1 + w) = output
  const gap = 3, op = 34;
  const xL = 64;
  const cs = Math.max(20, Math.min(40, (W - xL - 4 * op - 3 * gap * D) / (3 * D + 1.4)));
  const blockW = D * (cs + gap) - gap;
  const xR = xL + blockW + op, xWv = xR + cs * 1.4 + op, xO = xWv + blockW + op;
  const yT = 34;
  const gy = (t: number) => yT + t * (cs + gap);
  const H = yT + T * (cs + gap) + 8;
  const yMid = yT + (T * (cs + gap) - gap) / 2;
  const cells: Cell[] = [];
  const maxX = Math.max(...X.flat().map(Math.abs), 1);
  const maxO = Math.max(...res.flatMap((r) => r.out).map(Math.abs), 1);
  let ndiff = 0;
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const r = res[t];
    const diff = r.out[d] !== ref2[t].out[d];
    if (diff) ndiff++;
    cells.push({ id: `x${t}${d}`, x: xL + d * (cs + gap), y: gy(t), w: cs, h: cs, v: X[t][d] / maxX * 2.5, text: cs > 30 ? f(X[t][d]) : "", editable: true, group: `r${t}`,
      tip: () => <>x[{t},{d}] = <b>{fmtNum(X[t][d], 4)}</b><div className="muted">{tr("drag up/down", "yukarı/aşağı sürükle")}</div></> });
    cells.push({ id: `o${t}${d}`, x: xO + d * (cs + gap), y: gy(t), w: cs, h: cs, v: r.out[d] / maxO * 2.5, text: cs > 30 ? f(r.out[d]) : "", ring: dtype === "bf16" && diff, group: `r${t}`,
      tip: () => <>
        <div>x · rsqrt · (1+w) = {fmtNum(r.x[d], 3)} × {fmtNum(r.rsqrt, 4)} × {fmtNum(r.scale[d], 3)}</div>
        <div>= <b>{r.out[d]}</b> <span className="muted">({dtype}, {order} {tr("order", "sırası")})</span></div>
        {dtype === "bf16" && <div className="muted">{tr("other order", "öbür sıra")}: {ref2[t].out[d]} {diff ? tr("· DIFFERENT", "· FARKLI") : tr("· same", "· aynı")}</div>}
      </> });
  }
  for (let t = 0; t < T; t++) cells.push({ id: `r${t}`, x: xR, y: gy(t), w: cs * 1.4, h: cs, fill: "var(--ax-T)", text: f(res[t].rsqrt), group: `r${t}`,
    tip: () => <>rms(x) = {fmtNum(Math.sqrt(res[t].meanSq), 4)} → 1 / rms = <b>{fmtNum(res[t].rsqrt, 5)}</b><div className="muted">{tr("every element of the row is multiplied by this", "satırın bütün elemanları bununla çarpılıyor")}</div></> });
  for (let d = 0; d < D; d++) cells.push({ id: `w${d}`, x: xWv + d * (cs + gap), y: yMid - cs / 2, w: cs, h: cs, fill: "var(--ax-hd)", text: cs > 30 ? f(1 + w[d]) : "", editable: true,
    tip: () => <>1 + w[{d}] = <b>{fmtNum(1 + w[d], 4)}</b><div className="muted">{tr(`column ${d} of every row is multiplied by this · drag`, `her satırın ${d}. sütunu bununla çarpılıyor · sürükle`)}</div></> });
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
  const head = (x: number, wd: number, title: ReactNode, sub: string, color = "var(--fg)") => (
    <div style={{ position: "absolute", left: x, width: wd, top: 0, textAlign: "center", fontSize: 12.5, lineHeight: 1.25, whiteSpace: "nowrap" }}>
      <b style={{ color }}>{title}</b> <span className="muted mono" style={{ fontSize: 11 }}>{sub}</span>
    </div>
  );
  const opSign = (x: number, s: string) => <div style={{ position: "absolute", left: x, width: op, top: yMid - 14, textAlign: "center", fontSize: 24, lineHeight: "28px", color: "var(--fg-faint)", fontWeight: 300 }}>{s}</div>;
  const labels = <>
    {head(xL, blockW, <>x <span style={{ fontWeight: 400 }}>↕</span></>, "[T, D]")}
    {head(xR - 10, cs * 1.4 + 20, "1 / rms", "[T, 1]", "var(--ax-T)")}
    {head(xWv, blockW, <>1 + w <span style={{ fontWeight: 400 }}>↕</span></>, "[D]", "var(--ax-hd)")}
    {head(xO, blockW, tr("output", "çıktı"), "[T, D]")}
    {opSign(xL + blockW, "×")}{opSign(xR + cs * 1.4, "×")}{opSign(xWv + blockW, "=")}
    {X.map((_, t) => <div key={t} style={{ position: "absolute", left: 0, top: gy(t), height: cs, width: xL - 10, textAlign: "right", lineHeight: `${cs}px`, fontFamily: "var(--mono)", fontSize: 12, color: "var(--ax-T)" }}>{SHOW(v.rows[t].piece)}</div>)}
  </>;
  return (
    <div ref={ref}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-end", marginBottom: 10 }}>
        <div className="row" style={{ gap: 18 }}>
          <Seg value={dtype} onChange={setDtype} options={[["f32", "float32"], ["bf16", "bfloat16"]]} />
          <Seg value={order} onChange={setOrder} options={[["gemma", tr("Gemma order", "Gemma sırası")], ["llama", tr("Llama order", "Llama sırası")]]} />
        </div>
        <div className="row" style={{ gap: 4 }}>
          <button className="btn" onClick={() => setEdit({ ...edit, x: X.map((r) => r.map((a) => a * 10)) })}>x × 10</button>
          <button className="btn" onClick={() => setEdit({ ...edit, w: w.map(() => 0) })}>w = 0</button>
          <button className="btn" onClick={() => setEdit({})}>{tr("real values", "gerçek değerler")}</button>
        </div>
      </div>
      <CellCanvas cells={cells} width={W} height={H} max={2.5} onDrag={onDrag} labels={labels} />
      <div className="small muted" style={{ marginTop: 6 }}>
        <b style={{ color: "var(--fg)" }}>↕</b> {tr("The marked blocks (x and 1 + w) can be dragged up and down.", "işaretli bloklar (x ve 1 + w) yukarı-aşağı sürüklenebilir.")}{" "}
        {dtype === "bf16"
          ? <><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2, outline: "2px solid var(--accent)", outlineOffset: 1, verticalAlign: -1, margin: "0 5px 0 3px" }} />{tr(<>Ringed cells: where the Gemma and Llama orders give different results: <b style={{ color: "var(--fg)" }}>{ndiff} / {T * D}</b>. The difference is 1 ulp, but an <code>exact=True</code> test catches it.</>, <>Halkalı hücreler: Gemma ve Llama sırasının farklı sonuç verdiği yerler: <b style={{ color: "var(--fg)" }}>{ndiff} / {T * D}</b>. Fark 1 ulp, ama <code>exact=True</code> testi bunu yakalıyor.</>)}</>
          : tr(<>Try "x × 10": rsqrt shrinks 10×, the output does not change at all. Switch to bfloat16 to see the difference between the two orders.</>, <>"x × 10" dene: rsqrt 10 kat küçülür, çıktı hiç değişmez. bfloat16'ya geçince iki işlem sırasının farkı görünür.</>)}
      </div>
    </div>
  );
}

type NormMode = "rms" | "w" | "ln";

/** Geometry: controls on top (mode, dims, (1+w) sliders), a 3D or 2D scene below */
export function NormGeometry() {
  const tr = useT();
  const [mode, setMode] = useState<NormMode>("rms");
  const [dim, setDim] = useState<3 | 2>(3);
  const [w, setW] = useState([0.6, -0.4, 0.2]);
  const n = dim;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-end", marginBottom: 10, gap: 16, flexWrap: "wrap" }}>
        <div className="row" style={{ gap: 18 }}>
          <Seg value={mode} onChange={setMode} options={[["rms", "RMSNorm"], ["w", "RMSNorm · (1+w)"], ["ln", "LayerNorm"]]} />
          <Seg value={dim} onChange={setDim} options={[[3, tr("3 dims", "3 boyut")], [2, tr("2 dims", "2 boyut")]]} />
        </div>
        {mode === "w" && (
          <div className="row small" style={{ gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            {w.slice(0, n).map((wi, i) => (
              <label key={i}>1+w{"₀₁₂"[i]} = <b className="mono">{(1 + wi).toFixed(2)}</b> <input type="range" min={-1.5} max={1.5} step={0.05} value={wi} onChange={(e) => setW(w.map((q, j) => (j === i ? +e.target.value : q)))} /></label>
            ))}
          </div>
        )}
      </div>
      {dim === 3 ? <NormSpace3D mode={mode} w={w} /> : <NormPlane mode={mode} w={w} />}
    </div>
  );
}

const normMap = (mode: NormMode, q: number[], w: number[]) => mode === "ln" ? layernorm(q) : rmsnormSteps(q, mode === "w" ? w.slice(0, q.length) : q.map(() => 0)).out;

/** d = 3: RMSNorm maps onto a sphere (radius √3), (1+w) onto an ellipsoid, LayerNorm onto a circle in the plane x+y+z=0 */
function NormSpace3D({ mode, w }: { mode: NormMode; w: number[] }) {
  const tr = useT();
  const samples = useMemo(() => { const r = rng(7); return Array.from({ length: 60 }, () => [r.normal() * 1.3 + 0.5, r.normal() * 1.0 + 0.3, r.normal() * 1.1 + 0.2]); }, []);
  const [x, setX] = useState([2.2, 1.1, 0.6]);
  const R = Math.sqrt(3);
  const color = mode === "ln" ? "var(--ax-D)" : mode === "w" ? "var(--ax-hd)" : "var(--ax-T)";
  const items: Cube[] = [];
  samples.forEach((q, i) => {
    const m = normMap(mode, q, w);
    items.push({ id: "i" + i, pos: q as any, size: [0.07, 0.07, 0.07], color: cssColor("--fg-faint"), tip: () => <>{tr("input", "girdi")} ({q.map((a) => a.toFixed(2)).join(", ")})<div className="muted">rms = {Math.sqrt((q[0] ** 2 + q[1] ** 2 + q[2] ** 2) / 3).toFixed(2)}</div></> });
    items.push({ id: "o" + i, pos: m as any, size: [0.11, 0.11, 0.11], color: cssColor(color.slice(4, -1)), tip: () => <>norm = ({m.map((a) => a.toFixed(2)).join(", ")})<div className="muted">|norm| = {Math.hypot(...m).toFixed(3)}</div></> });
  });
  const img = normMap(mode, x, w);
  const sc: [number, number, number] = mode === "w" ? [Math.abs(1 + w[0]) * R, Math.abs(1 + w[1]) * R, Math.abs(1 + w[2]) * R] : [R, R, R];
  const nrm = new THREE.Vector3(1, 1, 1).normalize();
  const planeQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), nrm);
  return (
    <div>
      <Stage3D view={{ pos: [6.4, 4.6, 7.6], target: [0, 0.2, 0] }} height={500} minDist={3} maxDist={20}>
        <Axes3D origin={[0, 0, 0]} axes={[
          { dir: [1, 0, 0], len: 3.2, color: "var(--fg-faint)", label: "x₀" },
          { dir: [0, 1, 0], len: 3.2, color: "var(--fg-faint)", label: "x₁" },
          { dir: [0, 0, 1], len: 3.2, color: "var(--fg-faint)", label: "x₂" },
        ]} />
        {mode !== "ln" && <EaseScale target={sc}>
          <mesh><sphereGeometry args={[1, 48, 32]} /><meshStandardMaterial color={cssColor(color.slice(4, -1))} transparent opacity={0.1} depthWrite={false} /></mesh>
          <mesh><sphereGeometry args={[1.002, 24, 16]} /><meshBasicMaterial color={cssColor(color.slice(4, -1))} wireframe transparent opacity={0.16} /></mesh>
        </EaseScale>}
        {mode === "ln" && <group quaternion={planeQ}>
          <mesh><circleGeometry args={[3.4, 64]} /><meshBasicMaterial color={cssColor("--ax-D")} transparent opacity={0.07} side={THREE.DoubleSide} depthWrite={false} /></mesh>
          <mesh><torusGeometry args={[R, 0.018, 8, 96]} /><meshBasicMaterial color={cssColor("--ax-D")} /></mesh>
        </group>}
        <Cubes items={items} tween={{ duration: 0.9, stagger: 0.004 }} />
        <DragPoint value={x} onChange={setX} />
        <Segment a={x} b={img} />
        <mesh position={img as any}><sphereGeometry args={[0.11, 20, 20]} /><meshBasicMaterial color={cssColor("--accent")} /></mesh>
        <Tag position={[img[0], img[1] + 0.3, img[2]]} color="var(--accent)" weight={650}>norm(x)</Tag>
      </Stage3D>
      <div className="small muted" style={{ marginTop: 8 }}>
        {tr("Drag the big blue point by its arrows.", "Büyük mavi noktayı oklarıyla sürükle.")} x = ({x.map((a) => a.toFixed(2)).join(", ")}), rms(x) = {Math.sqrt((x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 3).toFixed(3)} → norm(x) = ({img.map((a) => a.toFixed(2)).join(", ")}).{" "}
        {mode === "rms" && tr("All points land on the sphere of radius √3: direction is kept, magnitude is dropped. At d = 2048 the radius is √2048 ≈ 45.25.", "Bütün noktalar √3 yarıçaplı küreye iniyor: yön korunuyor, büyüklük atılıyor. d = 2048'de küre yarıçapı √2048 ≈ 45.25.")}
        {mode === "w" && tr("(1+w) scales each axis separately: the sphere becomes an ellipsoid. Pull a coefficient below 0 and that axis flips.", "(1+w) her ekseni ayrı ölçekliyor: küre elipsoide dönüşüyor. Bir katsayıyı 0'ın altına çek, o eksen ters döner.")}
        {mode === "ln" && tr("LayerNorm subtracts the mean first: the points drop onto the plane x₀ + x₁ + x₂ = 0. Then it scales them onto the circle of radius √3 in that plane. The shared shift of all dims, the (1, 1, 1) direction, is erased. RMSNorm keeps it.", "LayerNorm önce ortalamayı çıkarıyor: noktalar x₀ + x₁ + x₂ = 0 düzlemine iniyor. Sonra ölçekliyor: düzlemdeki √3 yarıçaplı çembere. Bütün boyutların ortak kayması (1, 1, 1) yönü silindi. RMSNorm bu bilgiyi koruyor.")}
      </div>
    </div>
  );
}

/** Smoothly moves its children's scale to the target (sphere ↔ ellipsoid) */
function EaseScale({ target, children }: { target: [number, number, number]; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null);
  const tv = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dt) => { if (ref.current) easing.damp3(ref.current.scale, tv.set(...target), 0.3, dt); });
  return <group ref={ref} scale={target}>{children}</group>;
}

function Segment({ a, b }: { a: number[]; b: number[] }) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  geo.setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
  return <lineSegments geometry={geo}><lineBasicMaterial color={cssColor("--accent")} /></lineSegments>;
}

/** A point dragged by its axis arrows; camera control is off while dragging */
function DragPoint({ value, onChange }: { value: number[]; onChange: (v: number[]) => void }) {
  const controls = useThree((s) => s.controls) as any;
  const m = useMemo(() => new THREE.Matrix4().makeTranslation(value[0], value[1], value[2]), []);
  const p = useMemo(() => new THREE.Vector3(), []);
  return (
    <PivotControls matrix={m} autoTransform disableRotations disableScaling disableSliders scale={0.9} depthTest={false} lineWidth={2.5}
      onDragStart={() => controls && (controls.enabled = false)} onDragEnd={() => controls && (controls.enabled = true)}
      onDrag={(l) => { p.setFromMatrixPosition(l); onChange([+p.x.toFixed(2), +p.y.toFixed(2), +p.z.toFixed(2)]); }}>
      <mesh><sphereGeometry args={[0.13, 24, 24]} /><meshStandardMaterial color={cssColor("--accent")} /></mesh>
    </PivotControls>
  );
}

/** Mafs: at d = 2, RMSNorm (circle), (1+w) (ellipse), LayerNorm (two points) */
function NormPlane({ mode, w }: { mode: NormMode; w: number[] }) {
  const tr = useT();
  const [w0, w1] = w;
  const p = useMovablePoint([2.2, 0.8], { color: "var(--accent)" });
  const samples = useMemo(() => { const r = rng(4); return Array.from({ length: 36 }, () => [r.normal() * 1.4 + 0.4, r.normal() * 1.1 + 0.3] as [number, number]); }, []);
  const R = Math.SQRT2;
  const map = (q: [number, number]): [number, number] => { const o = normMap(mode, q, w); return [o[0], o[1]]; };
  const img = map(p.point as [number, number]);
  const rms = Math.sqrt((p.x * p.x + p.y * p.y) / 2);
  return (
    <div>
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
        <Text x={img[0]} y={img[1]} attach={img[1] >= 0 ? "n" : "s"} attachDistance={18} size={13} color="var(--accent)">norm(x)</Text>
        {p.element}
        <Text x={p.x} y={p.y} attach="ne" attachDistance={14} size={13} color="var(--accent)">x</Text>
      </Mafs>
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>
        {tr("Drag the point x.", "x noktasını sürükle.")} x = ({p.x.toFixed(2)}, {p.y.toFixed(2)}), rms(x) = {rms.toFixed(3)} → norm(x) = ({img[0].toFixed(3)}, {img[1].toFixed(3)}).{" "}
        {mode === "rms" && tr("Wherever you drag it, its image stays on the circle of radius √2: direction is kept, magnitude is dropped.", "Nereye sürüklersen sürükle, görüntüsü √2 yarıçaplı çemberden ayrılmıyor: yön korunuyor, büyüklük atılıyor.")}
        {mode === "w" && tr("(1+w) scales each axis separately: the circle becomes an ellipse. Pull a coefficient below 0 and that axis flips.", "(1+w) her ekseni ayrı ölçekliyor: çember elipse dönüşüyor. Bir katsayıyı 0'ın altına çek, o eksen ters döner.")}
        {mode === "ln" && tr("At d = 2 only two points survive LayerNorm: (1, −1) and (−1, 1). That is misleadingly harsh; switch to 3 dims to see what really happens.", "d = 2'de LayerNorm'dan geriye sadece iki nokta kalıyor: (1, −1) ve (−1, 1). Bu yanıltıcı derecede sert. 3 boyuta geçince ne olduğunu daha iyi görürsün.")}
      </div>
    </div>
  );
}

/** 3D: [B, T, D] → mean(-1, keepdim) → [B, T, 1] → broadcast. Every cube keeps its identity */
const B3 = 2, T3 = 3, D3 = 6;
export function Keepdim3DScene({ step }: { step: number }) {
  const tr = useT();
  const vals = useMemo(() => { const r = rng(9); return Array.from({ length: B3 * T3 * D3 }, () => r.normal()); }, []);
  const c = 0.5, g = 0.56, pad = 0.1;
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
    // the mean cube
    items.push({ id: `m${b}${t}`, pos: pos(b, t, D3 + 0.8), color: "#2a78d6", hidden: step === 0,
      tip: () => <>mean(x²)[b={b}, t={t}] = <b>{ms.toFixed(3)}</b><div className="muted">{tr("shape", "şekil")} [B, T, 1]</div></> });
    // broadcast copies
    for (let d = 0; d < D3; d++) items.push({ id: `c${b}${t}${d}`, pos: step === 2 ? [pos(b, t, d)[0], pos(b, t, d)[1] + 0.0, pos(b, t, d)[2] + 0.3] : pos(b, t, D3 + 0.8), color: "#2a78d6", hidden: step !== 2, size: [c * 0.7, c * 0.7, c * 0.25],
      tip: () => <>{tr("broadcast", "yayılmış")}: rms[b={b}, t={t}] ({tr("the same number, D times", "aynı sayı, D kez")})</> });
  }
  const views: View[] = [
    { pos: [2.0, 2.3, 4.6], target: [0.1, 0.1, -0.4] },
    { pos: [3.8, 2.0, 4.0], target: [1.3, 0.1, -0.4] },
    { pos: [1.1, 2.1, 4.9], target: [0.3, 0.1, -0.4] },
  ];
  return (
    <Stage3D view={views[Math.min(step, 2)]} height={420}>
      <Cubes items={items} cell={c} tween={{ duration: 0.8, stagger: 0.008 }} />
      {/* The axes start at the block's top-left-front corner: D to the right, T down, B back */}
      <Axes3D origin={[(step === 1 ? D3 + 0.8 - D3 / 2 : -D3 / 2) * g - c / 2 - pad, (T3 / 2) * g + c / 2 + pad, (B3 / 2 - 1) * g * 1.7 + c / 2 + pad]} axes={[
        { dir: [1, 0, 0], len: (step === 1 ? 1 : D3) * g + pad, ticks: step === 1 ? 1 : D3, color: "var(--ax-D)", label: step === 1 ? "D = 1" : `D = ${D3}`, labelOffset: [-0.45, 0.3, 0] },
        { dir: [0, -1, 0], len: T3 * g + pad, ticks: T3, color: "var(--ax-T)", label: `T = ${T3}` },
        { dir: [0, 0, -1], len: (B3 - 1) * g * 1.7 + c + pad, ticks: B3, color: "var(--ax-B)", label: `B = ${B3}` },
      ]} />
      {step >= 1 && <Tag position={[(D3 / 2 + 0.8) * g, (T3 / 2) * g + 0.75, 0.4]} color="var(--ax-T)" weight={650}>{step === 1 ? "mean(-1, keepdim=True) → [B, T, 1]" : "broadcast: [B, T, 1] → [B, T, D]"}</Tag>}
    </Stage3D>
  );
}
