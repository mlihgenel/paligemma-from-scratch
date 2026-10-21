// R3: Bölüm 4'in 3D scrolly sahneleri. Hepsi interact/three (Stage3D + instanced Cubes) üzerinde.
// Kural: her adımda tek bir şey hareket eder; geri kalanı yerinde ve soluk durur.
import { ReactNode, useMemo } from "react";
import * as THREE from "three";
import { rng, geluTanh } from "../../../lib/tensor";
import { Stage3D, Cubes, Cube, Tag, View, cssColor } from "../../../interact/three";
import { valueColor } from "../../../interact/cells";

const fx = (v: number, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));
const rmsOf = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
const vc = (v: number, max: number) => valueColor(v, max, document.documentElement.dataset.theme === "dark");

/** Bir vektörü x konumunda, z ekseni boyunca dizilmiş işaretli çubuklar olarak çiz */
function barItems(prefix: string, vals: number[], at: [number, number, number], color: string, k: number, opts: { hidden?: boolean; dim?: boolean; tip?: (i: number) => ReactNode } = {}): Cube[] {
  return vals.map((v, i) => {
    const h = Math.max(0.03, Math.abs(v) * k);
    return {
      id: `${prefix}${i}`,
      pos: [at[0] + (i - (vals.length - 1) / 2) * 0.13, at[1] + (v >= 0 ? h / 2 : -h / 2), at[2]],
      size: [0.1, h, 0.1], color: v >= 0 ? color : cssColor("--fg-faint"),
      hidden: opts.hidden, dim: opts.dim, tip: opts.tip ? () => opts.tip!(i) : undefined,
    };
  });
}

/* ------------------------------------------------------------------ 4.2 Otoyol */
const HD = 10;
const smooth = (x: number) => { const k = Math.min(1, Math.max(0, x)); return k * k * (3 - 2 * k); };
const mix = (a: number[], b: number[], k: number) => a.map((v, i) => v + (b[i] - v) * k);

// Yan yol: otoyoldan kalkıp norm ve f bloklarından geçen, ⊕'de otoyola inen eğri
const BR = [
  { from: -5.2, norm: -4.0, f: -2.2, add: -1.0, kind: "attn" as const },
  { from: 0.4, norm: 1.6, f: 3.4, add: 4.6, kind: "mlp" as const },
];
const BY = 2.1;
function branchCurve(b: (typeof BR)[number]) {
  return new THREE.CatmullRomCurve3([
    new THREE.Vector3(b.from, 0.05, 0), new THREE.Vector3(b.from + 0.55, 1.1, 0), new THREE.Vector3(b.norm, BY, 0),
    new THREE.Vector3((b.norm + b.f) / 2, BY, 0), new THREE.Vector3(b.f, BY, 0), new THREE.Vector3(b.add - 0.55, 1.1, 0), new THREE.Vector3(b.add, 0.05, 0),
  ], false, "centripetal");
}
// eğri üzerinde norm bloğu ≈ 0.33, f bloğu ≈ 0.67
const T_NORM = 0.33, T_F = 0.67;

function Road({ curve, color, on }: { curve: THREE.Curve<THREE.Vector3>; color: string; on: boolean }) {
  const geo = useMemo(() => new THREE.TubeGeometry(curve, 80, 0.045, 8, false), [curve]);
  return <mesh geometry={geo}><meshStandardMaterial color={color} transparent opacity={on ? 0.55 : 0.18} /></mesh>;
}

function Block({ x, color, on, w = 1.3 }: { x: number; color: string; on: boolean; w?: number }) {
  return (
    <mesh position={[x, BY - 0.02, 0]}>
      <boxGeometry args={[w, 0.08, 1.0]} />
      <meshStandardMaterial color={color} transparent opacity={on ? 0.75 : 0.22} />
    </mesh>
  );
}

function AddNode({ x, on }: { x: number; on: boolean }) {
  const col = cssColor("--accent");
  return (
    <group position={[x, 0.02, 0.75]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[0.2, 0.035, 10, 32]} /><meshBasicMaterial color={col} transparent opacity={on ? 1 : 0.35} /></mesh>
      <mesh><boxGeometry args={[0.24, 0.035, 0.035]} /><meshBasicMaterial color={col} transparent opacity={on ? 1 : 0.35} /></mesh>
      <mesh rotation={[0, Math.PI / 2, 0]}><boxGeometry args={[0.24, 0.035, 0.035]} /><meshBasicMaterial color={col} transparent opacity={on ? 1 : 0.35} /></mesh>
    </group>
  );
}

export function HighwayScene({ step, progress = 0 }: { step: number; progress?: number }) {
  const v = useMemo(() => {
    const r = rng(21);
    const h0 = Array.from({ length: HD }, () => r.normal() * 1.5);
    const u = [0, 1, 2, 3, 4, 5].map(() => Array.from({ length: HD }, () => r.normal() * 0.7));
    const h1 = h0.map((a, i) => a + u[0][i]);
    const h2 = h1.map((a, i) => a + u[1][i]);
    let hl = h2.slice();
    for (let k = 2; k < 6; k++) hl = hl.map((a, i) => a + u[k][i]);
    return { h0, u, h1, h2, hl };
  }, []);
  const curves = useMemo(() => BR.map(branchCurve), []);
  const T = cssColor("--ax-T"), D = cssColor("--ax-D"), N = cssColor("--ax-hd"), F = cssColor("--fg-faint");
  const K = 0.4, endX = 6.6;
  const p = progress;
  const norm = (a: number[]) => { const r = rmsOf(a); return a.map((x) => x / r); };

  // Her adımda paketlerin yeri ve değerleri, adım içi ilerlemeye (p) göre sürekli
  // akış paketi
  let sx = -6.4, sv = v.h0;
  // dal paketi: eğri üzerindeki konum (ct), değerleri (cv), rengi
  let ct = -1, cv = v.h0, cCol = N, b = 0;
  if (step === 0) sx = -6.6 + 1.2 * smooth(p);
  else if (step === 1) { sx = -5.4; ct = T_NORM * smooth(p * 1.3); cv = mix(v.h0, norm(v.h0), smooth((p - 0.45) * 2.5)); }
  else if (step === 2) { sx = -5.4; ct = T_NORM + (T_F - T_NORM) * smooth(p * 1.3); cv = mix(norm(v.h0), v.u[0], smooth((p - 0.45) * 2.5)); cCol = smooth((p - 0.45) * 2.5) > 0.5 ? D : N; }
  else if (step === 3) {
    const go = smooth(p * 1.6), merge = smooth((p - 0.6) * 2.5);
    sx = -5.4 + (BR[0].add - -5.4) * go; ct = merge >= 1 ? -1 : T_F + (1 - T_F) * go; cv = v.u[0]; cCol = D;
    sv = mix(v.h0, v.h1, merge);
  } else if (step === 4) {
    const a = smooth(p * 3), c = smooth((p - 0.33) * 3), d = smooth((p - 0.66) * 3), merge = smooth((p - 0.82) * 5.5);
    sx = BR[0].add + (BR[1].from - BR[0].add) * a + (BR[1].add - BR[1].from) * d;
    b = 1; sv = mix(v.h1, v.h2, merge);
    ct = merge >= 1 ? -1 : p < 0.33 ? T_NORM * a : p < 0.66 ? T_NORM + (T_F - T_NORM) * c : T_F + (1 - T_F) * d;
    cv = p < 0.33 ? mix(v.h1, norm(v.h1), a) : p < 0.66 ? mix(norm(v.h1), v.u[1], c) : v.u[1];
    cCol = p < 0.5 ? N : D;
  } else { sx = BR[1].add + (endX - BR[1].add) * smooth(p * 1.4); sv = mix(v.h2, v.hl, smooth(p * 1.4)); }
  const onBranch = ct >= 0;
  const cp = onBranch ? curves[b].getPointAt(Math.min(1, ct)) : new THREE.Vector3(sx, 0, 0);

  const items: Cube[] = [];
  items.push({ id: "lane", pos: [0, -0.04, 0], size: [14.6, 0.05, 1.1], color: cssColor("--border-strong") });
  const sRms = rmsOf(sv);
  items.push(...barItems("s", sv, [sx, 0, 0], T, K, { tip: (i) => <>akış h[{i}] = <b>{fx(sv[i])}</b><div className="muted">rms = {fx(sRms, 2)}</div></> }));
  items.push(...barItems("c", cv, [cp.x, cp.y + 0.06, 0.25], cCol, K, { hidden: !onBranch, tip: (i) => <>dal paketi[{i}] = <b>{fx(cv[i])}</b><div className="muted">rms = {fx(rmsOf(cv), 2)}</div></> }));

  const live = (k: number) => (step >= 1 && step <= 3 && k === 0) || (step === 4 && k === 1);
  // Kamera: dal adımlarında o dala, sonda bütün katmana
  const cx = step <= 3 ? -3.4 : step === 4 ? 2.0 : 0.2;
  const view: View = step === 5 ? { pos: [0.6, 4.2, 12.4], target: [0.2, 0.8, 0] } : { pos: [cx + 0.3, 2.9, 6.9], target: [cx, 0.95, 0] };
  const label = (x: number, y: number, txt: ReactNode, color: string, on: boolean, z = 0) => <Tag position={[x, y, z]} color={color} size={on ? 12.5 : 11.5} weight={on ? 700 : 500}>{txt}</Tag>;
  return (
    <Stage3D view={view} height={480} maxDist={30}>
      <Cubes items={items} />
      {BR.map((br, k) => (
        <group key={k}>
          <Road curve={curves[k]} color={k === 0 ? F : D} on={live(k)} />
          <Block x={br.norm} color={N} on={live(k)} />
          <Block x={br.f} color={br.kind === "attn" ? F : D} on={live(k)} />
          <AddNode x={br.add} on={(step === 3 && k === 0) || (step === 4 && k === 1)} />
          {label(br.norm, BY + 0.45, k === 0 ? "input_layernorm" : "post_attention_layernorm", "var(--ax-hd)", (step === 1 && k === 0) || (step === 4 && k === 1))}
          {label(br.f, BY + 0.45, k === 0 ? "self_attn (şimdilik Identity)" : "mlp", k === 0 ? "var(--fg-muted)" : "var(--ax-D)", (step === 2 && k === 0) || (step === 4 && k === 1))}
          {label(br.add + 0.2, -0.62, "⊕  h + f(norm(h))", "var(--accent)", (step === 3 && k === 0) || (step === 4 && k === 1), 1.05)}
        </group>
      ))}
      <Block x={endX} color={N} on={step === 5} w={1.2} />
      {label(endX, BY + 0.45, "model.norm (18 katman sonra)", "var(--ax-hd)", step === 5)}
      {label(-6.2, -0.5, "akış (hidden_states)", "var(--ax-T)", step === 0, 0.75)}
    </Stage3D>
  );
}

/* ------------------------------------------------------------------ 5.4 GeGLU */
const MT = 3, MD = 4, MI = 8;
export function MlpScene({ step }: { step: number }) {
  const v = useMemo(() => {
    const r = rng(5);
    const x = Array.from({ length: MT }, () => Array.from({ length: MD }, () => r.normal()));
    const Wg = Array.from({ length: MI }, () => Array.from({ length: MD }, () => r.normal() * 0.8));
    const Wu = Array.from({ length: MI }, () => Array.from({ length: MD }, () => r.normal() * 0.8));
    const Wd = Array.from({ length: MD }, () => Array.from({ length: MI }, () => r.normal() * 0.5));
    const mv = (W: number[][], a: number[]) => W.map((row) => row.reduce((s, w, j) => s + w * a[j], 0));
    const g = x.map((a) => mv(Wg, a)), u = x.map((a) => mv(Wu, a));
    const act = g.map((row) => row.map(geluTanh));
    const h = act.map((row, t) => row.map((a, i) => a * u[t][i]));
    const o = h.map((row) => mv(Wd, row));
    return { x, g, u, act, h, o };
  }, []);
  const cs = 0.3, gp = 0.34;
  const grid = (cx: number, cy: number, rows: number, cols: number, r: number, c: number): [number, number, number] =>
    [cx + (c - (cols - 1) / 2) * gp, cy + ((rows - 1) / 2 - r) * gp, 0];
  const items: Cube[] = [];
  const P = { x: -4.6, gate: -1.0, up: -1.0, act: 1.9, h: 1.9, o: 4.9 };
  const Y = { top: 1.25, bot: -1.25 };
  const maxG = 3, maxH = 3;
  for (let t = 0; t < MT; t++) for (let d = 0; d < MD; d++) {
    items.push({ id: `x${t}${d}`, pos: grid(P.x, 0, MT, MD, t, d), color: vc(v.x[t][d], 2.5), dim: step >= 2 && step < 4,
      tip: () => <>x[t={t}, d={d}] = <b>{fx(v.x[t][d])}</b><div className="muted">[T, D] = [3, 4] (gerçekte [T, 2048])</div></> });
    items.push({ id: `o${t}${d}`, pos: step >= 4 ? grid(P.o, 0, MT, MD, t, d) : grid(P.h, 0, MT, MI, t, d * 2), color: vc(v.o[t][d], 3), hidden: step < 4,
      tip: () => <>çıkış[t={t}, d={d}] = Σᵢ down[d,i] · h[t,i] = <b>{fx(v.o[t][d])}</b></> });
  }
  for (let t = 0; t < MT; t++) for (let i = 0; i < MI; i++) {
    const g = v.g[t][i], a = v.act[t][i], u = v.u[t][i], h = v.h[t][i];
    const closed = Math.abs(a) < 0.12;
    const gPos = step >= 3 ? grid(P.h, 0, MT, MI, t, i) : step === 2 ? grid(P.act, Y.top, MT, MI, t, i) : step >= 1 ? grid(P.gate, Y.top, MT, MI, t, i) : grid(P.x, 0, MT, MD, t, Math.min(MD - 1, i >> 1));
    items.push({ id: `g${t}${i}`, pos: gPos, color: step >= 2 ? (closed ? cssColor("--border-strong") : vc(a, maxG)) : vc(g, maxG), hidden: step < 1 || step >= 3,
      tip: () => step >= 2 ? <>gelu_tanh(gate) = gelu({fx(g)}) = <b>{fx(a)}</b><div className="muted">{closed ? "≈ 0 → kapı KAPALI" : "kapı açık"}</div></> : <>gate[t={t}, i={i}] = <b>{fx(g)}</b></> });
    items.push({ id: `u${t}${i}`, pos: step >= 3 ? grid(P.h, 0, MT, MI, t, i) : step >= 1 ? grid(P.up, Y.bot, MT, MI, t, i) : grid(P.x, 0, MT, MD, t, Math.min(MD - 1, i >> 1)), color: vc(u, maxG), hidden: step < 1 || step >= 3, dim: step === 2,
      tip: () => <>up[t={t}, i={i}] = <b>{fx(u)}</b><div className="muted">aktivasyon yok: "ne geçsin"</div></> });
    items.push({ id: `h${t}${i}`, pos: grid(P.h, 0, MT, MI, t, i), color: Math.abs(h) < 0.05 ? cssColor("--border-strong") : vc(h, maxH), hidden: step < 3, dim: step >= 4,
      tip: () => <>h = gelu(gate) ⊙ up = {fx(a)} × {fx(u)} = <b>{fx(h)}</b>{closed && <div className="muted">kapı kapalı: up ne olursa olsun ≈ 0</div>}</> });
  }
  // ağırlık levhaları
  const plate = (id: string, x: number, y: number, label: string, on: boolean, col: string) => items.push({ id, pos: [x, y, -0.5], size: [0.08, MT * gp + 0.6, 1.1], color: col, hidden: !on });
  plate("Wg", (P.x + P.gate) / 2 + 0.4, Y.top, "gate", step >= 1, cssColor("--ax-hd"));
  plate("Wu", (P.x + P.up) / 2 + 0.4, Y.bot, "up", step >= 1, cssColor("--ax-D"));
  plate("Wd", (P.h + P.o) / 2 + 0.2, 0, "down", step >= 4, cssColor("--ax-D"));
  const views: View[] = [
    { pos: [-3.9, 1.0, 5.0], target: [-4.5, 0, 0] },
    { pos: [-2.0, 1.2, 8.4], target: [-2.6, 0, 0] },
    { pos: [0.6, 1.4, 8.2], target: [0.2, 0.4, 0] },
    { pos: [1.9, 0.9, 6.2], target: [1.6, 0, 0] },
    { pos: [0.4, 1.4, 12.0], target: [0.2, 0, 0] },
  ];
  return (
    <Stage3D view={views[Math.min(step, 4)]} height={460}>
      <Cubes items={items} cell={cs} />
      <Tag position={[P.x, 0.95, 0]} color="var(--ax-T)">x <span className="mono" style={{ color: "var(--fg-muted)" }}>[T, D]</span></Tag>
      {step >= 1 && <Tag position={[P.gate, Y.top + 0.95, 0]} color="var(--ax-hd)">gate_proj(x) <span className="mono" style={{ color: "var(--fg-muted)" }}>[T, I]</span></Tag>}
      {step >= 1 && <Tag position={[P.up, Y.bot - 0.95, 0]} color="var(--ax-D)">up_proj(x) <span className="mono" style={{ color: "var(--fg-muted)" }}>[T, I]</span></Tag>}
      {step === 2 && <Tag position={[P.act, Y.top + 0.95, 0]} color="var(--accent)">gelu_tanh(gate) · gri = kapalı</Tag>}
      {step >= 3 && <Tag position={[P.h, 0.95, 0]} color="var(--ax-I)">h = gelu(gate) ⊙ up</Tag>}
      {step >= 4 && <Tag position={[P.o, 0.95, 0]} color="var(--ax-T)">down_proj(h) <span className="mono" style={{ color: "var(--fg-muted)" }}>[T, D]</span></Tag>}
    </Stage3D>
  );
}

/* ------------------------------------------------------------------ 5.7 Weight tying */
const TV = 10, TDim = 6, LOOK = 3;
export function TyingScene({ step }: { step: number }) {
  const v = useMemo(() => {
    const r = rng(13);
    const E = Array.from({ length: TV }, () => Array.from({ length: TDim }, () => r.normal()));
    const h = E[6].map((a) => a * 1.3 + r.normal() * 0.4);   // 6. satıra benzeyen bir gizli durum
    const logits = E.map((row) => row.reduce((s, e, j) => s + e * h[j], 0));
    return { E, h, logits, best: logits.indexOf(Math.max(...logits)) };
  }, []);
  const g = 0.36;
  const pos = (r: number, c: number): [number, number, number] => [(c - (TDim - 1) / 2) * g, ((TV - 1) / 2 - r) * g, 0];
  const items: Cube[] = [];
  const accent = cssColor("--accent");
  for (let r = 0; r < TV; r++) for (let c = 0; c < TDim; c++) {
    const e = v.E[r][c];
    const hl = (step === 1 && r === LOOK) || (step >= 2 && r === v.best);
    items.push({ id: `E${r}${c}`, pos: pos(r, c), color: vc(e, 2.2), dim: (step === 1 && r !== LOOK), tip: () => <>W[{r}, {c}] = <b>{fx(e)}</b><div className="muted">aynı sayı: embed_tokens.weight ve lm_head.weight</div></> });
    if (hl) items[items.length - 1].color = vc(e, 2.2);
  }
  // lookup: satırın kopyası sola uçuyor
  for (let c = 0; c < TDim; c++) {
    const p = pos(LOOK, c);
    items.push({ id: `L${c}`, pos: step >= 1 ? [p[0] - 3.6, p[1], 0] : p, color: vc(v.E[LOOK][c], 2.2), hidden: step < 1,
      tip: () => <>embed_tokens(id={LOOK})[{c}] = W[{LOOK}, {c}] = <b>{fx(v.E[LOOK][c])}</b></> });
  }
  // çıkış: h vektörü üstte, logit çubukları sağda
  for (let c = 0; c < TDim; c++) {
    const p = pos(0, c);
    items.push({ id: `H${c}`, pos: [p[0], p[1] + 0.9, 0], color: cssColor("--ax-D"), hidden: step < 2, tip: () => <>norm(h)[{c}] = <b>{fx(v.h[c])}</b></> });
  }
  const maxL = Math.max(...v.logits.map(Math.abs));
  for (let r = 0; r < TV; r++) {
    const L = v.logits[r];
    const w = Math.max(0.04, (Math.abs(L) / maxL) * 1.8);
    const p = pos(r, TDim - 1);
    items.push({ id: `G${r}`, pos: step >= 2 ? [p[0] + 0.5 + w / 2, p[1], 0] : [p[0] + 0.5, p[1], 0], size: [w, g * 0.8, 0.2], color: r === v.best ? accent : L >= 0 ? cssColor("--ax-V") : cssColor("--fg-faint"), hidden: step < 2,
      tip: () => <>logit[{r}] = Σ h·W[{r}] = <b>{fx(L)}</b>{r === v.best && <div className="muted">en büyük: argmax → token {r}</div>}</> });
  }
  const views: View[] = [
    { pos: [0.6, 0.6, 6.0], target: [0, 0, 0] },
    { pos: [-1.8, 0.8, 6.6], target: [-1.6, 0.2, 0] },
    { pos: [1.2, 1.2, 8.2], target: [0.9, 0.6, 0] },
    { pos: [-0.4, 1.2, 9.4], target: [-0.6, 0.4, 0] },
  ];
  const top = ((TV - 1) / 2) * g;
  return (
    <Stage3D view={views[Math.min(step, 3)]} height={460}>
      <Cubes items={items} cell={g * 0.88} />
      <Tag position={[0, -top - 0.45, 0]} color="var(--ax-V)" size={12}>W <span className="mono">[V, D]</span> · embed_tokens.weight <b>is</b> lm_head.weight</Tag>
      <Tag position={[-((TDim - 1) / 2) * g - 0.55, 0, 0]} color="var(--ax-V)">V ↓</Tag>
      {step >= 1 && <Tag position={[-3.6, pos(LOOK, 0)[1] + 0.4, 0]} color="var(--ax-T)">girişte: id = {LOOK} → satır {LOOK}</Tag>}
      {step >= 2 && <Tag position={[0, top + 1.35, 0]} color="var(--ax-D)">çıkışta: norm(h) · Wᵀ → logit</Tag>}
      {step >= 2 && <Tag position={[((TDim - 1) / 2) * g + 1.6, top + 0.4, 0]} color="var(--fg-muted)">logits <span className="mono">[V]</span></Tag>}
    </Stage3D>
  );
}
