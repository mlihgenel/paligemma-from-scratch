// R3: Bölüm 5'in 3D scrolly sahneleri. Hepsi interact/three (Stage3D + instanced Cubes) üzerinde.
// Kural: her adımda tek bir şey hareket eder; geri kalanı yerinde ve soluk durur.
import { ReactNode, useMemo } from "react";
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

/* ------------------------------------------------------------------ 5.1 Otoyol */
const HD = 10;
export function HighwayScene({ step }: { step: number }) {
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
  const T = cssColor("--ax-T"), D = cssColor("--ax-D"), N = cssColor("--ax-hd"), F = cssColor("--fg-faint");
  const K = 0.3;
  const xs = [-3.2, 1.6];                        // attn ve mlp dallarının merkezi
  const addX = [-0.8, 4.0];
  const endX = 6.6;
  const PW = 1.5;                                // levha genişliği (vektör sığsın)
  const stream = step === 0 ? v.h0 : step <= 2 ? v.h0 : step === 3 ? v.h1 : step === 4 ? v.h2 : v.hl;
  const streamX = step === 0 ? -6.2 : step <= 2 ? -5.6 : step === 3 ? addX[0] : step === 4 ? addX[1] : endX;
  const items: Cube[] = [];
  // sabit yapılar: şerit, norm ve f levhaları, ⊕ noktaları
  items.push({ id: "lane", pos: [0.2, -0.04, 0], size: [15.4, 0.05, 1.2], color: cssColor("--border-strong") });
  ["attn", "mlp"].forEach((kind, k) => {
    items.push({ id: `nb${k}`, pos: [xs[k] - 0.9, 2.0, 0], size: [PW, 0.05, 0.9], color: N, dim: !(step === 1 && k === 0) && !(step === 4 && k === 1) });
    items.push({ id: `fb${k}`, pos: [xs[k] + 0.9, 2.0, 0], size: [PW, 0.05, 0.9], color: kind === "attn" ? F : D, dim: !(step === 2 && k === 0) && !(step === 4 && k === 1) });
    items.push({ id: `add${k}`, pos: [addX[k], 0.02, 0], size: [1.6, 0.07, 1.3], color: cssColor("--accent"), dim: !(step === 3 && k === 0) && !(step === 4 && k === 1) });
  });
  items.push({ id: "fn", pos: [endX, 2.0, 0], size: [PW, 0.05, 0.9], color: N, dim: step !== 5 });
  // akış vektörü
  const sRms = rmsOf(stream);
  items.push(...barItems("s", stream, [streamX, 0, 0], T, K, { tip: (i) => <>akış h[{i}] = <b>{fx(stream[i])}</b><div className="muted">‖h‖ rms = {fx(sRms, 2)}</div></> }));
  // dalın norm'lu kopyası
  const src = step === 4 ? v.h1 : step === 5 ? v.hl : v.h0;
  const normed = src.map((a) => a / rmsOf(src));
  const copyX = step === 4 ? xs[1] - 0.9 : step === 5 ? endX : xs[0] - 0.9;
  const showCopy = step === 1 || step === 4 || step === 5;
  items.push(...barItems("c", normed, showCopy ? [copyX, 2.06, 0] : [streamX, 0, 0], N, K, { hidden: !showCopy, tip: (i) => <>norm'lu kopya[{i}] = <b>{fx(normed[i])}</b><div className="muted">rms = 1.00, akışın ölçeğinden bağımsız</div></> }));
  // güncelleme (dalın çıktısı)
  const uk = step === 4 ? v.u[1] : v.u[0];
  const upd = step === 2 ? [xs[0] + 0.9, 2.06, 0] : step === 3 ? [addX[0], 1.25, 0] : step === 4 ? [addX[1], 1.25, 0] : [xs[0] + 0.9, 2.06, 0];
  items.push(...barItems("u", uk, upd as any, D, K, { hidden: !(step >= 2 && step <= 4), tip: (i) => <>güncelleme f(norm(h))[{i}] = <b>{fx(uk[i])}</b><div className="muted">akışa eklenecek</div></> }));

  const cx = [-5.0, -3.6, -2.4, -1.6, 2.2, 1.0][Math.min(step, 5)];
  const view: View = step === 5 ? { pos: [3.0, 3.0, 10.5], target: [2.4, 0.9, 0] } : { pos: [cx + 0.6, 2.6, 7.4], target: [cx, 1.0, 0] };
  const tag = (x: number, y: number, txt: ReactNode, color: string, on: boolean) => <Tag position={[x, y, 0]} color={color} size={11.5} weight={on ? 650 : 500}>{txt}</Tag>;
  return (
    <Stage3D view={view} height={460}>
      <Cubes items={items} />
      <Tag position={[-6.6, -0.42, 0.8]} color="var(--ax-T)" size={12}>akış (hidden_states)</Tag>
      {tag(xs[0] - 0.9, 2.4, "input_layernorm", "var(--ax-hd)", step === 1)}
      {tag(xs[0] + 0.9, 2.4, "self_attn (Identity)", "var(--fg-muted)", step === 2)}
      {tag(xs[1] - 0.9, 2.4, "post_attention_layernorm", "var(--ax-hd)", step === 4)}
      {tag(xs[1] + 0.9, 2.4, "mlp", "var(--ax-D)", step === 4)}
      {tag(addX[0], -0.42, "⊕ residual + …", "var(--accent)", step === 3)}
      {tag(addX[1], -0.42, "⊕ residual + …", "var(--accent)", step === 4)}
      {tag(endX, 2.4, "model.norm", "var(--ax-hd)", step === 5)}
      {step === 5 && <Tag position={[endX - 1.3, -0.42, 0.8]} color="var(--fg-muted)" size={11.5}>… 18 katman sonra</Tag>}
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
