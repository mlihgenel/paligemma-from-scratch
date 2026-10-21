// Bölüm 2.3: attention'ın tensör koreografisi (3D), gerçek attention haritaları, PCA, parametreler
import { useEffect, useMemo, useRef, useState } from "react";
import { useData, Arr } from "../../lib/data";
import { rng, softmax } from "../../lib/tensor";
import { Bars } from "../../components/charts";
import { Stage3D, Cubes, Cube, Tag, View, cssColor } from "../../interact/three";
import { valueColor } from "../../interact/cells";
import { Loading, Seg, Stat } from "../../components/ui";
import { useStore } from "../../store";

const BASE = import.meta.env.BASE_URL + "data/";
const TT = 4, DD = 6, HH = 2, HD = 3;
const vc = (v: number, m: number) => valueColor(v, m, document.documentElement.dataset.theme === "dark");
const f2 = (v: number) => v.toFixed(2);

export function AttentionScene({ step }: { step: number }) {
  const v = useMemo(() => {
    const r = rng(77);
    const mat = () => Array.from({ length: DD }, () => Array.from({ length: DD }, () => r.normal() * 0.6));
    const x = Array.from({ length: TT }, () => Array.from({ length: DD }, () => r.normal()));
    const [Wq, Wk, Wv, Wo] = [mat(), mat(), mat(), mat()];
    const mv = (W: number[][], a: number[]) => W.map((row) => row.reduce((s, w, j) => s + w * a[j], 0));
    const q = x.map((a) => mv(Wq, a)), k = x.map((a) => mv(Wk, a)), vv = x.map((a) => mv(Wv, a));
    const sc = HD ** -0.5;
    const scores = [0, 1].map((h) => q.map((qi) => k.map((kj) => qi.slice(h * HD, h * HD + HD).reduce((s, a, d) => s + a * kj[h * HD + d], 0) * sc)));
    const att = scores.map((S) => S.map((row) => softmax(row)));
    const ho = [0, 1].map((h) => att[h].map((row) => Array.from({ length: HD }, (_, d) => row.reduce((s, a, j) => s + a * vv[j][h * HD + d], 0))));
    const cat = Array.from({ length: TT }, (_, t) => [...ho[0][t], ...ho[1][t]]);
    const out = cat.map((a) => mv(Wo, a));
    return { x, q, k, vv, scores, att, ho, out };
  }, []);
  const g = 0.32;
  const items: Cube[] = [];
  const at = (cx: number, cy: number, rows: number, cols: number, r: number, c: number): [number, number, number] => [cx + (c - (cols - 1) / 2) * g, cy + ((rows - 1) / 2 - r) * g, 0];
  const headX = (cx: number, d: number) => cx + (Math.floor(d / HD) - 0.5) * (HD * g + 0.55) + ((d % HD) - 1) * g;
  for (let t = 0; t < TT; t++) for (let d = 0; d < DD; d++)
    items.push({ id: `x${t}${d}`, pos: at(-5.2, 0, TT, DD, t, d), color: vc(v.x[t][d], 2.2), dim: step >= 2, tip: () => <>x[t={t}, d={d}] = <b>{f2(v.x[t][d])}</b></> });
  (["q", "k", "v"] as const).forEach((name, n) => {
    const M = name === "q" ? v.q : name === "k" ? v.k : v.vv;
    const cy = 1.75 - n * 1.75;
    for (let t = 0; t < TT; t++) for (let d = 0; d < DD; d++) {
      const split = step >= 2;
      const pos: [number, number, number] = step === 0 ? at(-5.2, 0, TT, DD, t, d) : split ? [headX(-2.0, d), at(0, cy, TT, DD, t, d)[1], 0] : at(-2.0, cy, TT, DD, t, d);
      items.push({ id: `${name}${t}${d}`, pos, color: vc(M[t][d], 2.5), hidden: step === 0, dim: (step >= 4 && name !== "v") || (step >= 6),
        tip: () => <>{name}[t={t}, d={d}] = <b>{f2(M[t][d])}</b>{split && <div className="muted">head {Math.floor(d / HD)}, hd={d % HD}</div>}</> });
    }
  });
  for (let h = 0; h < HH; h++) for (let i = 0; i < TT; i++) for (let j = 0; j < TT; j++) {
    const s = v.scores[h][i][j], a = v.att[h][i][j];
    const soft = step >= 4;
    items.push({ id: `s${h}${i}${j}`, pos: at(1.2, 0.85 - h * 1.9, TT, TT, i, j), color: soft ? vc(a, 0.8) : vc(s, 2.5), hidden: step < 3, dim: step >= 6,
      tip: () => soft ? <>softmax satır {i}: a[{i},{j}] = <b>{a.toFixed(3)}</b><div className="muted">satır toplamı = 1 · head {h}</div></> : <>skor[h={h}, {i}, {j}] = q{i}·k{j} / √{HD} = <b>{f2(s)}</b></> });
  }
  for (let h = 0; h < HH; h++) for (let t = 0; t < TT; t++) for (let d = 0; d < HD; d++) {
    const val = v.ho[h][t][d];
    const pos: [number, number, number] = step >= 6 ? at(4.0, 0, TT, DD, t, h * HD + d) : at(3.5, 0.85 - h * 1.9, TT, HD, t, d);
    items.push({ id: `o${h}${t}${d}`, pos, color: vc(val, 2), hidden: step < 5, dim: step >= 7,
      tip: () => <>(a @ v)[h={h}, t={t}, {d}] = Σⱼ a[t,j]·v[j] = <b>{f2(val)}</b>{step >= 6 && <div className="muted">birleşmiş: d = {h * HD + d}</div>}</> });
  }
  for (let t = 0; t < TT; t++) for (let d = 0; d < DD; d++)
    items.push({ id: `f${t}${d}`, pos: at(6.6, 0, TT, DD, t, d), color: vc(v.out[t][d], 2.5), hidden: step < 7, tip: () => <>out_proj çıkışı [t={t}, d={d}] = <b>{f2(v.out[t][d])}</b></> });
  const views: View[] = [
    { pos: [-4.6, 0.6, 6.2], target: [-5.0, 0, 0] },
    { pos: [-3.0, 0.8, 8.0], target: [-3.4, 0, 0] },
    { pos: [-2.4, 0.8, 8.4], target: [-2.6, 0, 0] },
    { pos: [0.0, 0.6, 8.0], target: [-0.2, 0, 0] },
    { pos: [0.6, 0.6, 7.6], target: [0.4, 0, 0] },
    { pos: [2.0, 0.6, 8.6], target: [1.8, 0, 0] },
    { pos: [3.6, 0.6, 7.4], target: [3.8, 0, 0] },
    { pos: [1.4, 1.2, 14.0], target: [0.8, 0, 0] },
  ];
  return (
    <Stage3D view={views[Math.min(step, 7)]} height={460} maxDist={30}>
      <Cubes items={items} cell={0.28} />
      <Tag position={[-5.2, 1.0, 0]} color="var(--ax-T)">x <span className="mono" style={{ color: "var(--fg-muted)" }}>[T, D]</span></Tag>
      {step >= 1 && ["q", "k", "v"].map((n, i) => <Tag key={n} position={[-2.0, 1.75 - i * 1.75 + 0.85, 0]} color="var(--fg)">{n}{step >= 2 ? " [H, T, hd]" : " [T, D]"}</Tag>)}
      {step >= 3 && [0, 1].map((h) => <Tag key={h} position={[1.2, 0.85 - h * 1.9 + 0.85, 0]} color="var(--ax-H)">head {h}: {step >= 4 ? "softmax" : "q·kᵀ/√hd"} [T, T]</Tag>)}
      {step >= 5 && step < 6 && [0, 1].map((h) => <Tag key={h} position={[3.5, 0.85 - h * 1.9 + 0.85, 0]} color="var(--ax-I)">a @ v [T, hd]</Tag>)}
      {step >= 6 && <Tag position={[4.0, 0.9, 0]} color="var(--ax-I)">transpose + reshape [T, D]</Tag>}
      {step >= 7 && <Tag position={[6.6, 0.9, 0]} color="var(--ax-T)">out_proj [T, D]</Tag>}
    </Stage3D>
  );
}

/* ---------------- gerçek attention haritaları ---------------- */
export function AttnMapExplorer() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx } = useStore();
  const [layer, setLayer] = useState(5);
  const [head, setHead] = useState<number | "ort">("ort");
  const [q, setQ] = useState(7 * 14 + 7);
  const cv = useRef<HTMLCanvasElement>(null);
  const im = data?.images[imageIdx];
  const n = 14;
  const row = useMemo(() => {
    if (!im) return null;
    const li = data.attn_layers.indexOf(layer);
    if (head !== "ort" && li >= 0) { const A: Arr = im.attn_heads; return A.data.subarray(((li * 12 + head) * 196 + q) * 196, ((li * 12 + head) * 196 + q + 1) * 196) as Float32Array; }
    const A: Arr = im.attn_mean; return A.data.subarray((layer * 196 + q) * 196, (layer * 196 + q + 1) * 196) as Float32Array;
  }, [im, layer, head, q]);
  useEffect(() => {
    if (!row || !cv.current) return;
    const c = cv.current; c.width = n; c.height = n;
    const ctx = c.getContext("2d")!; const img = ctx.createImageData(n, n);
    let mx = 0; for (const v of row) mx = Math.max(mx, v);
    for (let i = 0; i < n * n; i++) { const t = Math.sqrt(row[i] / (mx || 1)); img.data[i * 4] = 255; img.data[i * 4 + 1] = 120; img.data[i * 4 + 2] = 40; img.data[i * 4 + 3] = Math.round(t * 230); }
    ctx.putImageData(img, 0, 0);
  }, [row]);
  if (!data || !row) return <Loading what="attention haritaları" error={error} />;
  const top = Array.from(row).map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 5);
  const headsOk = data.attn_layers.includes(layer);
  return (
    <div className="grid2" style={{ alignItems: "start" }}>
      <div>
        <div style={{ position: "relative", width: "100%", maxWidth: 420, aspectRatio: "1", backgroundImage: `url(${BASE}img_${im.name}_224.png)`, backgroundSize: "cover", borderRadius: 3, cursor: "crosshair" }}
          onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const y = Math.floor(((e.clientY - r.top) / r.height) * n), x = Math.floor(((e.clientX - r.left) / r.width) * n); setQ(Math.max(0, Math.min(195, y * n + x))); }}>
          <canvas ref={cv} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", imageRendering: "pixelated", mixBlendMode: "multiply" }} />
          <div style={{ position: "absolute", left: `${((q % n) / n) * 100}%`, top: `${(Math.floor(q / n) / n) * 100}%`, width: `${100 / n}%`, height: `${100 / n}%`, outline: "2px solid var(--accent)", outlineOffset: -1 }} />
        </div>
        <div className="small muted" style={{ marginTop: 4 }}>sorgu yaması ({Math.floor(q / n)}, {q % n}) · turuncu: bu yamanın baktığı yerler</div>
      </div>
      <div>
        <label className="small" style={{ display: "flex", gap: 10, alignItems: "center" }}>katman <input type="range" min={0} max={11} value={layer} onChange={(e) => setLayer(+e.target.value)} style={{ flex: 1 }} /> <b className="mono">{layer}</b></label>
        <div style={{ marginTop: 10 }}>
          {headsOk ? (
            <select value={String(head)} onChange={(e) => setHead(e.target.value === "ort" ? "ort" : +e.target.value)}>
              <option value="ort">12 head'in ortalaması</option>
              {Array.from({ length: 12 }, (_, h) => <option key={h} value={h}>head {h}</option>)}
            </select>
          ) : <span className="small muted">Bu katmanda sadece head ortalaması var (head'ler: katman 0, 5, 11)</span>}
        </div>
        <div className="stats">
          <Stat label="kendine bakış" value={`${(row[q] * 100).toFixed(1)}%`} />
          <Stat label="en yüksek 5 yama toplamı" value={`${(top.reduce((a, b) => a + b[0], 0) * 100).toFixed(0)}%`} sub="196 yama içinde" />
        </div>
        <p className="small muted" style={{ marginTop: 10 }}>siglip-base (16 px yama, 14 × 14) ağırlıklarıyla, senin <code>SiglipAttention</code>'ının q/k projeksiyonlarından hesaplandı. Genellikle erken katmanlar komşulara, derin katmanlar anlamca ilgili uzak yamalara bakıyor. Head'leri karşılaştır: her biri farklı bir "soru" soruyor.</p>
      </div>
    </div>
  );
}

/* ---------------- PCA: ilk katman ve son katman ---------------- */
function PcaMap({ rgb }: { rgb: number[][] }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current; if (!c) return;
    c.width = 14; c.height = 14;
    const ctx = c.getContext("2d")!; const img = ctx.createImageData(14, 14);
    rgb.forEach((p, i) => { img.data[i * 4] = p[0] * 255; img.data[i * 4 + 1] = p[1] * 255; img.data[i * 4 + 2] = p[2] * 255; img.data[i * 4 + 3] = 255; });
    ctx.putImageData(img, 0, 0);
  }, [rgb]);
  return <canvas ref={cv} style={{ width: "100%", aspectRatio: "1", imageRendering: "pixelated", borderRadius: 3, display: "block" }} />;
}
export function PcaCompare() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx } = useStore();
  if (!data) return <Loading what="PCA" error={error} />;
  const im = data.images[imageIdx];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 18, maxWidth: 760 }}>
      <div><img src={`${BASE}img_${im.name}_224.png`} style={{ width: "100%", borderRadius: 3, display: "block" }} /><div className="small muted" style={{ marginTop: 4 }}>girdi</div></div>
      <div><PcaMap rgb={im.pca_embed} /><div className="small muted" style={{ marginTop: 4 }}>embedding sonrası (katman 0)</div></div>
      <div><PcaMap rgb={im.pca_last} /><div className="small muted" style={{ marginTop: 4 }}>son katman + post_layernorm</div></div>
    </div>
  );
}

/* ---------------- parametre dağılımı ---------------- */
export function SiglipParams() {
  const { data, error } = useData<any>("model_tree");
  if (!data) return <Loading what="parametreler" error={error} />;
  const g = (n: string) => data.nodes.find((x: any) => x.name === n)?.params ?? 0;
  const L = "vision_tower.vision_model.encoder.layers.0";
  const attn = g(L + ".self_attn"), mlp = g(L + ".mlp"), ln = g(L + ".layer_norm1") + g(L + ".layer_norm2");
  const emb = g("vision_tower.vision_model.embeddings");
  return (
    <div>
      <Bars items={[
        { label: "27 × attention (q, k, v, out)", value: attn * 27, color: "var(--ax-T)" },
        { label: "27 × MLP (fc1, fc2)", value: mlp * 27, color: "var(--ax-D)" },
        { label: "embeddings (Conv2d + konum)", value: emb, color: "var(--ax-V)" },
        { label: "54 LayerNorm + post_layernorm", value: ln * 27 + g("vision_tower.vision_model.post_layernorm"), color: "var(--ax-hd)" },
      ]} fmt={(v) => (v / 1e6).toFixed(1) + " M"} />
      <div className="small muted" style={{ marginTop: 8 }}>Katman başına: attention 4 · 1152² + 4 · 1152 = {(attn / 1e6).toFixed(2)} M, MLP 2 · 1152 · 4304 + 4304 + 1152 = {(mlp / 1e6).toFixed(2)} M. Toplam {(g("vision_tower") / 1e6).toFixed(1)} M ("So400m").</div>
    </div>
  );
}
