// Chapter 2.2: preprocessing, the Conv2d patch explorer, flatten/transpose in 3D, the position embedding, the shuffle experiment
import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import * as d3 from "d3";
import { useData, Arr } from "../../lib/data";
import { rng } from "../../lib/tensor";
import { Heatmap, LineChart, useWidth } from "../../components/charts";
import { Stage3D, Cubes, Cube, Tag, View } from "../../interact/three";
import { Loading, S, Stat } from "../../components/ui";
import { useStore } from "../../store";
import { useT } from "../../i18n";

const BASE = import.meta.env.BASE_URL + "data/";
const P = 14, N = 16;

/* ---------------- preprocessing scrolly ---------------- */
export function PreprocFigure({ step }: { step: number }) {
  const tr = useT();
  const { data, error } = useData<any>("pipeline");
  const { imageIdx } = useStore();
  if (!data) return <Loading what={tr("image", "görüntü")} error={error} />;
  const im = data.images[imageIdx];
  const hist = step <= 1 ? im.hist_u8 : step === 2 ? im.hist_rescaled : im.hist_normalized;
  const [lo, hi] = step <= 1 ? [0, 255] : step === 2 ? [0, 1] : [-1, 1];
  const xs = Array.from({ length: 64 }, (_, i) => lo + ((i + 0.5) / 64) * (hi - lo));
  const ch = [["R", "#d64545"], ["G", "#2f9e59"], ["B", "#3a6fd8"]] as const;
  return (
    <div>
      <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
        {step < 4 ? (
          <div style={{ width: step === 0 ? 240 : 224 }}>
            <img src={step === 0 ? `${BASE}img_${im.name}.jpg` : `${BASE}img_${im.name}_224.png`} style={{ width: "100%", borderRadius: 4, display: "block", aspectRatio: step === 0 ? `${im.orig_size[0]} / ${im.orig_size[1]}` : "1", objectFit: "fill" }} />
            <div className="small muted" style={{ marginTop: 4 }}>{step === 0 ? `${im.orig_size[0]} × ${im.orig_size[1]} · uint8` : "224 × 224 · bicubic"}</div>
          </div>
        ) : (
          <ChannelStack px={im.pixel_values} />
        )}
        <div style={{ flex: 1, minWidth: 240 }}>
          <div style={{ fontSize: 13, marginBottom: 4 }}><b>{tr("Pixel histogram", "Piksel histogramı")}</b> <span className="muted">· {tr("range", "aralık")} [{lo}, {hi}]</span></div>
          <LineChart height={190} xLabel={tr("value", "değer")} yDomain={undefined} fmtX={(v) => (Math.abs(v) < 2 ? v.toFixed(1) : v.toFixed(0))}
            series={ch.map(([n, c], k) => ({ name: n, color: c, values: hist[k], x: xs }))} />
          <div className="small" style={{ marginTop: 6 }}>
            {tr("shape", "şekil")} <S s={step === 4 ? "3, 224, 224" : step === 0 ? `${im.orig_size[1]}, ${im.orig_size[0]}, 3` : "224, 224, 3"} ax={step === 4 ? "C,," : ",,C"} />
            {" "}<span className="muted mono">{step <= 1 ? "uint8" : "float32"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChannelStack({ px }: { px: Arr }) {
  const refs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  useEffect(() => {
    refs.forEach((r, c) => {
      const cv = r.current; if (!cv) return;
      cv.width = 224; cv.height = 224;
      const ctx = cv.getContext("2d")!; const img = ctx.createImageData(224, 224);
      for (let i = 0; i < 224 * 224; i++) {
        const v = Math.round(((px.data[c * 224 * 224 + i] as number) * 0.5 + 0.5) * 255);
        img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    });
  }, [px]);
  return (
    <div style={{ position: "relative", width: 260, height: 260 }}>
      {refs.map((r, c) => (
        <div key={c} style={{ position: "absolute", left: (2 - c) * 18, top: c * 18, width: 200 }}>
          <canvas ref={r} style={{ width: 200, height: 200, borderRadius: 3, boxShadow: "0 2px 10px rgba(0,0,0,.15)", display: "block", outline: `3px solid ${["#d64545", "#2f9e59", "#3a6fd8"][c]}` }} />
          <span className="mono small" style={{ position: "absolute", left: 6, top: 4, color: "#fff", fontWeight: 700, background: ["#d64545", "#2f9e59", "#3a6fd8"][c], padding: "0 5px", borderRadius: 3 }}>{["R", "G", "B"][c]} · kanal {c}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------- Conv2d yama gezgini ---------------- */
function drawPatch(cv: HTMLCanvasElement | null, px: Arr, r: number, c: number, scale = 12) {
  if (!cv) return;
  cv.width = P; cv.height = P;
  const ctx = cv.getContext("2d")!; const img = ctx.createImageData(P, P);
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) for (let k = 0; k < 3; k++) {
    const v = px.data[k * 224 * 224 + (r * P + y) * 224 + (c * P + x)] as number;
    img.data[(y * P + x) * 4 + k] = Math.round((v * 0.5 + 0.5) * 255);
  }
  for (let i = 0; i < P * P; i++) img.data[i * 4 + 3] = 255;
  ctx.putImageData(img, 0, 0);
}
function drawFilter(cv: HTMLCanvasElement | null, f: Arr, i: number) {
  if (!cv) return;
  cv.width = P; cv.height = P;
  const ctx = cv.getContext("2d")!; const img = ctx.createImageData(P, P);
  const off = i * 3 * P * P;
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < 3 * P * P; j++) { const v = f.data[off + j] as number; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) for (let k = 0; k < 3; k++) {
    const v = f.data[off + k * P * P + y * P + x] as number;
    img.data[(y * P + x) * 4 + k] = Math.round(((v - lo) / (hi - lo || 1)) * 255);
  }
  for (let j = 0; j < P * P; j++) img.data[j * 4 + 3] = 255;
  ctx.putImageData(img, 0, 0);
}

export function PatchExplorer() {
  const tr = useT();
  const pipe = useData<any>("pipeline"), sl = useData<any>("pg_slices");
  const { imageIdx } = useStore();
  const [pr, setPr] = useState<[number, number]>([7, 7]);
  const [fi, setFi] = useState(0);
  const [hf, setHf] = useState<number | null>(null);   // the hovered filter: only a preview ring
  const zoom = useRef<HTMLCanvasElement>(null);
  const fz = useRef<HTMLCanvasElement>(null);
  const gal = useRef<(HTMLCanvasElement | null)[]>([]);
  const px: Arr | undefined = pipe.data?.images[imageIdx].pixel_values;
  const F: Arr | undefined = sl.data?.patch_filters;
  const fmap = useMemo(() => {
    if (!px || !F) return null;
    const out = new Float32Array(N * N);
    const off = fi * 3 * P * P;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) for (let y = 0; y < P; y++) for (let x = 0; x < P; x++)
        s += (px.data[k * 224 * 224 + (r * P + y) * 224 + (c * P + x)] as number) * (F.data[off + k * P * P + y * P + x] as number);
      out[r * N + c] = s;
    }
    return out;
  }, [px, F, fi]);
  useEffect(() => { if (px) drawPatch(zoom.current, px, pr[0], pr[1]); }, [px, pr]);
  useEffect(() => { if (F) drawFilter(fz.current, F, fi); }, [F, fi]);
  useEffect(() => { if (F) gal.current.forEach((cv, i) => drawFilter(cv, F, i)); }, [F]);
  const onGalKey = (e: React.KeyboardEvent) => {
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -16, ArrowDown: 16 }[e.key];
    if (d === undefined) return;
    e.preventDefault();
    setFi((f) => Math.min(63, Math.max(0, f + d)));
  };
  if (!px || !F || !fmap) return <Loading what={tr("pixels and filters", "pikseller ve filtreler")} error={pipe.error ?? sl.error} />;
  const im = pipe.data.images[imageIdx];
  const v = fmap[pr[0] * N + pr[1]];
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 300px) minmax(220px, 1fr) minmax(220px, 300px)", gap: 28, alignItems: "start" }}>
        <div>
          <div className="small" style={{ marginBottom: 6 }}><b>{tr("Image", "Görüntü")}</b> <span className="muted">· {tr("hover a patch", "bir yamanın üstüne gel")}</span></div>
          <div style={{ position: "relative", width: "100%", aspectRatio: "1", backgroundImage: `url(${BASE}img_${im.name}_224.png)`, backgroundSize: "cover", borderRadius: 3, cursor: "crosshair", overflow: "hidden" }}
            onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setPr([Math.min(N - 1, Math.floor(((e.clientY - r.top) / r.height) * N)), Math.min(N - 1, Math.floor(((e.clientX - r.left) / r.width) * N))]); }}>
            <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(to right, rgba(255,255,255,.35) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,.35) 1px, transparent 1px)", backgroundSize: `${100 / N}% ${100 / N}%` }} />
            <div style={{ position: "absolute", left: `${(pr[1] / N) * 100}%`, top: `${(pr[0] / N) * 100}%`, width: `${100 / N}%`, height: `${100 / N}%`, outline: "2px solid var(--accent)", boxShadow: "0 0 0 999px rgba(0,0,0,.18)" }} />
          </div>
          <div className="small muted" style={{ marginTop: 4 }}>yama ({pr[0]}, {pr[1]}) → token {pr[0] * N + pr[1]}</div>
        </div>
        <div>
          <div className="small" style={{ marginBottom: 6 }}><b>{tr("One patch × one filter = one number", "Bir yama × bir filtre = bir sayı")}</b></div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <div><canvas ref={zoom} style={{ width: 112, height: 112, imageRendering: "pixelated", borderRadius: 3, display: "block" }} /><div className="small muted">14 × 14 × 3 piksel</div></div>
            <span style={{ fontSize: 22, color: "var(--fg-faint)" }}>·</span>
            <div><canvas ref={fz} style={{ width: 112, height: 112, imageRendering: "pixelated", borderRadius: 3, display: "block" }} /><div className="small muted">filtre #{sl.data.patch_filters_idx[fi]}</div></div>
            <span style={{ fontSize: 22, color: "var(--fg-faint)" }}>=</span>
            <div style={{ fontFamily: "var(--display)", fontSize: 30, fontWeight: 650, color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>{v.toFixed(2)}</div>
          </div>
          <p className="small muted" style={{ margin: "10px 0 0" }}>{tr("588 pixels × 588 weights, summed (+ bias). The real model has 1152 filters: each patch becomes 1152 numbers, i.e. one token.", "588 piksel × 588 ağırlık, toplam (+ bias). Gerçek modelde 1152 filtre var: her yama 1152 sayıya, yani bir token'a dönüşüyor.")}</p>
          <div className="small" style={{ margin: "12px 0 6px", display: "flex", justifyContent: "space-between", gap: 8 }}>
            <span><b>{tr("Click a filter in the gallery", "Galeriden bir filtreye tıkla")}</b></span>
            <span className="muted mono">{hf !== null && hf !== fi ? tr(`#${sl.data.patch_filters_idx[hf]} · click`, `#${sl.data.patch_filters_idx[hf]} · tıkla`) : tr(`selected #${sl.data.patch_filters_idx[fi]}`, `seçili #${sl.data.patch_filters_idx[fi]}`)}</span>
          </div>
          <div role="listbox" aria-label="filtre galerisi" tabIndex={0} onKeyDown={onGalKey} onPointerLeave={() => setHf(null)}
            style={{ display: "grid", gridTemplateColumns: "repeat(16, 1fr)", gap: 3, outline: "none" }}>
            {Array.from({ length: 64 }, (_, i) => (
              <canvas key={i} ref={(el) => (gal.current[i] = el)} role="option" aria-selected={i === fi} onClick={() => setFi(i)} onPointerEnter={() => setHf(i)}
                style={{ width: "100%", aspectRatio: "1", imageRendering: "pixelated", cursor: "pointer", borderRadius: 2, transition: "transform .12s, opacity .12s",
                  outline: i === fi ? "2.5px solid var(--accent)" : i === hf ? "1.5px dashed var(--fg-muted)" : "none", outlineOffset: 1,
                  transform: i === hf && i !== fi ? "scale(1.12)" : "none", opacity: i === fi || i === hf ? 1 : 0.82 }} />
            ))}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>{tr("real PaliGemma filters (the 64 with the largest norm), scaled to RGB · arrow keys work too", "gerçek PaliGemma filtreleri (en büyük normlu 64'ü), RGB'ye ölçeklenmiş · ok tuşlarıyla da gezebilirsin")}</div>
        </div>
        <div>
          <div className="small" style={{ marginBottom: 6 }}><b>{tr("This filter's response on every patch", "Bu filtrenin bütün yamalardaki cevabı")}</b> <span className="muted">· [16, 16]</span></div>
          <Heatmap values={fmap} rows={N} cols={N} diverging height={260} highlight={{ r: pr[0], c: pr[1] }} legend
            onCell={(r, c) => setPr([r, c])} cellLabel={(r, c, val) => `yama (${r}, ${c}) · ${val.toFixed(2)}`} />
          <div className="small muted">{tr("A single channel of the Conv2d output", "Conv2d çıktısının tek bir kanalı")}: <span className="mono">patch_embeds[0, {sl.data.patch_filters_idx[fi]}]</span></div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- flatten + transpose (3D) ---------------- */
const C3 = 4, H3 = 3, W3 = 3;
export function FlattenScene({ step }: { step: number }) {
  const tr = useT();
  const cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"];
  const g = 0.46;
  const items: Cube[] = [];
  for (let c = 0; c < C3; c++) for (let y = 0; y < H3; y++) for (let x = 0; x < W3; x++) {
    const p = y * W3 + x;
    let pos: [number, number, number];
    if (step === 0) pos = [(x - 1) * g, (1 - y) * g, (c - 1.5) * g * 1.6];
    else if (step === 1) pos = [(p - 4) * g, (1.5 - c) * g, 0];
    else pos = [(c - 1.5) * g, (4 - p) * g, 0];
    items.push({ id: `${c}${p}`, pos, color: cols[c], tip: () => <>{tr("channel", "kanal")} c={c}, {tr("patch", "yama")} p={p} (y={y}, x={x})<div className="muted">{step === 0 ? "[C, H, W]" : step === 1 ? tr(`[C, H·W] → row ${c}, column ${p}`, `[C, H·W] → satır ${c}, sütun ${p}`) : tr(`[H·W, C] → row ${p} (token ${p}), column ${c}`, `[H·W, C] → satır ${p} (token ${p}), sütun ${c}`)}</div></> });
  }
  const views: View[] = [{ pos: [3.2, 2.4, 4.6], target: [0, 0, 0] }, { pos: [0.4, 0.6, 6.2], target: [0, 0, 0] }, { pos: [0.6, 0.4, 6.8], target: [0, 0, 0] }];
  return (
    <Stage3D view={views[Math.min(step, 2)]} height={420}>
      <Cubes items={items} cell={0.4} />
      {step === 0 && <><Tag position={[0, 1.0, -1.6]} color="var(--fg-muted)">{tr("C = 4 channels (1152 in reality) ↗", "C = 4 kanal (gerçekte 1152) ↗")}</Tag><Tag position={[0, -1.0, 1.6]} color="var(--fg-muted)">{tr("H × W = 3 × 3 (16 × 16 in reality)", "H × W = 3 × 3 (gerçekte 16 × 16)")}</Tag></>}
      {step === 1 && <Tag position={[0, 1.3, 0]} color="var(--fg-muted)">[C, H·W] = [4, 9] · {tr("row = channel", "satır = kanal")}</Tag>}
      {step === 2 && <Tag position={[0, 2.4, 0]} color="var(--accent)">[H·W, C] = [9, 4] · {tr("row = token", "satır = token")}</Tag>}
    </Stage3D>
  );
}

/* ---------------- konum embedding'i ---------------- */
export function PosEmbedExplorer() {
  const tr = useT();
  const { data, error } = useData<any>("pg_slices");
  const [p, setP] = useState(N * 8 + 8);
  const over = useRef<HTMLCanvasElement>(null);
  const cos: Arr | undefined = data?.pos_cos;
  useEffect(() => {
    if (!cos || !over.current) return;
    const s = 6, gap = 2, size = N * (N * s + gap);
    const cv = over.current; cv.width = size; cv.height = size;
    const ctx = cv.getContext("2d")!;
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#fff";
    ctx.fillRect(0, 0, size, size);
    for (let q = 0; q < N * N; q++) {
      const qr = Math.floor(q / N), qc = q % N;
      for (let k = 0; k < N * N; k++) {
        const v = cos.data[q * N * N + k] as number;
        ctx.fillStyle = d3.interpolateInferno(Math.max(0, Math.min(1, (v + 0.2) / 1.2)));
        ctx.fillRect(qc * (N * s + gap) + (k % N) * s, qr * (N * s + gap) + Math.floor(k / N) * s, s, s);
      }
    }
  }, [cos]);
  if (!cos) return <Loading what={tr("position embeddings", "konum embedding'leri")} error={error} />;
  const row = cos.data.subarray(p * N * N, (p + 1) * N * N) as Float32Array;
  return (
    <div className="grid2" style={{ alignItems: "start" }}>
      <div>
        <div className="small" style={{ marginBottom: 6 }}><b>{tr("All 256 positions", "256 konumun hepsi")}</b> <span className="muted">· {tr("each small square is one position's similarity to the others, hover", "her küçük kare bir konumun ötekilerle benzerliği, üstüne gel")}</span></div>
        <div style={{ position: "relative", maxWidth: 420 }}>
        <canvas ref={over} style={{ width: "100%", imageRendering: "pixelated", cursor: "crosshair", display: "block" }}
          onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const qc = Math.floor(((e.clientX - r.left) / r.width) * N), qr = Math.floor(((e.clientY - r.top) / r.height) * N); setP(Math.max(0, Math.min(N * N - 1, qr * N + qc))); }} />
        <div style={{ position: "absolute", left: `${((p % N) / N) * 100}%`, top: `${(Math.floor(p / N) / N) * 100}%`, width: `${100 / N}%`, height: `${100 / N}%`, outline: "2px solid var(--accent)", pointerEvents: "none" }} />
        </div>
      </div>
      <div>
        <div className="small" style={{ marginBottom: 6 }}><b>{tr("Position", "Konum")} ({Math.floor(p / N)}, {p % N})</b> <span className="muted">· {tr("cosine similarity with the other 255 positions", "öteki 255 konumla kosinüs benzerliği")}</span></div>
        <Heatmap values={row} rows={N} cols={N} height={300} domain={[-0.2, 1]} highlight={{ r: Math.floor(p / N), c: p % N }} onCell={(r, c) => setP(r * N + c)} cellLabel={(r, c, v) => `(${r}, ${c}) · cos = ${v.toFixed(3)}`} />
        <p className="small muted" style={{ marginTop: 6 }}>{tr("Nobody told the model \"this patch is top left\". The vectors the real PaliGemma learned carry a 2D grid layout on their own: neighboring positions are similar, and so are positions in the same row and column.", "Kimse modele \"bu yama sol üstte\" demedi. Gerçek PaliGemma'nın öğrendiği vektörler kendiliğinden 2D bir ızgara düzeni taşıyor: komşu konumlar birbirine benziyor, aynı satır ve sütun da öyle.")}</p>
      </div>
    </div>
  );
}

/* ---------------- shuffle experiment ---------------- */
export function ShuffleDemo() {
  const tr = useT();
  const { data } = useData<any>("pipeline");
  const { imageIdx } = useStore();
  const [order, setOrder] = useState<number[]>(() => Array.from({ length: 64 }, (_, i) => i));
  if (!data) return null;
  const im = data.images[imageIdx];
  const n = 8;
  const shuffle = () => { const r = rng(Date.now() % 9973); const a = order.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } setOrder(a); };
  const sorted = order.every((v, i) => v === i);
  return (
    <div className="grid2" style={{ alignItems: "center" }}>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 2, maxWidth: 320 }}>
        {order.map((t) => (
          <motion.div key={t} layout transition={{ type: "spring", stiffness: 120, damping: 18 }}
            style={{ aspectRatio: "1", backgroundImage: `url(${BASE}img_${im.name}_224.png)`, backgroundSize: `${n * 100}% ${n * 100}%`, backgroundPosition: `${((t % n) / (n - 1)) * 100}% ${(Math.floor(t / n) / (n - 1)) * 100}%`, borderRadius: 2 }} />
        ))}
      </div>
      <div>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn primary" onClick={shuffle}>{tr("Shuffle the patches", "Yamaları karıştır")}</button>
          <button className="btn" disabled={sorted} onClick={() => setOrder(order.slice().sort((a, b) => a - b))}>{tr("Sort", "Sırala")}</button>
        </div>
        <p style={{ fontSize: 15, marginTop: 12 }}>
          {tr(<>Without the position embedding the encoder would see these two images as the <b>same</b>: attention and the MLP ignore token order (they are permutation-equivariant). The output of the shuffled sequence would just be the original output shuffled the same way. What makes a sign a sign is <i>where</i> the patches are, and that information only comes from the position embedding.</>, <>Konum embedding'i olmasa encoder bu iki görüntüyü <b>aynı</b> görürdü: attention ve MLP token sırasına duyarsız (permütasyona eşdeğer). Karıştırılmış dizinin çıktısı, orijinal çıktının aynı şekilde karıştırılmış hali olurdu. "Tabela"yı "tabela" yapan şey yamaların <i>nerede</i> olduğu, ve bu bilgi sadece konum embedding'iyle geliyor.</>)}
        </p>
      </div>
    </div>
  );
}
