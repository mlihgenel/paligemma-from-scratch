// Bölüm 4: dizi şeridi, <loc> kutu aracı, projektör istatistikleri
import { useState } from "react";
import { useData } from "../../lib/data";
import { Histogram } from "../../components/charts";
import { Inspect } from "../../interact/Inspect";
import { Loading, Stat } from "../../components/ui";
import { useStore } from "../../store";
import { Pickers } from "../oyun/Tour";

const BASE = import.meta.env.BASE_URL + "data/";
const SHOW = (p: string) => p.replace("▁", "␣").replace("\n", "\\n");

export function SequenceStrip() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx, promptIdx } = useStore();
  const [hov, setHov] = useState<{ i: number; x: number; y: number } | null>(null);
  if (!data) return <Loading what="dizi" error={error} />;
  const p = data.prompts[promptIdx];
  const im = data.images[imageIdx];
  const nI = 256;
  return (
    <div>
      <Pickers data={data} />
      <div style={{ position: "relative" }} onPointerLeave={() => setHov(null)}>
        <div style={{ display: "flex", alignItems: "stretch", gap: 2, overflowX: "auto", paddingBottom: 6 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(64, 5px)", gridAutoRows: "11px", gap: 1, flex: "none", padding: 4, background: "color-mix(in srgb, var(--ax-T) 10%, transparent)", borderRadius: 4 }}>
            {Array.from({ length: nI }, (_, i) => (
              <div key={i} onPointerEnter={(e) => { const r = (e.currentTarget.closest("[data-strip]") as HTMLElement).getBoundingClientRect(); setHov({ i, x: e.clientX - r.left, y: e.clientY - r.top }); }}
                style={{ background: hov?.i === i ? "var(--accent)" : "var(--ax-T)", opacity: hov?.i === i ? 1 : 0.75, borderRadius: 1, cursor: "crosshair" }} />
            ))}
          </div>
          {p.text_pieces.map((t: string, j: number) => (
            <div key={j} onPointerEnter={(e) => { const r = (e.currentTarget.closest("[data-strip]") as HTMLElement).getBoundingClientRect(); setHov({ i: nI + j, x: e.clientX - r.left, y: e.clientY - r.top }); }}
              style={{ flex: "none", minWidth: 46, padding: "6px 8px", borderRadius: 4, background: hov?.i === nI + j ? "var(--accent-soft)" : "color-mix(in srgb, var(--ax-V) 14%, transparent)", textAlign: "center", fontFamily: "var(--mono)", fontSize: 12.5 }}>
              <div style={{ fontWeight: 600 }}>{SHOW(t)}</div>
              <div style={{ fontSize: 10.5, color: "var(--fg-muted)" }}>{p.text_ids[j]}</div>
            </div>
          ))}
        </div>
        <div data-strip style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
          {hov && (
            <Inspect x={hov.x} y={hov.y}>
              {hov.i < nI ? (
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <div style={{ width: 56, height: 56, backgroundImage: `url(${BASE}img_${im.name}_224.png)`, backgroundSize: "896px 896px", backgroundPosition: `-${(hov.i % 16) * 56}px -${Math.floor(hov.i / 16) * 56}px`, imageRendering: "pixelated", borderRadius: 3 }} />
                  <div><b>konum {hov.i}</b> · &lt;image&gt; (id {data.image_token_id})<div className="muted">birleştirmede yama ({Math.floor(hov.i / 16)}, {hov.i % 16}) buraya yazılacak</div></div>
                </div>
              ) : (
                <><b>konum {hov.i}</b> · <span className="mono">{SHOW(p.text_pieces[hov.i - nI])}</span> · id {p.text_ids[hov.i - nI]}</>
              )}
            </Inspect>
          )}
        </div>
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>
        <span style={{ color: "var(--ax-T)" }}>■</span> 256 × &lt;image&gt; (her biri bir hücre, üstüne gel) · <span style={{ color: "var(--ax-V)" }}>■</span> metin · toplam <b style={{ color: "var(--fg)" }}>{p.length}</b> token · attention_mask hepsi 1
      </div>
    </div>
  );
}

/** Görüntü üzerinde kutu çiz → PaliGemma'nın <loc> token'ları */
export function LocBoxTool() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx } = useStore();
  const [box, setBox] = useState<[number, number, number, number]>([0.3, 0.25, 0.7, 0.8]);   // y1 x1 y2 x2, [0, 1]
  const [drag, setDrag] = useState<[number, number] | null>(null);
  if (!data) return <Loading what="görüntü" error={error} />;
  const im = data.images[imageIdx];
  const bin = (v: number) => Math.min(1023, Math.max(0, Math.floor(v * 1024)));
  const [y1, x1, y2, x2] = box;
  const bins = [bin(y1), bin(x1), bin(y2), bin(x2)];
  const toks = bins.map((b) => `<loc${String(b).padStart(4, "0")}>`);
  const pos = (e: React.PointerEvent): [number, number] => { const r = e.currentTarget.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)), Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))]; };
  return (
    <div className="grid2" style={{ alignItems: "start" }}>
      <div>
        <div style={{ position: "relative", width: "100%", maxWidth: 380, aspectRatio: "1", backgroundImage: `url(${BASE}img_${im.name}_224.png)`, backgroundSize: "cover", borderRadius: 3, cursor: "crosshair", touchAction: "none", userSelect: "none" }}
          onPointerDown={(e) => { const p = pos(e); setDrag(p); setBox([p[0], p[1], p[0], p[1]]); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); }}
          onPointerMove={(e) => { if (!drag) return; const p = pos(e); setBox([Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])]); }}
          onPointerUp={() => setDrag(null)}>
          <div style={{ position: "absolute", top: `${y1 * 100}%`, left: `${x1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`, border: "2px solid var(--accent)", background: "color-mix(in srgb, var(--accent) 12%, transparent)", pointerEvents: "none" }} />
        </div>
        <div className="small muted" style={{ marginTop: 4 }}>sürükleyerek bir kutu çiz</div>
      </div>
      <div>
        <div className="small" style={{ marginBottom: 6 }}><b>Modelin yazacağı metin</b> (<code>detect tabela</code> için)</div>
        <div className="mono" style={{ fontSize: 15, background: "var(--code-bg)", padding: "10px 12px", borderRadius: 6, wordBreak: "break-all" }}>
          {toks.map((t, i) => <span key={i} style={{ color: ["var(--ax-T)", "var(--ax-D)", "var(--ax-T)", "var(--ax-D)"][i] }}>{t}</span>)} tabela
        </div>
        <table style={{ marginTop: 12, width: "100%", fontSize: 13.5 }}>
          <thead><tr><th>koordinat</th><th>[0, 1]</th><th>bin (÷1024)</th><th>token id</th><th>224 px'te</th></tr></thead>
          <tbody>
            {(["y_min", "x_min", "y_max", "x_max"] as const).map((n, i) => (
              <tr key={n}><td className="mono">{n}</td><td className="mono">{box[i].toFixed(3)}</td><td className="mono">{bins[i]}</td><td className="mono">{256000 + bins[i]}</td><td className="mono">{((bins[i] / 1024) * 224).toFixed(1)}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 8 }}>Sıra <b>y, x, y, x</b>. Model kutuyu sayı olarak değil, sözlükteki 1024 özel token'dan dördünü <i>yazarak</i> üretiyor. Bu token'lar id 256000–257023 aralığında. 5.2'de normlarının ≈ 1 olduğunu görmüştük. Segmentasyon için ayrıca 128 <code>&lt;segXXX&gt;</code> token'ı var (bir VQ-VAE'nin kod kitabı).</p>
      </div>
    </div>
  );
}

export function ProjectorStats() {
  const { data, error } = useData<any>("pg_slices");
  if (!data) return <Loading what="projektör" error={error} />;
  const h = data.proj_hist;
  const centers: number[] = h.counts.map((_: number, i: number) => (h.edges[i] + h.edges[i + 1]) / 2);
  const synth: number[] = [];
  h.counts.forEach((c: number, i: number) => { const n = Math.round(c / 2000); for (let k = 0; k < n; k++) synth.push(centers[i]); });
  return (
    <div className="grid2">
      <div>
        <div className="small"><b>Ağırlık</b> <span className="mono muted">[2048, 1152]</span> · 2.36 M değer</div>
        <Histogram values={synth} bins={60} color="var(--ax-I)" xLabel="ağırlık değeri" />
        <div className="stats">
          <Stat label="std" value={data.proj_stats.std.toFixed(4)} sub={`ortalama ${data.proj_stats.mean.toExponential(1)}`} />
          <Stat label="max |w|" value={data.proj_stats.absmax.toFixed(3)} />
        </div>
      </div>
      <div>
        <div className="small"><b>Bias</b> <span className="mono muted">[2048]</span></div>
        <Histogram values={Array.from(data.proj_bias.data as Float32Array)} bins={60} color="var(--ax-D)" xLabel="bias değeri" />
      </div>
    </div>
  );
}
