// 4.3 √d ölçeği: gerçek embedding normları, ölçek dengesi, eski yolun gidiş-dönüşü, bf16 bitleri
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { bits, toBf16 } from "../../../lib/bf16";
import { fmtNum, l2, rng } from "../../../lib/tensor";
import { Bars, Histogram, LineChart } from "../../../components/charts";
import { Loading, Stat } from "../../../components/ui";
import type { GemmaToy, PgSlices } from "./types";

const SQRT = Math.sqrt(2048);

export function EmbedNorms({ compact = false }: { compact?: boolean }) {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [scaled, setScaled] = useState(false);
  if (!data) return <Loading what="embedding satırları" error={error} />;
  const s = data.sqrt_d;
  const k = scaled ? s.sqrt_bf16 : 1;
  const all = [...s.named_norms].sort((a, b) => b.norm - a.norm);
  const named = compact ? [...all.slice(0, 7), ...all.filter((n) => n.id >= 256000).slice(0, 2)] : all;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <label className="small" style={{ display: "flex", gap: 8, alignItems: "center", cursor: "pointer" }}>
          <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} /> <b>×√2048</b> uygula (senin <code>normalizer</code>'ın)
        </label>
        <span className="small muted">gerçek checkpoint · {s.row_norm_raw.length} rastgele satır + {named.length} seçili token</span>
      </div>
      <div className="row" style={{ gap: 10, marginBottom: 14 }}>
        <Stat label="eleman başına std" value={fmtNum(s.elem_std_raw * k, 3)} sub={scaled ? "ölçekten sonra" : "tablodaki ham değer"} />
        <Stat label="satır normu (ortalama)" value={fmtNum((s.row_norm_raw.reduce((a, b) => a + b, 0) / s.row_norm_raw.length) * k, 1)} sub="‖e‖, 2048 boyut" />
        <Stat label="RMSNorm sonrası (karşılaştırma)" value={fmtNum(Math.sqrt(2048), 1)} sub="‖norm(x)‖ ≈ √d · |1+w|" />
      </div>
      <Histogram values={s.row_norm_raw.map((v) => v * k)} bins={40} xLabel="satır normu ‖e‖" color="var(--ax-V)" height={compact ? 140 : 180} />
      <div style={{ marginTop: 16 }}>
        <Bars items={named.map((n) => ({
          label: <span className="mono" style={{ fontSize: 12.5 }}>{n.piece.replace("▁", "␣")} <span className="muted">#{n.id}</span></span>,
          value: n.norm * k,
          color: n.id >= 256000 ? "var(--ax-I)" : n.piece.startsWith("<") ? "var(--accent)" : "var(--ax-V)",
        }))} fmt={(v) => v.toFixed(1)} />
      </div>
    </div>
  );
}

/** Basit model: akış = s·e, her katman rastgele yönlü sabit büyüklükte bir güncelleme ekliyor */
export function ScaleBalance() {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [s, setS] = useState(16);   // oyuncakta d=256 -> √d = 16
  const sim = useMemo(() => {
    if (!data) return null;
    const d = data.toy_config.hidden_size;
    const r = rng(3);
    const e = Array.from({ length: d }, () => r.normal() * data.sqrt_d.elem_std_raw);
    const st = data.stream.normal;
    const run = (scale: number) => {
      let h = e.map((v) => v * scale);
      const h0 = h.slice();
      const cos = [1];
      for (let l = 0; l < st.attn_update.length; l++) {
        for (const U of [st.attn_update[l], st.mlp_update[l]]) {
          const u = Array.from({ length: d }, () => r.normal());
          const n = l2(u);
          h = h.map((v, i) => v + (u[i] / n) * U);
        }
        cos.push(h.reduce((a, v, i) => a + v * h0[i], 0) / (l2(h) * l2(h0)));
      }
      return cos;
    };
    return { run, d, measured: data.stream };
  }, [data]);
  if (!data || !sim) return <Loading what="oyuncak model" error={error} />;
  const sq = Math.sqrt(sim.d);
  return (
    <div>
      <div className="row" style={{ alignItems: "center", gap: 12 }}>
        <span className="small">ölçek s =</span>
        <input type="range" min={0.5} max={48} step={0.5} value={s} onChange={(e) => setS(+e.target.value)} style={{ flex: 1 }} />
        <b className="mono" style={{ minWidth: 70 }}>{s.toFixed(1)}</b>
        <button className="btn" onClick={() => setS(sq)}>√d = {sq}</button>
        <button className="btn" onClick={() => setS(1)}>1</button>
      </div>
      <LineChart height={240} yDomain={[0, 1]} xLabel="katman" yLabel="cos(h, h₀)"
        series={[
          { name: `basit model, s = ${s.toFixed(1)}`, color: "var(--accent)", values: sim.run(s) },
          { name: "ölçülen: senin kodun (s = √d)", color: "var(--ax-T)", values: sim.measured.normal.cos_to_embed, dash: true },
          { name: "ölçülen: ×√d olmadan (s = 1)", color: "var(--warn)", values: sim.measured.no_sqrt_d.cos_to_embed, dash: true },
        ]} />
      <p className="small muted" style={{ margin: "6px 0 0" }}>
        Turuncu çizgi tek bir varsayımla hesaplanıyor: her dal akışa rastgele yönlü, oyuncak modelde ölçülen büyüklükte bir güncelleme ekliyor.
        Kesikli çizgiler senin <code>GemmaModel</code>'inle gerçekten ölçüldü. s = 16 ve s = 1'de ikisi neredeyse üst üste.
      </p>
    </div>
  );
}

export function RoundTrip() {
  const { data } = useData<GemmaToy>("gemma_toy");
  const [v, setV] = useState(3.7);
  const a = v / SQRT;                         // ESKİ paligemma.py: image_features / sqrt(d) (artık kodda yok)
  const f32 = Math.fround(Math.fround(a) * Math.fround(SQRT));
  const ab = toBf16(a);                       // bf16 modelde bölmenin sonucu bf16'ya yuvarlanır
  const nb = toBf16(SQRT);                    // senin normalizer'ın bf16'da 45.25
  const b16 = toBf16(ab * nb);
  const vb = toBf16(v);
  const row = (label: string, val: number, note?: string, tone?: string) => (
    <tr><td className="muted">{label}</td><td className="mono" style={{ color: tone }}>{val.toPrecision(8)}</td><td className="small muted">{note}</td></tr>
  );
  const rt = data?.sqrt_d.roundtrip.bfloat16;
  return (
    <div>
      <p className="small muted" style={{ margin: "0 0 12px" }}>
        <b>Eski yol.</b> Kodun ilk sürümü görsel özellikleri önce √2048'e bölüyor, sonra Gemma'nın ×√2048'iyle çarpıyordu. Şimdiki kod görselleri
        hiç ölçeklemiyor. Bu figür yolu neden kısalttığımızı gösteriyor.
      </p>
      <div className="row" style={{ alignItems: "center", gap: 12, marginBottom: 12 }}>
        <span className="small">bir görsel özellik değeri</span>
        <input type="range" min={-8} max={8} step={0.01} value={v} onChange={(e) => setV(+e.target.value)} style={{ flex: 1 }} />
        <b className="mono">{v.toFixed(2)}</b>
      </div>
      <div className="grid2">
        <div>
          <div className="eyebrow" style={{ color: "var(--ax-T)" }}>float32</div>
          <table style={{ width: "100%" }}><tbody>
            {row("projektör çıktısı", Math.fround(v))}
            {row("÷ √2048", Math.fround(a), "paligemma.py")}
            {row("× √2048", f32, "gemma.py")}
            {row("fark", f32 - Math.fround(v), Math.abs(f32 - Math.fround(v)) < 1e-6 ? "≈ 0" : "", "var(--ok)")}
          </tbody></table>
        </div>
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>bfloat16</div>
          <table style={{ width: "100%" }}><tbody>
            {row("projektör çıktısı", vb)}
            {row("÷ √2048 → bf16", ab, "7 bit mantis")}
            {row("× 45.25 → bf16", b16, "normalizer bf16")}
            {row("fark", b16 - vb, b16 === vb ? "aynı" : `göreli ${((Math.abs(b16 - vb) / Math.abs(vb || 1)) * 100).toFixed(2)}%`, b16 === vb ? "var(--ok)" : "var(--bad)")}
          </tbody></table>
        </div>
      </div>
      {rt && (
        <div className="row" style={{ marginTop: 14 }}>
          <Stat label="bf16'da gidiş-dönüşte değişen değer" value={`%${((100 * rt.n_diff) / rt.n).toFixed(1)}`}
            sub={`${rt.n_diff} / ${rt.n} rastgele değer · en büyük göreli sapma %${(rt.rel * 100).toFixed(2)}`} />
        </div>
      )}
    </div>
  );
}

export function Bits({ x = SQRT }: { x?: number }) {
  const f = bits(x, "f32"), b = bits(x, "bf16");
  const cell = (s: string, color: string, dim = false) => s.split("").map((c, i) => (
    <span key={i} style={{ display: "inline-block", width: 15, textAlign: "center", background: `color-mix(in srgb, ${color} ${dim ? 8 : 22}%, transparent)`, color: dim ? "var(--fg-faint)" : "var(--fg)", borderRadius: 2, margin: "0 0.5px" }}>{c}</span>
  ));
  return (
    <div className="mono" style={{ fontSize: 13, overflowX: "auto" }}>
      <div style={{ marginBottom: 8 }}>
        <span className="muted" style={{ display: "inline-block", width: 70 }}>float32</span>
        {cell(f.sign, "var(--bad)")} {cell(f.exp, "var(--ax-T)")} {cell(f.mant.slice(0, 7), "var(--ax-D)")}{cell(f.mant.slice(7), "var(--ax-D)", true)}
        <span style={{ marginLeft: 10 }}>= {Math.fround(x)}</span>
      </div>
      <div>
        <span className="muted" style={{ display: "inline-block", width: 70 }}>bfloat16</span>
        {cell(b.sign, "var(--bad)")} {cell(b.exp, "var(--ax-T)")} {cell(b.mant, "var(--ax-D)")}
        <span style={{ marginLeft: 10 }}>= <b>{toBf16(x)}</b></span>
      </div>
      <div className="small muted" style={{ fontFamily: "var(--sans)", marginTop: 8 }}>
        <span style={{ color: "var(--bad)" }}>■</span> işaret · <span style={{ color: "var(--ax-T)" }}>■</span> üs (8 bit, ikisinde aynı) · <span style={{ color: "var(--ax-D)" }}>■</span> mantis: float32'de 23 bit, bf16'da sadece ilk 7'si kalıyor
      </div>
    </div>
  );
}

export function NormalizerVariants() {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  if (!data) return <Loading what="deney" error={error} />;
  const v = data.sqrt_d.normalizer_variants;
  const rows: [string, string, { n_diff: number; n: number; maxdiff: number } | null][] = [
    ["torch.tensor(√d, dtype=hidden_states.dtype)", "senin kodun = HF", null],
    ["hidden_states * (d ** 0.5)", "Python float", v.python_float],
    ["torch.tensor(√d)  # float32", "float32 tensör", v.float32_tensor],
  ];
  return (
    <table>
      <thead><tr><th>normalizer</th><th></th><th>senin kodundan farklı eleman</th><th>max fark</th></tr></thead>
      <tbody>
        {rows.map(([code, label, r]) => (
          <tr key={code}>
            <td className="mono" style={{ fontSize: 12.5 }}>{code}</td><td className="small muted">{label}</td>
            <td>{r ? <span className="pill bad">{r.n_diff} / {r.n} ({((r.n_diff / r.n) * 100).toFixed(1)}%)</span> : <span className="pill ok">referans</span>}</td>
            <td className="mono">{r ? r.maxdiff : "0"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LocTokensNote() {
  const { data } = useData<PgSlices>("pg_slices");
  return data ? <span className="mono">{data.embed_shape.join(" × ")}</span> : null;
}
