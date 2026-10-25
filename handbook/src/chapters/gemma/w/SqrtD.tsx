// 4.3 the √d scale: real embedding norms, balance of scales, the old path's round trip, bf16 bits
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { bits, toBf16 } from "../../../lib/bf16";
import { fmtNum, l2, rng } from "../../../lib/tensor";
import { Bars, Histogram, LineChart } from "../../../components/charts";
import { Loading, Stat } from "../../../components/ui";
import type { GemmaToy, PgSlices } from "./types";
import { useT } from "../../../i18n";

const SQRT = Math.sqrt(2048);

export function EmbedNorms({ compact = false }: { compact?: boolean }) {
  const tr = useT();
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [scaled, setScaled] = useState(false);
  if (!data) return <Loading what={tr("embedding rows", "embedding satırları")} error={error} />;
  const s = data.sqrt_d;
  const k = scaled ? s.sqrt_bf16 : 1;
  const all = [...s.named_norms].sort((a, b) => b.norm - a.norm);
  const named = compact ? [...all.slice(0, 7), ...all.filter((n) => n.id >= 256000).slice(0, 2)] : all;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <label className="small" style={{ display: "flex", gap: 8, alignItems: "center", cursor: "pointer" }}>
          <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} /> {tr(<>apply <b>×√2048</b> (the code's <code>normalizer</code>)</>, <><b>×√2048</b> uygula (koddaki <code>normalizer</code>)</>)}
        </label>
        <span className="small muted">{tr("real checkpoint", "gerçek checkpoint")} · {s.row_norm_raw.length} {tr("random rows", "rastgele satır")} + {named.length} {tr("selected tokens", "seçili token")}</span>
      </div>
      <div className="row" style={{ gap: 10, marginBottom: 14 }}>
        <Stat label={tr("std per element", "eleman başına std")} value={fmtNum(s.elem_std_raw * k, 3)} sub={scaled ? tr("after scaling", "ölçekten sonra") : tr("raw value in the table", "tablodaki ham değer")} />
        <Stat label={tr("row norm (mean)", "satır normu (ortalama)")} value={fmtNum((s.row_norm_raw.reduce((a, b) => a + b, 0) / s.row_norm_raw.length) * k, 1)} sub={tr("‖e‖, 2048 dims", "‖e‖, 2048 boyut")} />
        <Stat label={tr("after RMSNorm (for comparison)", "RMSNorm sonrası (karşılaştırma)")} value={fmtNum(Math.sqrt(2048), 1)} sub="‖norm(x)‖ ≈ √d · |1+w|" />
      </div>
      <Histogram values={s.row_norm_raw.map((v) => v * k)} bins={40} xLabel={tr("row norm ‖e‖", "satır normu ‖e‖")} color="var(--ax-V)" height={compact ? 140 : 180} />
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

/** Simple model: stream = s·e, each layer adds an update of fixed size in a random direction */
export function ScaleBalance() {
  const tr = useT();
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
  if (!data || !sim) return <Loading what={tr("toy model", "oyuncak model")} error={error} />;
  const sq = Math.sqrt(sim.d);
  return (
    <div>
      <div className="row" style={{ alignItems: "center", gap: 12 }}>
        <span className="small">{tr("scale s =", "ölçek s =")}</span>
        <input type="range" min={0.5} max={48} step={0.5} value={s} onChange={(e) => setS(+e.target.value)} style={{ flex: 1 }} />
        <b className="mono" style={{ minWidth: 70 }}>{s.toFixed(1)}</b>
        <button className="btn" onClick={() => setS(sq)}>√d = {sq}</button>
        <button className="btn" onClick={() => setS(1)}>1</button>
      </div>
      <LineChart height={240} yDomain={[0, 1]} xLabel={tr("layer", "katman")} yLabel="cos(h, h₀)"
        series={[
          { name: tr(`simple model, s = ${s.toFixed(1)}`, `basit model, s = ${s.toFixed(1)}`), color: "var(--accent)", values: sim.run(s) },
          { name: tr("measured: the real code (s = √d)", "ölçülen: gerçek kod (s = √d)"), color: "var(--ax-T)", values: sim.measured.normal.cos_to_embed, dash: true },
          { name: tr("measured: without ×√d (s = 1)", "ölçülen: ×√d olmadan (s = 1)"), color: "var(--warn)", values: sim.measured.no_sqrt_d.cos_to_embed, dash: true },
        ]} />
      <p className="small muted" style={{ margin: "6px 0 0" }}>
        {tr(<>The orange line follows from a single assumption: every branch adds an update in a random direction with the size measured in the toy model.
        The dashed lines were really measured with the project's <code>GemmaModel</code>. At s = 16 and s = 1 they almost overlap.</>, <>Turuncu çizgi tek bir varsayımla hesaplanıyor: her dal akışa rastgele yönlü, oyuncak modelde ölçülen büyüklükte bir güncelleme ekliyor.
        Kesikli çizgiler projedeki <code>GemmaModel</code> ile gerçekten ölçüldü. s = 16 ve s = 1'de ikisi neredeyse üst üste.</>)}
      </p>
    </div>
  );
}

export function RoundTrip() {
  const tr = useT();
  const { data } = useData<GemmaToy>("gemma_toy");
  const [v, setV] = useState(3.7);
  const a = v / SQRT;                         // OLD paligemma.py: image_features / sqrt(d) (no longer in the code)
  const f32 = Math.fround(Math.fround(a) * Math.fround(SQRT));
  const ab = toBf16(a);                       // in a bf16 model the quotient is rounded to bf16
  const nb = toBf16(SQRT);                    // the code's normalizer is 45.25 in bf16
  const b16 = toBf16(ab * nb);
  const vb = toBf16(v);
  const row = (label: string, val: number, note?: string, tone?: string) => (
    <tr><td className="muted">{label}</td><td className="mono" style={{ color: tone }}>{val.toPrecision(8)}</td><td className="small muted">{note}</td></tr>
  );
  const rt = data?.sqrt_d.roundtrip.bfloat16;
  return (
    <div>
      <p className="small muted" style={{ margin: "0 0 12px" }}>
        {tr(<><b>The old path.</b> The first version of the code divided the image features by √2048 and then multiplied them by Gemma's ×√2048. The current code does not
        scale the image at all. This figure shows why the path was shortened.</>, <><b>Eski yol.</b> Kodun ilk sürümü görsel özellikleri önce √2048'e bölüyor, sonra Gemma'nın ×√2048'iyle çarpıyordu. Şimdiki kod görselleri
        hiç ölçeklemiyor. Bu figür yolu neden kısalttığımızı gösteriyor.</>)}
      </p>
      <div className="row" style={{ alignItems: "center", gap: 12, marginBottom: 12 }}>
        <span className="small">{tr("an image feature value", "bir görsel özellik değeri")}</span>
        <input type="range" min={-8} max={8} step={0.01} value={v} onChange={(e) => setV(+e.target.value)} style={{ flex: 1 }} />
        <b className="mono">{v.toFixed(2)}</b>
      </div>
      <div className="grid2">
        <div>
          <div className="eyebrow" style={{ color: "var(--ax-T)" }}>float32</div>
          <table style={{ width: "100%" }}><tbody>
            {row(tr("projector output", "projektör çıktısı"), Math.fround(v))}
            {row("÷ √2048", Math.fround(a), "paligemma.py")}
            {row("× √2048", f32, "gemma.py")}
            {row(tr("difference", "fark"), f32 - Math.fround(v), Math.abs(f32 - Math.fround(v)) < 1e-6 ? "≈ 0" : "", "var(--ok)")}
          </tbody></table>
        </div>
        <div>
          <div className="eyebrow" style={{ color: "var(--accent)" }}>bfloat16</div>
          <table style={{ width: "100%" }}><tbody>
            {row(tr("projector output", "projektör çıktısı"), vb)}
            {row("÷ √2048 → bf16", ab, tr("7-bit mantissa", "7 bit mantis"))}
            {row("× 45.25 → bf16", b16, "normalizer bf16")}
            {row(tr("difference", "fark"), b16 - vb, b16 === vb ? tr("same", "aynı") : `${tr("relative", "göreli")} ${((Math.abs(b16 - vb) / Math.abs(vb || 1)) * 100).toFixed(2)}%`, b16 === vb ? "var(--ok)" : "var(--bad)")}
          </tbody></table>
        </div>
      </div>
      {rt && (
        <div className="row" style={{ marginTop: 14 }}>
          <Stat label={tr("values changed by the bf16 round trip", "bf16'da gidiş-dönüşte değişen değer")} value={`%${((100 * rt.n_diff) / rt.n).toFixed(1)}`}
            sub={tr(`${rt.n_diff} / ${rt.n} random values · largest relative deviation ${(rt.rel * 100).toFixed(2)}%`, `${rt.n_diff} / ${rt.n} rastgele değer · en büyük göreli sapma %${(rt.rel * 100).toFixed(2)}`)} />
        </div>
      )}
    </div>
  );
}

export function Bits({ x = SQRT }: { x?: number }) {
  const tr = useT();
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
        <span style={{ color: "var(--bad)" }}>■</span> {tr("sign", "işaret")} · <span style={{ color: "var(--ax-T)" }}>■</span> {tr("exponent (8 bits, same in both)", "üs (8 bit, ikisinde aynı)")} · <span style={{ color: "var(--ax-D)" }}>■</span> {tr("mantissa: 23 bits in float32, only the first 7 survive in bf16", "mantis: float32'de 23 bit, bf16'da sadece ilk 7'si kalıyor")}
      </div>
    </div>
  );
}

export function NormalizerVariants() {
  const tr = useT();
  const { data, error } = useData<GemmaToy>("gemma_toy");
  if (!data) return <Loading what={tr("experiment", "deney")} error={error} />;
  const v = data.sqrt_d.normalizer_variants;
  const rows: [string, string, { n_diff: number; n: number; maxdiff: number } | null][] = [
    ["torch.tensor(√d, dtype=hidden_states.dtype)", tr("the project's code = HF", "projenin kodu = HF"), null],
    ["hidden_states * (d ** 0.5)", "Python float", v.python_float],
    ["torch.tensor(√d)  # float32", tr("float32 tensor", "float32 tensör"), v.float32_tensor],
  ];
  return (
    <table>
      <thead><tr><th>normalizer</th><th></th><th>{tr("elements that differ from the project's code", "projenin kodundan farklı eleman")}</th><th>{tr("max diff", "max fark")}</th></tr></thead>
      <tbody>
        {rows.map(([code, label, r]) => (
          <tr key={code}>
            <td className="mono" style={{ fontSize: 12.5 }}>{code}</td><td className="small muted">{label}</td>
            <td>{r ? <span className="pill bad">{r.n_diff} / {r.n} ({((r.n_diff / r.n) * 100).toFixed(1)}%)</span> : <span className="pill ok">{tr("reference", "referans")}</span>}</td>
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
