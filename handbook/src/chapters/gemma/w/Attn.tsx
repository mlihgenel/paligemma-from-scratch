// 4.11 GemmaAttention: shapes step by step, the causal mask, real weights, the √head_dim scale
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { rng, softmax } from "../../../lib/tensor";
import { Bars, Heatmap } from "../../../components/charts";
import { CodeLines } from "../../../components/CodeLines";
import { Loading, S, Seg, Stat } from "../../../components/ui";
import type { AttentionToy } from "./types";
import { useLang, useT } from "../../../i18n";
import type { L } from "../../../i18n";

const ACTIVE: string[][] = [
  ["def forward"],
  ["q = self.q_proj", "k = self.k_proj", "v = self.v_proj"],
  ["q, k = apply_rotary_pos_emb(q, k, cos, sin)"],
  ["k = repeat_kv", "v = repeat_kv"],
  ["scores = torch.matmul(q, k.transpose(2, 3))"],
  ["scores = scores + attention_mask"],
  ["weights = F.softmax"],
  ["out = torch.matmul(weights, v)"],
  ["out = out.transpose(1, 2).reshape", "return self.o_proj(out)"],
];

/** [name, shape, axes, step it first appears, description] real config, 260 tokens */
const ROWS: [string, string, string, number, L<string>][] = [
  ["hidden_states", "1, 260, 2048", "B,T,D", 0, { en: "the layer input (normalized copy)", tr: "katmanın girişi (norm'lanmış kopya)" }],
  ["cos, sin", "1, 260, 256", "B,T,hd", 0, { en: "computed once by the model", tr: "model bir kez hesaplıyor" }],
  ["attention_mask", "1, 1, 260, 260", "B,H,T,T", 0, { en: "additive mask", tr: "toplanan maske" }],
  ["q", "1, 8, 260, 256", "B,H,T,hd", 1, { en: "8 query heads", tr: "8 sorgu head'i" }],
  ["k, v", "1, 1, 260, 256", "B,H,T,hd", 1, { en: "a single KV head (MQA)", tr: "tek KV head'i (MQA)" }],
  ["k, v (expanded)", "1, 8, 260, 256", "B,H,T,hd", 3, { en: "repeat_kv: same head count as Q", tr: "repeat_kv: Q ile aynı head sayısı" }],
  ["scores", "1, 8, 260, 260", "B,H,T,T", 4, { en: "a T × T attention table per head", tr: "her head için T × T ilgi tablosu" }],
  ["weights", "1, 8, 260, 260", "B,H,T,T", 6, { en: "rows sum to 1", tr: "satırlar 1'e toplanıyor" }],
  ["out (per head)", "1, 8, 260, 256", "B,H,T,hd", 7, { en: "a weighted mix of V", tr: "V'nin ağırlıklı karışımı" }],
  ["out (merged) → o_proj", "1, 260, 2048", "B,T,D", 8, { en: "same shape as the layer input", tr: "katmanın girişiyle aynı şekil" }],
];
const NOTE: L<string>[] = [
  { en: "Inputs: the normalized copy of the stream, the cos/sin table and the mask. The output will have the same shape as the input.", tr: "Girdiler: akışın norm'lu kopyası, cos/sin tablosu ve maske. Çıktı girişle aynı şekilde olacak." },
  { en: "Three projections, then the last dim is split into heads and the head axis moves forward. Q has 8 heads, K and V 1 head.", tr: "Üç projeksiyon, sonra son boyut head'lere bölünüyor ve head ekseni öne alınıyor. Q 8 head, K ve V 1 head." },
  { en: "Q and K are rotated, V is left alone. The shapes stay the same, only the values change.", tr: "Q ve K döndürülüyor, V'ye dokunulmuyor. Şekiller aynı kalıyor, sadece değerler değişiyor." },
  { en: "The single KV head is expanded so it can serve all 8 Q heads (only at compute time).", tr: "Tek KV head'i 8 Q head'inin hepsine hizmet edecek biçimde genişletiliyor (sadece hesap anında)." },
  { en: "One score per token pair in every head: 260 × 260 = 67,600 numbers, 540,800 for 8 heads. Scaled by 1/√256 = 1/16.", tr: "Her head'de her token çifti için bir skor: 260 × 260 = 67.600 sayı, 8 head için 540.800. 1/√256 = 1/16 ile ölçekleniyor." },
  { en: "A very small number is added at the blocked positions (future tokens). The shape does not change.", tr: "Yasak yerlere (gelecekteki token'lar) çok küçük bir sayı ekleniyor. Şekil değişmiyor." },
  { en: "Softmax per row, in float32. Blocked positions become 0 and every row sums to 1.", tr: "Satır başına softmax, float32'de. Yasak yerler 0 oluyor, her satır 1'e toplanıyor." },
  { en: "Each token takes a weighted mix of the V's of the tokens it looks at.", tr: "Her token, baktığı token'ların V'lerinin ağırlıklı karışımını alıyor." },
  { en: "The heads are glued side by side (8 × 256 = 2048) and mixed by o_proj.", tr: "Head'ler yan yana yapıştırılıyor (8 × 256 = 2048) ve o_proj ile karıştırılıyor." },
];

/** Scrolly figure: the active code line and a shape table of the tensors created up to this step */
export function AttnFlow({ step }: { step: number }) {
  const lang = useLang();
  return (
    <div>
      <CodeLines file="gemma.py" symbol="GemmaAttention.forward" maxHeight={250} active={ACTIVE[step]} />
      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "auto auto 1fr", gap: "4px 12px", fontSize: 13.5, alignItems: "baseline", fontFamily: "var(--sans)" }}>
        {ROWS.filter((r) => r[3] <= step).map((r) => {
          const fresh = r[3] === step || (step === 5 && r[0] === "scores") || (step === 2 && r[0] === "q");
          return (
            <div key={r[0]} style={{ display: "contents", fontWeight: fresh ? 700 : 400, opacity: fresh ? 1 : 0.6 }}>
              <span className="mono">{r[0]}</span>
              <S s={r[1]} ax={r[2]} />
              <span className="muted small">{r[4][lang]}</span>
            </div>
          );
        })}
      </div>
      <p className="small" style={{ margin: "10px 0 0" }}>{NOTE[step][lang]}</p>
    </div>
  );
}

/** Output of make_causal_mask(8, ...): 0 = allowed, −min = blocked. Hovering a row highlights what that token can see. */
export function MaskGrid() {
  const tr = useT();
  const T = 8;
  const [row, setRow] = useState<number | null>(null);
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: `auto repeat(${T}, 1fr)`, gap: 3, maxWidth: 480, fontFamily: "var(--mono)", fontSize: 11.5 }} data-testid="mask-grid">
        <span />
        {Array.from({ length: T }, (_, j) => <span key={j} className="muted" style={{ textAlign: "center" }}>k{j}</span>)}
        {Array.from({ length: T }, (_, i) => (
          <div key={i} style={{ display: "contents" }} onMouseEnter={() => setRow(i)} onMouseLeave={() => setRow(null)}>
            <span className="muted" style={{ textAlign: "right", paddingRight: 4, alignSelf: "center" }}>q{i}</span>
            {Array.from({ length: T }, (_, j) => {
              const ok = j <= i;
              const hl = row === i;
              return (
                <span key={j} style={{ textAlign: "center", padding: "7px 0", borderRadius: 4, background: ok ? (hl ? "color-mix(in srgb, var(--ok) 35%, transparent)" : "var(--bg-sunk)") : (hl ? "color-mix(in srgb, var(--bad) 35%, transparent)" : "color-mix(in srgb, var(--bad) 12%, transparent)"), color: ok ? "var(--fg-muted)" : "var(--bad)", transition: "background .15s" }}>
                  {ok ? "0" : "−min"}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <p className="small muted" style={{ margin: "8px 0 0", minHeight: 20 }}>
        {row === null ? tr("Hover a row.", "Bir satırın üstüne gel.") : tr(<>Token <b className="mono">{row}</b> can only see <b className="mono">k0 … k{row}</b> ({row + 1} tokens). Everything after gets −min added before softmax and ends up with weight 0.</>, <>Token <b className="mono">{row}</b> yalnızca <b className="mono">k0 … k{row}</b>'ı görebiliyor ({row + 1} token). Sonrası softmax'tan önce −min ile toplanıp 0 ağırlık alıyor.</>)}
      </p>
    </div>
  );
}

/** With the project's GemmaAttention (toy, T = 8, 4 heads): attention weights with and without the mask */
export function MaskWeights() {
  const tr = useT();
  const { data, error } = useData<AttentionToy>("attention_toy");
  const [masked, setMasked] = useState(true);
  const [head, setHead] = useState(0);
  if (!data) return <Loading what={tr("attention weights", "attention ağırlıkları")} error={error} />;
  const a = data.attention;
  const arr = masked ? a.weights_causal : a.weights_nomask;
  const T = arr.shape[1];
  const slice = arr.data.slice(head * T * T, (head + 1) * T * T);
  return (
    <div>
      <div className="row" style={{ gap: 12, marginBottom: 10, flexWrap: "wrap" }}>
        <Seg value={masked ? "m" : "n"} onChange={(v) => setMasked(v === "m")} options={[["m", tr("causal mask on", "nedensel maske var")], ["n", tr("no mask", "maske yok")]]} />
        <Seg value={head} onChange={setHead} options={[[0, "head 0"], [1, "head 1"], [2, "head 2"], [3, "head 3"]]} />
      </div>
      <div style={{ maxWidth: 420 }}>
        <Heatmap values={slice} rows={T} cols={T} height={300} domain={[0, 1]} rowLabel={tr("query token i", "sorgu token'ı i")} colLabel={tr("key token j", "anahtar token'ı j")}
          cellLabel={(i, j, v) => `token ${i} → token ${j}: ${v.toFixed(3)}${j > i ? (masked ? " (yasak)" : " (gelecek)") : ""}`} />
      </div>
      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <Stat label={tr("row sums, largest deviation from 1", "satır toplamı, 1'den en büyük sapma")} value={a.row_sums_maxdiff.toExponential(1)} tone="ok" />
        <Stat label={tr("largest weight above the diagonal (masked)", "köşegen üstü en büyük ağırlık (maskeli)")} value={a.upper_triangle_max.toExponential(1)} tone="ok" />
        <Stat label={tr("difference between manual math and forward", "elle hesap ile forward arası fark")} value={a.manual_vs_forward_maxdiff.toExponential(1)} sub={tr("the same steps, done outside", "aynı adımlar, dışarıda")} tone="ok" />
      </div>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label={tr("earlier outputs after changing the last token, masked", "son token'ı değiştirince önceki çıktılar, maskeli")} value={a.leak_with_mask.toExponential(1)} sub={tr("unchanged: the future does not affect the past", "değişmiyor: gelecek geçmişi etkilemiyor")} tone="ok" />
        <Stat label={tr("same experiment, no mask", "aynı deney, maskesiz")} value={a.leak_without_mask.toFixed(1)} sub={tr("changed: the future leaks", "değişiyor: gelecek sızıyor")} tone="bad" />
      </div>
    </div>
  );
}

/** 1/√head_dim: without it the scores grow with head_dim and softmax collapses onto a single token */
export function ScaleDemo() {
  const tr = useT();
  const { data } = useData<AttentionToy>("attention_toy");
  const [hd, setHd] = useState(256);
  const [scaled, setScaled] = useState(true);
  const [seed, setSeed] = useState(1);
  const K = 8;
  const raw = useMemo(() => {
    const r = rng(seed * 1000 + hd);
    const q = Array.from({ length: hd }, () => r.normal());
    return Array.from({ length: K }, () => { let s = 0; for (let i = 0; i < hd; i++) s += q[i] * r.normal(); return s; });
  }, [hd, seed]);
  const sc = scaled ? raw.map((s) => s / Math.sqrt(hd)) : raw;
  const w = softmax(sc);
  const st = data?.scale_stats[String(hd)];
  return (
    <div>
      <div className="row" style={{ gap: 12, marginBottom: 10, flexWrap: "wrap" }}>
        <Seg value={hd} onChange={setHd} options={[[16, "head_dim 16"], [64, "head_dim 64"], [256, "head_dim 256 (Gemma)"]]} />
        <Seg value={scaled ? "s" : "r"} onChange={(v) => setScaled(v === "s")} options={[["s", "× 1/√head_dim"], ["r", tr("unscaled", "ölçeksiz")]]} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>yeni rastgele q, k</button>
      </div>
      <Bars max={1} fmt={(v) => `${(v * 100).toFixed(0)}%`} items={w.map((v, j) => ({ label: <span className="mono">k{j}</span>, value: v, color: v === Math.max(...w) ? "var(--accent)" : "var(--ax-T)", note: <span className="muted small">skor {sc[j].toFixed(1)}</span> }))} />
      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <Stat label={tr("largest weight", "en büyük ağırlık")} value={`${(Math.max(...w) * 100).toFixed(0)}%`} tone={Math.max(...w) > 0.9 ? "bad" : undefined} />
        {st && <Stat label={tr("std of the dot product, unscaled", "iç çarpımın std'si, ölçeksiz")} value={st.std_raw.toFixed(1)} sub={`≈ √${hd} = ${Math.sqrt(hd).toFixed(0)}`} />}
        {st && <Stat label={tr("scaled", "ölçekli")} value={st.std_scaled.toFixed(2)} sub={tr("independent of head_dim", "head_dim'den bağımsız")} tone="ok" />}
      </div>
    </div>
  );
}
