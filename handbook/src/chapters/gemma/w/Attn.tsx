// 4.11 GemmaAttention: adım adım şekiller, nedensel maske, gerçek ağırlıklar, √head_dim ölçeği
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { rng, softmax } from "../../../lib/tensor";
import { Bars, Heatmap } from "../../../components/charts";
import { CodeLines } from "../../../components/CodeLines";
import { Loading, S, Seg, Stat } from "../../../components/ui";
import type { AttentionToy } from "./types";

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

/** [ad, şekil, eksenler, ilk görünüştüğü adım, açıklama] gerçek config, 260 token */
const ROWS: [string, string, string, number, string][] = [
  ["hidden_states", "1, 260, 2048", "B,T,D", 0, "katmanın girişi (norm'lanmış kopya)"],
  ["cos, sin", "1, 260, 256", "B,T,hd", 0, "model bir kez hesaplıyor"],
  ["attention_mask", "1, 1, 260, 260", "B,H,T,T", 0, "toplanan maske"],
  ["q", "1, 8, 260, 256", "B,H,T,hd", 1, "8 sorgu head'i"],
  ["k, v", "1, 1, 260, 256", "B,H,T,hd", 1, "tek KV head'i (MQA)"],
  ["k, v (genişletilmiş)", "1, 8, 260, 256", "B,H,T,hd", 3, "repeat_kv: Q ile aynı head sayısı"],
  ["scores", "1, 8, 260, 260", "B,H,T,T", 4, "her head için T × T ilgi tablosu"],
  ["weights", "1, 8, 260, 260", "B,H,T,T", 6, "satırlar 1'e toplanıyor"],
  ["out (head başına)", "1, 8, 260, 256", "B,H,T,hd", 7, "V'nin ağırlıklı karışımı"],
  ["out (birleşik) → o_proj", "1, 260, 2048", "B,T,D", 8, "katmanın girişiyle aynı şekil"],
];
const NOTE = [
  "Girdiler: akışın norm'lu kopyası, cos/sin tablosu ve maske. Çıktı girişle aynı şekilde olacak.",
  "Üç projeksiyon, sonra son boyut head'lere bölünüyor ve head ekseni öne alınıyor. Q 8 head, K ve V 1 head.",
  "Q ve K döndürülüyor, V'ye dokunulmuyor. Şekiller aynı kalıyor, sadece değerler değişiyor.",
  "Tek KV head'i 8 Q head'inin hepsine hizmet edecek biçimde genişletiliyor (sadece hesap anında).",
  "Her head'de her token çifti için bir skor: 260 × 260 = 67.600 sayı, 8 head için 540.800. 1/√256 = 1/16 ile ölçekleniyor.",
  "Yasak yerlere (gelecekteki token'lar) çok küçük bir sayı ekleniyor. Şekil değişmiyor.",
  "Satır başına softmax, float32'de. Yasak yerler 0 oluyor, her satır 1'e toplanıyor.",
  "Her token, baktığı token'ların V'lerinin ağırlıklı karışımını alıyor.",
  "Head'ler yan yana yapıştırılıyor (8 × 256 = 2048) ve o_proj ile karıştırılıyor.",
];

/** Scrolly figürü: aktif kod satırı, bu adıma kadar oluşmuş tensörlerin şekil tablosu */
export function AttnFlow({ step }: { step: number }) {
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
              <span className="muted small">{r[4]}</span>
            </div>
          );
        })}
      </div>
      <p className="small" style={{ margin: "10px 0 0" }}>{NOTE[step]}</p>
    </div>
  );
}

/** make_causal_mask(8, ...) çıktısı: 0 = izinli, −min = yasak. Satıra gelince o token'ın görebildikleri vurgulanıyor. */
export function MaskGrid() {
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
        {row === null ? "Bir satırın üstüne gel." : <>Token <b className="mono">{row}</b> yalnızca <b className="mono">k0 … k{row}</b>'ı görebiliyor ({row + 1} token). Sonrası softmax'tan önce −min ile toplanıp 0 ağırlık alıyor.</>}
      </p>
    </div>
  );
}

/** Senin GemmaAttention'ınla (oyuncak, T = 8, 4 head): maskeli ve maskesiz attention ağırlıkları */
export function MaskWeights() {
  const { data, error } = useData<AttentionToy>("attention_toy");
  const [masked, setMasked] = useState(true);
  const [head, setHead] = useState(0);
  if (!data) return <Loading what="attention ağırlıkları" error={error} />;
  const a = data.attention;
  const arr = masked ? a.weights_causal : a.weights_nomask;
  const T = arr.shape[1];
  const slice = arr.data.slice(head * T * T, (head + 1) * T * T);
  return (
    <div>
      <div className="row" style={{ gap: 12, marginBottom: 10, flexWrap: "wrap" }}>
        <Seg value={masked ? "m" : "n"} onChange={(v) => setMasked(v === "m")} options={[["m", "nedensel maske var"], ["n", "maske yok"]]} />
        <Seg value={head} onChange={setHead} options={[[0, "head 0"], [1, "head 1"], [2, "head 2"], [3, "head 3"]]} />
      </div>
      <div style={{ maxWidth: 420 }}>
        <Heatmap values={slice} rows={T} cols={T} height={300} domain={[0, 1]} rowLabel="sorgu token'ı i" colLabel="anahtar token'ı j"
          cellLabel={(i, j, v) => `token ${i} → token ${j}: ${v.toFixed(3)}${j > i ? (masked ? " (yasak)" : " (gelecek)") : ""}`} />
      </div>
      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <Stat label="satır toplamı, 1'den en büyük sapma" value={a.row_sums_maxdiff.toExponential(1)} tone="ok" />
        <Stat label="köşegen üstü en büyük ağırlık (maskeli)" value={a.upper_triangle_max.toExponential(1)} tone="ok" />
        <Stat label="elle hesap ile forward arası fark" value={a.manual_vs_forward_maxdiff.toExponential(1)} sub="aynı adımlar, dışarıda" tone="ok" />
      </div>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label="son token'ı değiştirince önceki çıktılar, maskeli" value={a.leak_with_mask.toExponential(1)} sub="değişmiyor: gelecek geçmişi etkilemiyor" tone="ok" />
        <Stat label="aynı deney, maskesiz" value={a.leak_without_mask.toFixed(1)} sub="değişiyor: gelecek sızıyor" tone="bad" />
      </div>
    </div>
  );
}

/** 1/√head_dim: yoksa skorlar head_dim büyüdükçe büyür ve softmax tek token'a sıkışır */
export function ScaleDemo() {
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
        <Seg value={scaled ? "s" : "r"} onChange={(v) => setScaled(v === "s")} options={[["s", "× 1/√head_dim"], ["r", "ölçeksiz"]]} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>yeni rastgele q, k</button>
      </div>
      <Bars max={1} fmt={(v) => `${(v * 100).toFixed(0)}%`} items={w.map((v, j) => ({ label: <span className="mono">k{j}</span>, value: v, color: v === Math.max(...w) ? "var(--accent)" : "var(--ax-T)", note: <span className="muted small">skor {sc[j].toFixed(1)}</span> }))} />
      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <Stat label="en büyük ağırlık" value={`${(Math.max(...w) * 100).toFixed(0)}%`} tone={Math.max(...w) > 0.9 ? "bad" : undefined} />
        {st && <Stat label="iç çarpımın std'si, ölçeksiz" value={st.std_raw.toFixed(1)} sub={`≈ √${hd} = ${Math.sqrt(hd).toFixed(0)}`} />}
        {st && <Stat label="ölçekli" value={st.std_scaled.toFixed(2)} sub="head_dim'den bağımsız" tone="ok" />}
      </div>
    </div>
  );
}
