// 4.9 RoPE: dönen çiftler, konum farkı, frekans tablosu, boyut düzeni, bf16'da açı
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { toBf16 } from "../../../lib/bf16";
import { LineChart } from "../../../components/charts";
import { Loading, Seg, Stat } from "../../../components/ui";
import type { AttentionToy } from "./types";

const rot = ([a, b]: [number, number], ang: number): [number, number] => [a * Math.cos(ang) - b * Math.sin(ang), b * Math.cos(ang) + a * Math.sin(ang)];
const PAIR_COLORS = ["#d95f4b", "#e0a030", "#6aa84f", "#2f9e8f", "#3d85c6", "#6f6fd8", "#b05fc0", "#c05f8a"];

/** Tek bir 2 boyutlu çift: q m. konumda, k n. konumda döndürülüyor. İç çarpım yalnızca m − n'ye bağlı. */
export function RopeDial() {
  const { data, error } = useData<AttentionToy>("attention_toy");
  const [m, setM] = useState(5);
  const [n, setN] = useState(2);
  const [pair, setPair] = useState(0);
  if (!data) return <Loading what="RoPE frekansları" error={error} />;
  const theta = data.rope_real.inv_freq[pair];
  const q0: [number, number] = [0.9, 0.35], k0: [number, number] = [0.6, -0.55];
  const q = rot(q0, m * theta), k = rot(k0, n * theta);
  const dot = q[0] * k[0] + q[1] * k[1];
  const R = 110, C = 130;
  const P = (v: [number, number]) => [C + v[0] * R, C - v[1] * R] as const;
  const arrow = (v: [number, number], color: string, label: string, dim = false) => {
    const [x, y] = P(v);
    return (
      <g opacity={dim ? 0.35 : 1}>
        <line x1={C} y1={C} x2={x} y2={y} stroke={color} strokeWidth={3} strokeLinecap="round" />
        <circle cx={x} cy={y} r={5} fill={color} />
        <text x={x + (v[0] >= 0 ? 8 : -8)} y={y - 8} fontSize="12" fontWeight={700} textAnchor={v[0] >= 0 ? "start" : "end"} fill={color}>{label}</text>
      </g>
    );
  };
  const pairs: [number, string][] = [[0, "çift 0 (hızlı)"], [16, "çift 16"], [48, "çift 48"], [100, "çift 100 (yavaş)"]];
  return (
    <div className="grid2" style={{ alignItems: "center" }}>
      <svg viewBox="0 0 260 260" style={{ width: "100%", maxWidth: 300, margin: "0 auto", display: "block" }} data-testid="rope-dial">
        <circle cx={C} cy={C} r={R} fill="none" stroke="var(--border-strong)" strokeDasharray="3 4" />
        <line x1={C - R} x2={C + R} y1={C} y2={C} stroke="var(--border)" />
        <line x1={C} x2={C} y1={C - R} y2={C + R} stroke="var(--border)" />
        {arrow(q0, "var(--ax-T)", "q₀", true)}
        {arrow(k0, "var(--warn)", "k₀", true)}
        {arrow(q, "var(--ax-T)", "q")}
        {arrow(k, "var(--warn)", "k")}
      </svg>
      <div style={{ fontFamily: "var(--sans)", fontSize: 14 }}>
        <div style={{ marginBottom: 8 }}><Seg value={pair} onChange={setPair} options={pairs} /></div>
        <label className="small">q'nun konumu m <b className="mono">{m}</b>
          <input aria-label="m" type="range" min={0} max={40} value={m} onChange={(e) => setM(+e.target.value)} style={{ width: "100%" }} /></label>
        <label className="small">k'nın konumu n <b className="mono">{n}</b>
          <input aria-label="n" type="range" min={0} max={40} value={n} onChange={(e) => setN(+e.target.value)} style={{ width: "100%" }} /></label>
        <div className="row" style={{ gap: 8, margin: "8px 0" }}>
          <button className="btn" onClick={() => { setM(Math.min(40, m + 1)); setN(Math.min(40, n + 1)); }} disabled={m >= 40 || n >= 40}>ikisini +1 kaydır</button>
          <button className="btn" onClick={() => { setM(Math.max(0, m - 1)); setN(Math.max(0, n - 1)); }} disabled={m <= 0 || n <= 0}>ikisini −1 kaydır</button>
        </div>
        <div className="row" style={{ gap: 10 }}>
          <Stat label="θ (rad / konum)" value={theta.toPrecision(3)} />
          <Stat label="m − n" value={m - n} />
          <Stat label="q · k" value={dot.toFixed(4)} tone="ok" />
        </div>
        <p className="small muted" style={{ margin: "8px 0 0" }}>
          Soluk oklar dönmemiş q₀ ve k₀; q m. konumdaki, k n. konumdaki hali. Ok ucuna dönen açı <b>konum × θ</b>. İkisini birlikte kaydır: oklar dönüyor ama aralarındaki açı, yani iç çarpım, değişmiyor.
        </p>
      </div>
    </div>
  );
}

/** Gerçek head_dim = 256 için 128 çiftin frekansı (senin GemmaRotaryEmbedding.inv_freq'in) */
export function RopeFreqs() {
  const { data, error } = useData<AttentionToy>("attention_toy");
  if (!data) return <Loading what="RoPE frekansları" error={error} />;
  const f = data.rope_real.inv_freq;
  const wl = (i: number) => (2 * Math.PI) / f[i];
  return (
    <div>
      <LineChart height={200} xLabel="çift numarası i (0–127)" yLabel="log₁₀ θᵢ" legend={false}
        series={[{ name: "θᵢ", color: "var(--ax-hd)", values: f.map((v) => Math.log10(v)) }]} />
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label="çift 0: θ" value={f[0].toPrecision(3)} sub={`bir tur ≈ ${wl(0).toFixed(1)} token`} />
        <Stat label="çift 64: θ" value={f[64].toPrecision(3)} sub={`bir tur ≈ ${Math.round(wl(64))} token`} />
        <Stat label="çift 127: θ" value={f[127].toExponential(2)} sub={`bir tur ≈ ${Math.round(wl(127)).toLocaleString("tr-TR")} token`} />
      </div>
    </div>
  );
}

/** Eşit uzunlukta q ve k için skor, yalnızca konum farkı Δ = m − n'nin fonksiyonu: Σ cos(Δ θᵢ) */
export function RopeScoreCurve() {
  const { data, error } = useData<AttentionToy>("attention_toy");
  const [pairs, setPairs] = useState<"hepsi" | "hızlı" | "yavaş">("hepsi");
  const series = useMemo(() => {
    if (!data) return [];
    const f = data.rope_real.inv_freq;
    const sel = pairs === "hepsi" ? f : pairs === "hızlı" ? f.slice(0, 32) : f.slice(96);
    const xs = Array.from({ length: 129 }, (_, d) => d);
    return [{ name: `Σ cos(Δ·θᵢ), ${sel.length} çift`, color: "var(--ax-T)", x: xs, values: xs.map((d) => sel.reduce((s, t) => s + Math.cos(d * t), 0) / sel.length) }];
  }, [data, pairs]);
  if (!data) return <Loading what="RoPE frekansları" error={error} />;
  const r = data.rope_relative;
  return (
    <div>
      <div style={{ marginBottom: 8 }}><Seg value={pairs} onChange={setPairs} options={[["hepsi", "128 çiftin hepsi"], ["hızlı", "ilk 32 çift (hızlı)"], ["yavaş", "son 32 çift (yavaş)"]]} /></div>
      <LineChart height={220} xLabel="konum farkı Δ = m − n" yLabel="ortalama skor" yDomain={[-1, 1]} series={series} legend={false} />
      <p className="small muted" style={{ margin: "8px 0 10px" }}>
        Her çiftte q ve k aynı yönde başlıyor (soyut bir en iyi durum). Hızlı çiftler yakın konumları ayırt ediyor ve hemen sönüyor. Yavaş çiftler uzaktaki konumları ayırt ediyor.
      </p>
      <div className="row" style={{ gap: 10 }}>
        <Stat label="senin kodunla, Δ = 5, dört farklı (m, n)" value={r.scores_diff5.map((v) => v.toFixed(5)).join(" · ")} sub={`(5,0) (15,10) (305,300) (3005,3000): en büyük fark ${r.spread.toExponential(1)}`} tone="ok" />
        <Stat label="Δ = 6" value={r.score_diff6.toFixed(4)} sub="fark değişince skor değişiyor" />
        <Stat label="Δ = −5" value={r.score_diff_neg5.toFixed(4)} sub="yön de önemli" />
      </div>
    </div>
  );
}

/** Boyut düzeni: çiftler (i, i + d/2) mi, (2i, 2i+1) mi? */
export function PairLayout() {
  const { data } = useData<AttentionToy>("attention_toy");
  const [layout, setLayout] = useState<"half" | "adjacent">("half");
  const [hover, setHover] = useState<number | null>(null);
  const d = 16;
  const pairOf = (dim: number) => (layout === "half" ? dim % (d / 2) : Math.floor(dim / 2));
  const partner = (dim: number) => (layout === "half" ? (dim + d / 2) % d : dim ^ 1);
  const hp = hover === null ? null : pairOf(hover);
  return (
    <div>
      <div style={{ marginBottom: 10 }}>
        <Seg value={layout} onChange={setLayout} options={[["half", "ilk yarı / ikinci yarı (rotate_half, Gemma)"], ["adjacent", "komşu çiftler (makaledeki yazım)"]]} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${d}, 1fr)`, gap: 4 }}>
        {Array.from({ length: d }, (_, dim) => {
          const on = hp === pairOf(dim);
          return (
            <div key={dim} onMouseEnter={() => setHover(dim)} onMouseLeave={() => setHover(null)}
              style={{ textAlign: "center", padding: "10px 0", borderRadius: 6, fontFamily: "var(--mono)", fontSize: 12, color: "#fff", background: PAIR_COLORS[pairOf(dim) % 8], opacity: hp === null || on ? 1 : 0.3, transition: "opacity .15s, background .3s", cursor: "default" }}>
              {dim}
            </div>
          );
        })}
      </div>
      <div className="small muted" style={{ margin: "8px 0 10px", minHeight: 20 }}>
        {hover === null ? "Bir boyutun üstüne gel: birlikte dönen eşini göreceksin." : <>Boyut <b className="mono">{hover}</b> ile boyut <b className="mono">{partner(hover)}</b> aynı 2B düzlemde birlikte dönüyor (çift {pairOf(hover)}).</>}
      </div>
      {data && (
        <div className="row" style={{ gap: 10 }}>
          <Stat label="aynı q, iki düzen, ham çıktı farkı" value={data.rope_layout.half_vs_adjacent_maxdiff.toFixed(3)} sub="farklı çiftler döndüğü için" tone="bad" />
          <Stat label="boyutları yeniden sıralayınca" value={data.rope_layout.half_vs_adjacent_permuted_maxdiff.toExponential(1)} sub="aynı döndürme, sadece etiketler farklı" tone="ok" />
        </div>
      )}
    </div>
  );
}

/** bf16'da konum: büyük konumlar tam temsil edilemiyor, açı kayıyor */
export function Bf16Angle() {
  const [pos, setPos] = useState(4097);
  const b = toBf16(pos);
  return (
    <div>
      <label className="small">konum <b className="mono">{pos}</b>
        <input aria-label="konum" type="range" min={1} max={8191} value={pos} onChange={(e) => setPos(+e.target.value)} style={{ width: "100%" }} /></label>
      <div className="row" style={{ gap: 10, marginTop: 8 }}>
        <Stat label="float32'de konum" value={pos} tone="ok" />
        <Stat label="bf16'da konum" value={b} tone={b === pos ? "ok" : "bad"} sub={b === pos ? "tam" : `${b - pos > 0 ? "+" : ""}${b - pos} kayma`} />
        <Stat label="çift 0'da açı hatası" value={`${Math.abs(b - pos).toFixed(0)} rad`} sub="θ₀ = 1 rad/konum, yani açı konum kadar kayıyor" tone={b === pos ? undefined : "bad"} />
      </div>
    </div>
  );
}
