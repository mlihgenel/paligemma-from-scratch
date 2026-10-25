// 4.10 MQA / GQA: which Q head uses which KV head, KV cache size, repeat_kv order and memory
import { useState } from "react";
import { useData } from "../../../lib/data";
import { Bars } from "../../../components/charts";
import { Loading, Seg, Stat } from "../../../components/ui";
import type { AttentionToy } from "./types";
import { useT } from "../../../i18n";

const KV_COLORS = ["#d95f4b", "#3d85c6", "#6aa84f", "#e0a030", "#b05fc0", "#2f9e8f", "#c05f8a", "#6f6fd8"];
const PRESETS: [number, string][] = [[8, "MHA · 8 KV"], [4, "GQA · 4 KV (Gemma 2 2B)"], [2, "GQA · 2 KV"], [1, "MQA · 1 KV (Gemma 2B)"]];

/** Which KV head each of the 8 Q heads is connected to. Hover highlights the heads in the same group. */
export function HeadMap() {
  const tr = useT();
  const [kv, setKv] = useState(1);
  const [hover, setHover] = useState<number | null>(null);
  const H = 8;
  const rep = H / kv;
  const group = (h: number) => Math.floor(h / rep);
  const W = 560, xq = (h: number) => 40 + h * ((W - 80) / (H - 1)), xk = (g: number) => (kv === 1 ? W / 2 : 40 + g * ((W - 80) / (kv - 1)));
  const hg = hover === null ? null : group(hover);
  return (
    <div>
      <div style={{ marginBottom: 8 }}><Seg value={kv} onChange={setKv} options={PRESETS} /></div>
      <svg viewBox={`0 0 ${W} 200`} style={{ width: "100%", maxWidth: 640, margin: "0 auto", display: "block" }} data-testid="head-map">
        {Array.from({ length: H }, (_, h) => (
          <line key={h} x1={xq(h)} y1={56} x2={xk(group(h))} y2={144} stroke={KV_COLORS[group(h) % 8]} strokeWidth={hg === null || hg === group(h) ? 2.5 : 1} opacity={hg === null || hg === group(h) ? 0.9 : 0.2} style={{ transition: "all .3s" }} />
        ))}
        {Array.from({ length: H }, (_, h) => (
          <g key={h} onMouseEnter={() => setHover(h)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
            <rect x={xq(h) - 20} y={22} width={40} height={34} rx={6} fill={KV_COLORS[group(h) % 8]} opacity={hg === null || hg === group(h) ? 1 : 0.3} style={{ transition: "all .3s" }} />
            <text x={xq(h)} y={44} textAnchor="middle" fontSize="13" fontWeight={700} fill="#fff">Q{h}</text>
          </g>
        ))}
        {Array.from({ length: kv }, (_, g) => (
          <g key={g}>
            <rect x={xk(g) - 22} y={144} width={44} height={34} rx={6} fill={KV_COLORS[g % 8]} opacity={hg === null || hg === g ? 1 : 0.3} style={{ transition: "all .3s" }} />
            <text x={xk(g)} y={166} textAnchor="middle" fontSize="13" fontWeight={700} fill="#fff">KV{g}</text>
          </g>
        ))}
        <text x={4} y={14} fontSize="11" fill="var(--fg-faint)">sorgu head'leri (q_proj: 8 × 256)</text>
        <text x={4} y={196} fontSize="11" fill="var(--fg-faint)">{tr("key/value heads", "anahtar/değer head'leri")} (k_proj, v_proj: {kv} × 256)</text>
      </svg>
      <div className="small" style={{ margin: "6px 0 0" }}>
        <span className="mono">n_rep = {H} / {kv} = <b>{rep}</b></span>: her KV head'i {rep} Q head'ine hizmet ediyor. Q head'i <span className="mono">h</span> → KV head'i <span className="mono">h // {rep}</span>.
        {hover !== null && <> {tr("Now", "Şu an")}: <b className="mono">Q{hover}</b> → <b className="mono">KV{group(hover)}</b>.</>}
      </div>
    </div>
  );
}

const MiB = 1024 * 1024;
const fmtBytes = (b: number) => (b >= 1024 * MiB ? `${(b / (1024 * MiB)).toFixed(2)} GiB` : `${(b / MiB).toFixed(1)} MiB`);

/** KV cache: per token per layer 2 (K and V) × kv_heads × head_dim × 2 bytes (bf16) */
export function KvCacheSize() {
  const tr = useT();
  const { data } = useData<AttentionToy>("attention_toy");
  const [T, setT] = useState(260);
  const layers = 18, hd = 256, bytes = 2;
  const size = (kv: number) => 2 * layers * kv * hd * bytes * T;
  const pr = data?.params_real;
  return (
    <div>
      <label className="small">{tr("sequence length T", "dizi uzunluğu T")} <b className="mono">{T}</b> token
        <input aria-label={tr("sequence length", "dizi uzunluğu")} type="range" min={64} max={8192} step={64} value={T} onChange={(e) => setT(+e.target.value)} style={{ width: "100%" }} /></label>
      <div className="row" style={{ gap: 8, margin: "6px 0 12px" }}>
        <button className="btn" onClick={() => setT(260)}>PaliGemma 224: 260</button>
        <button className="btn" onClick={() => setT(1024 + 4)}>448 piksel: 1028</button>
        <button className="btn" onClick={() => setT(8192)}>8192 ({tr("Gemma context limit", "Gemma bağlam sınırı")})</button>
      </div>
      <Bars max={size(8)} fmt={fmtBytes} items={[
        { label: "MHA · 8 KV", value: size(8), color: "var(--fg-faint)" },
        { label: "GQA · 4 KV", value: size(4), color: "var(--ax-I)" },
        { label: "GQA · 2 KV", value: size(2), color: "var(--ax-D)" },
        { label: "MQA · 1 KV (Gemma 2B)", value: size(1), color: "var(--accent)" },
      ]} />
      <p className="small muted" style={{ margin: "10px 0 8px" }}>
        {tr(<>18 layers × (K and V) × KV heads × 256 × 2 bytes (bf16) × T tokens. Only the <b>stored</b> K/V count, Q is not stored.</>, <>18 katman × (K ve V) × KV head sayısı × 256 × 2 bayt (bf16) × T token. Sadece <b>saklanan</b> K/V sayılıyor, Q saklanmıyor.</>)}
      </p>
      {pr && (
        <div className="row" style={{ gap: 10 }}>
          <Stat label={tr("k_proj + v_proj, MQA (one layer)", "k_proj + v_proj, MQA (bir katman)")} value={((pr.k + pr.v) / 1e6).toFixed(2) + " M"} sub="2 × 2048 × 256" tone="ok" />
          <Stat label={tr("k_proj + v_proj, if it were MHA", "k_proj + v_proj, MHA olsaydı")} value={((pr.q + pr.q) / 1e6).toFixed(2) + " M"} sub="2 × 2048 × 2048" />
          <Stat label={tr("q_proj and o_proj", "q_proj ve o_proj")} value={((pr.q + pr.o) / 1e6).toFixed(2) + " M"} sub={tr("both 2048 × 2048, unchanged by MQA", "her ikisi de 2048 × 2048, MQA'da değişmiyor")} />
        </div>
      )}
    </div>
  );
}

/** Where the new axis goes in repeat_kv: (n_kv, n_rep) is right, (n_rep, n_kv) is wrong */
export function OrderMap({ wrong }: { wrong: boolean }) {
  const tr = useT();
  const kv = 2, rep = 4, H = 8;
  const used = (h: number) => (wrong ? h % kv : Math.floor(h / rep));
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${H}, 1fr)`, gap: 4 }}>
        {Array.from({ length: H }, (_, h) => (
          <div key={h} style={{ textAlign: "center", borderRadius: 6, padding: "6px 0", color: "#fff", background: KV_COLORS[used(h)], transition: "background .3s" }}>
            <div className="mono" style={{ fontSize: 11, opacity: 0.85 }}>Q{h}</div>
            <div className="mono" style={{ fontWeight: 700 }}>KV{used(h)}</div>
          </div>
        ))}
      </div>
      <p className="small muted" style={{ margin: "8px 0 0" }}>
        {wrong
          ? tr(<>Merging the axes in the order <span className="mono">[n_rep, n_kv]</span> connects the Q heads to the KV heads <b>alternately</b> (0, 1, 0, 1, …). The weights were trained for the <span className="mono">h // n_rep</span> layout, so this mapping looks at the wrong K/V. The shape is right and nothing raises.</>, <>Eksenler <span className="mono">[n_rep, n_kv]</span> sırasıyla birleşince Q head'leri KV head'lerine <b>dönüşümlü</b> bağlanıyor (0, 1, 0, 1, …). Ağırlıklar <span className="mono">h // n_rep</span> düzenine göre eğitildi, bu eşleşme yanlış K/V'ye bakıyor. Şekil doğru, hata vermiyor.</>)
          : tr(<>Merging the axes in the order <span className="mono">[n_kv, n_rep]</span> makes neighboring Q heads share a KV head: Q0–Q3 → KV0, Q4–Q7 → KV1. This is HF's layout and the one the weights expect.</>, <>Eksenler <span className="mono">[n_kv, n_rep]</span> sırasıyla birleşince komşu Q head'leri aynı KV head'ini paylaşıyor: Q0–Q3 → KV0, Q4–Q7 → KV1. HF'nin ve ağırlıkların düzeni bu.</>)}
      </p>
    </div>
  );
}

/** Memory behavior of expand and reshape (measured with the project's repeat_kv and torch data_ptr) */
export function ExpandMemory() {
  const tr = useT();
  const { data, error } = useData<AttentionToy>("attention_toy");
  if (!data) return <Loading what={tr("repeat_kv measurement", "repeat_kv ölçümü")} error={error} />;
  const r1 = data.repeat_kv["1"], r2 = data.repeat_kv["2"];
  const yes = (b: boolean) => (b ? tr("yes", "evet") : tr("no", "hayır"));
  return (
    <div>
      <table style={{ width: "100%", fontSize: 14, borderCollapse: "collapse" }}>
        <thead>
          <tr className="muted small" style={{ textAlign: "left" }}><th></th><th>MQA (n_kv = 1)</th><th>GQA (n_kv = 2)</th></tr>
        </thead>
        <tbody>
          <tr><td>{tr("after expand, same memory as the original?", "expand sonrası, orijinal veriyle aynı bellek mi?")}</td><td><b>{yes(r1.expand_shares_memory)}</b></td><td><b>{yes(r2.expand_shares_memory)}</b></td></tr>
          <tr><td>{tr("stride of the expanded axis", "expand edilen eksenin adımı (stride)")}</td><td className="mono">{r1.expand_stride_rep}</td><td className="mono">{r2.expand_stride_rep}</td></tr>
          <tr><td>{tr("repeat_kv output (after reshape) same memory?", "repeat_kv çıktısı (reshape sonrası) aynı bellek mi?")}</td><td><b style={{ color: "var(--ok)" }}>{yes(r1.repeat_shares_memory)}</b> ({tr("view", "görünüm")})</td><td><b style={{ color: "var(--warn)" }}>{yes(r2.repeat_shares_memory)}</b> ({tr("copy", "kopya")})</td></tr>
          <tr><td>{tr("does every Q head see the right KV head?", "her Q head'i doğru KV head'ini görüyor mu?")}</td><td><b>{yes(r1.all_equal)}</b></td><td><b>{yes(r2.all_equal)}</b></td></tr>
        </tbody>
      </table>
    </div>
  );
}
