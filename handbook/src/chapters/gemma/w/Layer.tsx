// 5.5 ve 5.6: katman akış şeması, Gemma 1/2 karşılaştırması, konteynerler, akış büyümesi
import { useState } from "react";
import { useData } from "../../../lib/data";
import { LineChart } from "../../../components/charts";
import { Loading, Seg, Stat, S } from "../../../components/ui";
import type { GemmaToy } from "./types";

const FLOW = [
  { id: "in", label: "hidden_states", sub: "akış", y: 0 },
  { id: "res1", label: "residual = …", sub: "kopya sakla", y: 1 },
  { id: "n1", label: "input_layernorm", sub: "RMSNorm", y: 2 },
  { id: "attn", label: "self_attn", sub: "şimdilik Identity", y: 3 },
  { id: "add1", label: "residual + …", sub: "⊕ akışa yaz", y: 4 },
  { id: "res2", label: "residual = …", sub: "kopya sakla", y: 5 },
  { id: "n2", label: "post_attention_layernorm", sub: "RMSNorm (MLP girişi)", y: 6 },
  { id: "mlp", label: "mlp", sub: "GeGLU", y: 7 },
  { id: "add2", label: "residual + …", sub: "⊕ akışa yaz", y: 8 },
];

export function LayerFlow({ at }: { at?: number } = {}) {
  const [i0, setI] = useState(0);
  const i = at ?? i0;
  const W = 560, rowH = 46, x0 = 150;
  const color = (id: string) => id.startsWith("n") ? "var(--ax-hd)" : id === "attn" ? "var(--fg-faint)" : id === "mlp" ? "var(--ax-D)" : id.startsWith("add") ? "var(--accent)" : "var(--ax-T)";
  return (
    <div>
      <div className="row" style={{ gap: 8, marginBottom: 8, alignItems: "center" }}>
        <button className="btn" disabled={i <= 0} onClick={() => setI(i - 1)}>←</button>
        <button className="btn primary" disabled={i >= FLOW.length - 1} onClick={() => setI(i + 1)}>Sonraki satır →</button>
        <button className="btn" onClick={() => setI(0)}>⟲</button>
        <span className="small muted">her adımda şekil: <S s="B, T, 2048" ax="B,T,D" /></span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg width={W} height={FLOW.length * rowH + 10} style={{ display: "block", margin: "0 auto", fontFamily: "var(--sans)" }}>
          {/* akış hattı (sol) */}
          <line x1={40} x2={40} y1={20} y2={FLOW.length * rowH - 10} stroke="var(--ax-T)" strokeWidth={8} opacity={0.25} strokeLinecap="round" />
          <text x={40} y={14} textAnchor="middle" fontSize={11} fill="var(--ax-T)">akış</text>
          {/* residual atlama kavisleri */}
          {[[1, 4], [5, 8]].map(([a, b]) => (
            <path key={a} d={`M 40 ${a * rowH + 25} C 0 ${a * rowH + 25}, 0 ${b * rowH + 25}, 40 ${b * rowH + 25}`} fill="none"
              stroke={i >= a && i <= b ? "var(--accent)" : "var(--border-strong)"} strokeWidth={2} strokeDasharray="5 4" />
          ))}
          {FLOW.map((f, k) => {
            const on = k === i, done = k < i;
            return (
              <g key={f.id} opacity={done || on ? 1 : 0.35}>
                <line x1={40} x2={x0} y1={k * rowH + 25} y2={k * rowH + 25} stroke="var(--border)" />
                <rect x={x0} y={k * rowH + 8} width={250} height={34} rx={8} fill={on ? `color-mix(in srgb, ${color(f.id)} 22%, var(--bg-elev))` : "var(--bg-elev)"} stroke={color(f.id)} strokeWidth={on ? 2.5 : 1.2} />
                <text x={x0 + 12} y={k * rowH + 24} fontSize={13} fontWeight={600} fill="var(--fg)" fontFamily="var(--mono)">{f.label}</text>
                <text x={x0 + 12} y={k * rowH + 37} fontSize={10.5} fill="var(--fg-muted)">{f.sub}</text>
                {k < FLOW.length - 1 && <line x1={x0 + 125} x2={x0 + 125} y1={k * rowH + 42} y2={k * rowH + 54} stroke="var(--fg-faint)" markerEnd="url(#arr)" />}
                {on && <text x={x0 + 262} y={k * rowH + 30} fontSize={12} fill="var(--accent)">◀ şimdi</text>}
              </g>
            );
          })}
          <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="var(--fg-faint)" /></marker></defs>
        </svg>
      </div>
    </div>
  );
}

export function SandwichCompare() {
  const [v, setV] = useState<"g1" | "g2">("g1");
  const box = (label: string, color: string, dashed = false) => (
    <div style={{ border: `1.5px ${dashed ? "dashed" : "solid"} ${color}`, borderRadius: 8, padding: "6px 10px", fontFamily: "var(--mono)", fontSize: 12.5, background: `color-mix(in srgb, ${color} 10%, var(--bg-elev))`, textAlign: "center" }}>{label}</div>
  );
  const arrow = <div style={{ textAlign: "center", color: "var(--fg-faint)", lineHeight: 1 }}>↓</div>;
  const norm = "var(--ax-hd)";
  return (
    <div>
      <Seg value={v} onChange={setV} options={[["g1", "Gemma 1 (senin kodun)"], ["g2", "Gemma 2 (aşama 8)"]]} />
      <div className="grid2" style={{ marginTop: 12, gap: 24 }}>
        {(["attn", "mlp"] as const).map((b) => (
          <div key={b} style={{ display: "grid", gap: 4 }}>
            <div className="small muted" style={{ textAlign: "center" }}>{b === "attn" ? "attention bloğu" : "MLP bloğu"}</div>
            {box(b === "attn" ? "input_layernorm" : v === "g1" ? "post_attention_layernorm" : "pre_feedforward_layernorm", norm)}
            {arrow}
            {box(b === "attn" ? "self_attn" : "mlp", b === "attn" ? "var(--fg-faint)" : "var(--ax-D)")}
            {arrow}
            {v === "g2" && <>{box(b === "attn" ? "post_attention_layernorm" : "post_feedforward_layernorm", norm, true)}{arrow}</>}
            {box("⊕ residual", "var(--accent)")}
          </div>
        ))}
      </div>
      <p className="small muted" style={{ margin: "10px 0 0" }}>
        {v === "g1"
          ? <>Gemma 1'de katman başına 2 norm. <code>post_attention_layernorm</code> attention'dan sonra geliyor ama <b>MLP'nin girişini</b> normalize ediyor.</>
          : <>Gemma 2'de 4 norm ("sandwich"). Aynı ad (<code>post_attention_layernorm</code>) artık <b>attention'ın çıktısını</b> normalize ediyor. MLP'nin girişi <code>pre_feedforward_layernorm</code> oldu. Checkpoint'i yüklerken bu adları karıştırmak, hatası en zor bulunan şey olur.</>}
      </p>
    </div>
  );
}

export function NormSwapStats({ broken }: { broken: boolean }) {
  const { data } = useData<GemmaToy>("gemma_toy");
  if (!data || !broken) return null;
  return (
    <div className="row" style={{ gap: 10 }}>
      <Stat label="norm ağırlıkları sıfırken: fark" value={data.norm_swap.zero_w.toFixed(3)} tone="warn" sub="iki norm aynı davranıyor, hata görünmez" />
      <Stat label="gerçek ağırlıklarla: fark" value={data.norm_swap.real_w.toFixed(3)} tone="bad" sub="oyuncak model, katman 0" />
    </div>
  );
}

export function Containers() {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  if (!data) return <Loading what="deney" error={error} />;
  const c = data.containers;
  const rows: [string, string, string, string][] = [
    ["WithModuleList", "nn.ModuleList([...])", "kaydediyor", "for döngüsü, istediğin argümanla"],
    ["WithList", "[...] (düz Python listesi)", "KAYDETMİYOR", "for döngüsü"],
    ["WithSequential", "nn.Sequential(*[...])", "kaydediyor", "self.layers(x): tek argüman"],
  ];
  return (
    <table>
      <thead><tr><th>konteyner</th><th>parametreler</th><th>state_dict anahtarları</th><th>forward</th></tr></thead>
      <tbody>
        {rows.map(([k, code, reg, fwd]) => (
          <tr key={k}>
            <td className="mono" style={{ fontSize: 12.5 }}>{code}</td>
            <td><span className={"pill " + (c[k].n_params ? "ok" : "bad")}>{reg} · {c[k].n_params}</span></td>
            <td className="mono" style={{ fontSize: 12 }}>{c[k].keys.length ? c[k].keys.join(", ") : <span style={{ color: "var(--bad)" }}>(boş)</span>}</td>
            <td className="small">{fwd}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SiglipGrowth() {
  const { data, error } = useData<any>("pipeline");
  if (!data) return <Loading what="SigLIP gizli durumları" error={error} />;
  const im = data.images[0];
  const mean = im.hidden_norms.map((r: number[]) => r.reduce((a, b) => a + b, 0) / r.length);
  return (
    <div>
      <LineChart height={220} xLabel="katman (0 = embedding)" yLabel="‖h‖ ve max |h|"
        series={[
          { name: "ortalama ‖h‖ (token başına)", color: "var(--ax-T)", values: mean },
          { name: "max |h| (tek eleman)", color: "var(--bad)", values: im.hidden_absmax, dash: true },
        ]} />
      <div className="row" style={{ gap: 10, marginTop: 8 }}>
        <Stat label="post_layernorm sonrası ‖h‖" value={im.post_ln_norm_mean.toFixed(1)} sub="tek normalizasyon, en sonda" tone="ok" />
      </div>
    </div>
  );
}
