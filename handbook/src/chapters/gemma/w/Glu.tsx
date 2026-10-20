// 5.4 GeGLU MLP: 3D matrisler, aktivasyon eğrileri, kapı ısı haritaları, parametre hesabı
import { useMemo, useState } from "react";
import { useData, rowOf } from "../../../lib/data";
import { fmtCount, geluErf, geluTanh, relu, silu } from "../../../lib/tensor";
import { Heatmap, LineChart, Bars } from "../../../components/charts";
import { Label, Stage, TBox, cssVar, vis } from "../../../components/three/core";
import { Loading, Seg, Stat, S } from "../../../components/ui";
import type { GemmaToy } from "./types";


export function GateMaps() {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [tok, setTok] = useState(0);
  if (!data) return <Loading what="GeGLU değerleri" error={error} />;
  const g = data.geglu;
  const [T, K] = g.act.shape;
  const row = (a: typeof g.act) => Array.from(rowOf(a, tok));
  const act = row(g.act), up = row(g.up), prod = row(g.prod);
  const dims = Array.from({ length: 12 }, (_, i) => i);
  // export ile aynı tanım: |gelu(gate)| < en büyük |gelu(gate)|'nin %5'i
  let amax = 0;
  for (const v of g.act.data) amax = Math.max(amax, Math.abs(v));
  const thr = 0.05 * amax;
  return (
    <div>
      <div className="grid2">
        {([["gelu_tanh(gate)", g.act, false, "var(--ax-hd)"], ["up", g.up, true, "var(--ax-D)"], ["gelu(gate) ⊙ up", g.prod, true, "var(--ax-I)"], ["gate (aktivasyondan önce)", g.gate, true, "var(--fg-faint)"]] as const).map(([name, arr, div]) => (
          <div key={name}>
            <div className="small"><b>{name}</b> <span className="muted">· {T} token × ilk {K} ara boyut</span></div>
            <Heatmap values={arr.data} rows={T} cols={K} height={130} diverging={div || name.startsWith("gelu_tanh")} onCell={(r) => setTok(r)} highlight={{ r: tok }} legend />
          </div>
        ))}
      </div>
      <div style={{ marginTop: 14 }}>
        <div className="small" style={{ marginBottom: 6 }}><b>Token {tok}</b>, ilk 12 ara boyut: kapı, değer, sonuç</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 4, fontFamily: "var(--mono)", fontSize: 11 }}>
          {dims.map((i) => {
            const closed = Math.abs(act[i]) < thr;
            return (
              <div key={i} style={{ textAlign: "center", padding: 4, borderRadius: 6, background: closed ? "var(--bg-sunk)" : "var(--accent-soft)" }}>
                <div style={{ color: "var(--ax-hd)" }}>{act[i].toFixed(2)}</div>
                <div className="muted">×</div>
                <div style={{ color: "var(--ax-D)" }}>{up[i].toFixed(2)}</div>
                <div className="muted">=</div>
                <div style={{ color: "var(--ax-I)", fontWeight: 700 }}>{prod[i].toFixed(2)}</div>
                <div style={{ fontSize: 9.5 }} className="muted">{closed ? "kapalı" : "açık"}</div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <Stat label="kapalı kapılar (|gelu| < %5 max)" value={`${(g.frac_closed * 100).toFixed(0)}%`} sub="oyuncak model, katman 0" />
        <Stat label="negatif aktivasyon" value={`${(g.frac_negative_act * 100).toFixed(0)}%`} sub="GELU'nun küçük negatif kuyruğu" />
        <Stat label="token bağımsızlığı: max fark" value={g.token_independence_maxdiff.toExponential(1)} sub="mlp(tek token) ile mlp(dizi) arası" tone="ok" />
      </div>
    </div>
  );
}

export function MlpParams() {
  const [d, setD] = useState(2048);
  const [I, setI] = useState(16384);
  const classic = 2 * d * 4 * d;
  const glu = 3 * d * I;
  const eq = Math.round((8 * d) / 3);
  return (
    <div>
      <div className="grid2">
        <label className="small">d (hidden_size) <b className="mono">{d}</b><input type="range" min={256} max={4096} step={256} value={d} onChange={(e) => setD(+e.target.value)} style={{ width: "100%" }} /></label>
        <label className="small">I (intermediate_size) <b className="mono">{I}</b><input type="range" min={1024} max={32768} step={256} value={I} onChange={(e) => setI(+e.target.value)} style={{ width: "100%" }} /></label>
      </div>
      <div className="row" style={{ gap: 8, margin: "8px 0 12px" }}>
        <button className="btn" onClick={() => { setD(2048); setI(16384); }}>Gemma 2B (I = 8d)</button>
        <button className="btn" onClick={() => { setD(2304); setI(9216); }}>Gemma 2 2B (I = 4d)</button>
        <button className="btn" onClick={() => setI(eq)}>eşit parametre (I = 8d/3)</button>
      </div>
      <Bars items={[
        { label: "klasik MLP: 2 · d · 4d", value: classic, color: "var(--fg-faint)" },
        { label: "GeGLU: 3 · d · I", value: glu, color: "var(--accent)", note: <span className="muted small">({(glu / classic).toFixed(2)}×)</span> },
      ]} fmt={fmtCount} />
      <p className="small muted" style={{ margin: "10px 0 0" }}>
        Klasik MLP'de ara boyut 4d ve 2 matris var. GLU'da 3 matris olduğu için "aynı bütçe" için I = 8d/3 ≈ {eq} seçilir (Llama böyle yapar).
        Gemma 1 bu kuralı uygulamıyor: I = 8d, yani klasik bir MLP'nin <b>3 katı</b> parametre. Gemma 2 2B'de I = 4d'ye dönülmüş.
      </p>
    </div>
  );
}
