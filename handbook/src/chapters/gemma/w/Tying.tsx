// 5.7 weight tying: tek levha iki rol, Python nesne şeması, bellek ve hassasiyet
import { useState } from "react";
import { useData } from "../../../lib/data";
import { bf16Ulp } from "../../../lib/bf16";
import { fmtCount } from "../../../lib/tensor";
import { Label, Stage, TBox, cssVar } from "../../../components/three/core";
import { Loading, Seg, Stat, S } from "../../../components/ui";
import type { ModelTree, PgSlices } from "./types";

export function ObjectDiagram() {
  const [st, setSt] = useState<"before" | "after" | "load" | "break">("before");
  const tied = st !== "before" && st !== "break";
  const box = (x: number, y: number, label: string, sub: string, color: string, dim = false) => (
    <g opacity={dim ? 0.35 : 1}>
      <rect x={x} y={y} width={190} height={50} rx={8} fill={`color-mix(in srgb, ${color} 12%, var(--bg-elev))`} stroke={color} strokeWidth={1.5} />
      <text x={x + 12} y={y + 21} fontSize={12.5} fontFamily="var(--mono)" fill="var(--fg)">{label}</text>
      <text x={x + 12} y={y + 38} fontSize={11} fill="var(--fg-muted)">{sub}</text>
    </g>
  );
  const name = (y: number, label: string) => <text x={10} y={y} fontSize={12.5} fontFamily="var(--mono)" fill="var(--fg)">{label}</text>;
  return (
    <div>
      <Seg value={st} onChange={setSt} options={[["before", "1. __init__ sonrası"], ["after", "2. tie_weights()"], ["load", "3. load_state_dict"], ["break", "✗ yanlış: .data kopyası"]]} />
      <svg width="100%" viewBox="0 0 560 170" style={{ marginTop: 10, fontFamily: "var(--sans)", maxWidth: 620 }}>
        <defs><marker id="ar2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg-muted)" /></marker></defs>
        {name(50, "model.embed_tokens.weight")}
        {name(130, "lm_head.weight")}
        <line x1={200} y1={46} x2={318} y2={tied ? 76 : 46} stroke="var(--fg-muted)" strokeWidth={1.5} markerEnd="url(#ar2)" />
        <line x1={120} y1={126} x2={318} y2={tied ? 96 : 126} stroke={tied ? "var(--accent)" : "var(--fg-muted)"} strokeWidth={tied ? 2.5 : 1.5} markerEnd="url(#ar2)" />
        {tied ? (
          box(320, 60, "Parameter #A", st === "load" ? "[257216, 2048] · değerler checkpoint'ten" : "[257216, 2048] · iki ad, tek nesne", "var(--ax-V)")
        ) : (
          <>
            {box(320, 22, "Parameter #A", "[257216, 2048] · embedding", "var(--ax-V)")}
            {box(320, 102, "Parameter #B", st === "break" ? "[257216, 2048] · A'nın KOPYASI" : "[257216, 2048] · rastgele, ayrı", st === "break" ? "var(--bad)" : "var(--ax-D)")}
          </>
        )}
      </svg>
      <div className="small" style={{ marginTop: 4 }}>
        {st === "before" && <>İki ayrı <code>nn.Parameter</code>: 2 × 527 M = 1.05 B parametre. lm_head rastgele başlatılmış.</>}
        {st === "after" && <><code>self.lm_head.weight = self.model.embed_tokens.weight</code>: Python'da atama <b>nesneyi</b> bağlar, kopyalamaz. Artık iki ad aynı nesneyi gösteriyor ve #B çöpe gidiyor. <code>lm_head.weight is embed_tokens.weight</code> → True.</>}
        {st === "load" && <><code>load_state_dict</code> nesneleri değiştirmez, mevcut tensörlerin <b>içine</b> değer kopyalar (<code>param.copy_(…)</code>). Bağ korunuyor. Senin oyuncak deneyinde: sonra embedding'e yazılan değer lm_head'de de görünüyor.</>}
        {st === "break" && <><code>lm_head.weight.data = embed_tokens.weight.data.clone()</code> ya da <code>nn.Parameter(w.clone())</code>: değerler aynı ama <b>nesneler farklı</b>. Bellek iki katı, eğitimde ikisi ayrı ayrı güncellenip birbirinden uzaklaşır.</>}
      </div>
    </div>
  );
}

export function TyingFacts() {
  const t = useData<ModelTree>("model_tree"), s = useData<PgSlices>("pg_slices");
  if (!t.data || !s.data) return <Loading what="checkpoint" error={t.error ?? s.error} />;
  const lm = t.data.keys_only_ours.filter((k) => k.includes("lm_head"));
  return (
    <div className="row" style={{ gap: 10 }}>
      <Stat label="checkpoint'te lm_head anahtarı" value={s.data.has_lm_head_key ? "var" : "YOK"} tone={s.data.has_lm_head_key ? undefined : "warn"} sub={`${s.data.num_keys} anahtar arasında`} />
      <Stat label="senin state_dict'inde fazladan" value={lm.length} sub={lm.join(", ")} tone="warn" />
      <Stat label="tekil parametre" value={fmtCount(t.data.params_unique)} sub={`state_dict toplamı ${fmtCount(t.data.params_state_dict)}`} />
    </div>
  );
}

export function LogitMemory() {
  const [T, setT] = useState(260);
  const V = 257216;
  const mb = (bytes: number) => (bytes / 2 ** 20).toFixed(0) + " MB";
  return (
    <div>
      <label className="small" style={{ display: "flex", gap: 10, alignItems: "center" }}>dizi uzunluğu T
        <input type="range" min={1} max={4200} value={T} onChange={(e) => setT(+e.target.value)} style={{ flex: 1 }} />
        <b className="mono">{T}</b>
        <button className="btn" onClick={() => setT(260)}>224 px</button>
        <button className="btn" onClick={() => setT(1028)}>448 px</button>
        <button className="btn" onClick={() => setT(4100)}>896 px</button>
      </label>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label={<>logits <S s={`1, ${T}, ${V}`} ax="B,T,V" /> bf16</>} value={mb(T * V * 2)} />
        <Stat label="float32 (.float())" value={mb(T * V * 4)} tone={T * V * 4 > 2 ** 30 ? "bad" : undefined} />
        <Stat label="sadece son token (üretimde yeterli)" value={mb(V * 4)} tone="ok" />
      </div>
    </div>
  );
}

export function LogitPrecision() {
  const [z, setZ] = useState(20);
  const u = bf16Ulp(z);
  return (
    <div>
      <label className="small" style={{ display: "flex", gap: 10, alignItems: "center" }}>logit değeri
        <input type="range" min={1} max={60} step={0.5} value={z} onChange={(e) => setZ(+e.target.value)} style={{ flex: 1 }} /><b className="mono">{z}</b>
      </label>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label="bf16'da bu civarda adım (1 ulp)" value={u} />
        <Stat label="olasılık oranına etkisi: e^ulp" value={`×${Math.exp(u).toFixed(3)}`} sub={`bir adımlık yuvarlama, olasılığı %${((Math.exp(u) - 1) * 100).toFixed(1)} kaydırır`} tone={u >= 0.125 ? "bad" : undefined} />
      </div>
    </div>
  );
}
