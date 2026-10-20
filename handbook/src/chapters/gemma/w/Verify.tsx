// 5.8 doğrulama: anahtar eşleştirici, check çıktıları, akış şeması
import { useMemo, useState } from "react";
import { useData } from "../../../lib/data";
import { Loading, Seg, Stat } from "../../../components/ui";
import type { ModelTree } from "./types";

const RENAMES: [string, string, string][] = [
  ["yok", "", ""],
  ["mlp → feed_forward", ".mlp.", ".feed_forward."],
  ["input_layernorm → ln1", ".input_layernorm.", ".ln1."],
  ["embed_tokens → wte", ".embed_tokens.", ".wte."],
  ["layers → blocks", ".layers.", ".blocks."],
  ["gate_proj → w1", ".gate_proj.", ".w1."],
];

export function KeyMatcher() {
  const { data, error } = useData<ModelTree>("model_tree");
  const [ren, setRen] = useState(0);
  const [scope, setScope] = useState<"layer0" | "all">("layer0");
  const res = useMemo(() => {
    if (!data) return null;
    const [, from, to] = RENAMES[ren];
    const ours = data.keys_ours.filter((k) => k.startsWith("language_model")).map((k) => (from ? k.split(from).join(to) : k));
    const ckpt = data.keys_ckpt.filter((k) => k.startsWith("language_model"));
    const so = new Set(ours), sc = new Set(ckpt);
    const filt = (k: string) => scope === "all" || !/layers\.\d+|blocks\.\d+/.test(k) || /(layers|blocks)\.0\./.test(k);
    const all = [...new Set([...ours, ...ckpt])].filter(filt).sort();
    return { all, so, sc, onlyOurs: ours.filter((k) => !sc.has(k)).length, onlyCkpt: ckpt.filter((k) => !so.has(k)).length };
  }, [data, ren, scope]);
  if (!data || !res) return <Loading what="anahtarlar" error={error} />;
  return (
    <div>
      <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 10 }}>
        <span className="small">bir modülü yeniden adlandır:</span>
        <select value={ren} onChange={(e) => setRen(+e.target.value)} style={{ padding: "5px 8px", borderRadius: 8, border: "1px solid var(--border-strong)", background: "var(--bg-elev)" }}>
          {RENAMES.map(([l], i) => <option key={i} value={i}>{l}</option>)}
        </select>
        <Seg value={scope} onChange={setScope} options={[["layer0", "katman 0 + diğerleri"], ["all", "hepsi"]]} />
      </div>
      <div className="row" style={{ gap: 10, marginBottom: 10 }}>
        <Stat label="sadece senin modelinde" value={res.onlyOurs} tone={res.onlyOurs > 1 ? "bad" : "warn"} sub="checkpoint'te karşılığı yok" />
        <Stat label="sadece checkpoint'te" value={res.onlyCkpt} tone={res.onlyCkpt > 72 ? "bad" : "warn"} sub="yüklenemeyecek ağırlık" />
      </div>
      <div style={{ maxHeight: 340, overflowY: "auto", fontFamily: "var(--mono)", fontSize: 12, border: "1px solid var(--border)", borderRadius: 8 }}>
        {res.all.map((k) => {
          const a = res.so.has(k), b = res.sc.has(k);
          const tone = a && b ? "var(--ok)" : k.includes("self_attn") ? "var(--fg-faint)" : k.includes("lm_head") ? "var(--warn)" : "var(--bad)";
          return (
            <div key={k} style={{ display: "grid", gridTemplateColumns: "24px 24px 1fr", gap: 6, padding: "3px 10px", borderBottom: "1px solid var(--border)", color: tone }}>
              <span title="senin modelin">{a ? "●" : "○"}</span><span title="checkpoint">{b ? "●" : "○"}</span><span style={{ wordBreak: "break-all" }}>{k}</span>
            </div>
          );
        })}
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>sütunlar: senin modelin · checkpoint. Gri: attention (aşama 5'te gelecek). Turuncu: lm_head (tying, 5.7). Kırmızı: gerçek uyumsuzluk.</div>
    </div>
  );
}

export function CheckOutput({ name = "check_gemma.py" }: { name?: string }) {
  const { data, error } = useData<any>("checks");
  if (!data) return <Loading what="check çıktıları" error={error} />;
  const r = data.runs.find((x: any) => x.name === name);
  if (!r) return null;
  return (
    <div>
      <div className="row small muted" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <span className="mono">$ .venv/bin/python checks/{name}</span>
        <span>{data.ran_at} · {r.seconds} s · çıkış kodu <b style={{ color: r.returncode ? "var(--bad)" : "var(--ok)" }}>{r.returncode}</b></span>
      </div>
      <pre className="mono" style={{ margin: 0, padding: 12, background: "var(--code-bg)", borderRadius: 8, fontSize: 12, whiteSpace: "pre-wrap", lineHeight: 1.55 }}>
        {r.output.split("\n").map((l: string, i: number) => (
          <div key={i} style={{ color: l.startsWith("[OK") ? "var(--ok)" : l.startsWith("[HATA") ? "var(--bad)" : undefined }}>{l}</div>
        ))}
      </pre>
    </div>
  );
}

export function AllChecks() {
  const { data, error } = useData<any>("checks");
  if (!data) return <Loading what="check çıktıları" error={error} />;
  return (
    <div className="row" style={{ gap: 8 }}>
      {data.runs.map((r: any) => (
        <span key={r.name} className={"pill " + (r.returncode ? "bad" : "ok")}>{r.returncode ? "✗" : "✓"} {r.name}</span>
      ))}
    </div>
  );
}

const PIPE = [
  ["1", "HF modülünü kur", "küçük config, rastgele ağırlık", "HFGemmaMLP(hf_config)"],
  ["2", "Rastgeleleştir", "norm'lar sıfırsa testi kör eder", "randomize_norms / .normal_()"],
  ["3", "Stub koy", "attention'ı iki tarafta aynı modülle değiştir", "StubAttention"],
  ["4", "Anahtarları karşılaştır", "state_dict kümeleri eşit mi?", "copy_weights"],
  ["5", "Ağırlıkları kopyala", "HF → senin modülün", "load_state_dict"],
  ["6", "Aynı girdi", "×40: gerçek akış ölçeği", "torch.randn(...) * 40"],
  ["7", "Karşılaştır", "şekil + dtype + allclose / equal", "compare"],
];

export function CheckFlow({ at }: { at?: number } = {}) {
  const [i0, setI] = useState(0);
  const i = at ?? i0;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 8 }}>
      {PIPE.map(([n, t, s, code], k) => (
        <button key={n} onClick={() => setI(k)} style={{ all: "unset", cursor: "pointer", padding: 10, borderRadius: 8, border: `1.5px solid ${k === i ? "var(--accent)" : "var(--border)"}`, background: k === i ? "var(--accent-soft)" : "var(--bg-elev)" }}>
          <div className="small muted">{n}</div>
          <div style={{ fontWeight: 600, fontSize: 14 }}>{t}</div>
          <div className="small muted" style={{ marginTop: 2 }}>{s}</div>
          <div className="mono" style={{ fontSize: 11, marginTop: 6, color: "var(--ax-I)" }}>{code}</div>
        </button>
      ))}
    </div>
  );
}
