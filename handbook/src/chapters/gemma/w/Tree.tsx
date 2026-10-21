// Modül ağacı: senin sınıfların, gerçek config ile meta cihazında kurulmuş haliyle
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useData } from "../../../lib/data";
import { fmtCount } from "../../../lib/tensor";
import { Loading, S } from "../../../components/ui";
import type { ModelTree, TreeNode } from "./types";

const WHAT: Record<string, [string, string?]> = {
  GemmaForCausalLM: ["Dil modeli + lm_head. Gizli durumları sözlük üzerindeki logit'lere çevirir.", "/gemma/tying"],
  GemmaModel: ["Embedding → ×√d → 18 decoder katmanı → son RMSNorm.", "/gemma/model"],
  Embedding: ["Token id → tablo satırı (lookup). Hesap değil, indeksleme.", "/gemma/sqrt-d"],
  ModuleList: ["Alt modülleri kaydeden liste. Kendisi forward yapmaz.", "/gemma/model"],
  GemmaDecoderLayer: ["Pre-norm katman: x + attn(norm(x)), sonra x + mlp(norm(x)).", "/gemma/katman"],
  Identity: ["Şimdilik attention yerine: girdiyi aynen döndürür. Sonra yerine GemmaAttention gelecek.", "/gemma/katman"],
  GemmaMLP: ["GeGLU: down(gelu_tanh(gate(x)) ⊙ up(x)). Token'lar arası karışma yok.", "/gemma/geglu"],
  GemmaRMSNorm: ["x / rms(x) · (1 + w). Ortalama çıkarılmaz, bias yok, hesap float32'de.", "/gemma/rmsnorm"],
  Linear: ["y = x Wᵀ (+ b). Son eksen dönüşür, diğer eksenler aynen kalır."],
  PaliGemmaForConditionalGeneration: ["Üç parçayı birleştiren kök: vision_tower, multi_modal_projector, language_model.", "/kopru"],
  SiglipVisionModel: ["SigLIP sarmalayıcı: anahtarların vision_model.* olması için.", "/kopru"],
  SiglipVisionTransformer: ["Embedding → 27 encoder katmanı → post_layernorm."],
  SiglipVisionEmbeddings: ["Conv2d ile yamalar + öğrenilen konum embedding'i."],
  Conv2d: ["kernel = stride = 14: üst üste binmeyen 14×14 yamalar, her yamaya aynı linear dönüşüm."],
  SiglipEncoder: ["27 SiglipEncoderLayer'ı sırayla uygular."],
  SiglipEncoderLayer: ["Pre-norm: x + attn(LN(x)), x + mlp(LN(x)). Maske yok: her yama her yamayı görür."],
  SiglipAttention: ["16 head, head_dim 72. softmax(QKᵀ/√72)V."],
  SiglipMLP: ["fc1 → gelu_tanh → fc2. Klasik (kapısız) MLP."],
  LayerNorm: ["Ortalama çıkar, varyansa böl, γ ve β ile ölçekle. eps = 1e-6."],
  PaliGemmaMultiModalProjector: ["Tek Linear: 1152 → 2048. Görsel token'ları dil modelinin boyutuna taşır.", "/kopru"],
};

const join = (parent: string, k: string) => (parent === "(kök)" ? k : parent + "." + k);

export function ModuleTreeView({ root = "language_model", initialSel, initialOpen }: { root?: string; initialSel?: string; initialOpen?: string[] }) {
  const { data, error } = useData<ModelTree>("model_tree");
  const [sel, setSel] = useState<string>(initialSel ?? root + ".model.layers.0.mlp");
  const [open, setOpen] = useState<Set<string>>(new Set(initialOpen ?? [root, root + ".model", root + ".model.layers", root + ".model.layers.0"]));
  const byName = useMemo(() => new Map((data?.nodes ?? []).map((n) => [n.name, n])), [data]);
  if (!data) return <Loading what="model ağacı" error={error} />;
  const node = byName.get(sel);

  const render = (name: string, d: number): JSX.Element | null => {
    const n = byName.get(name);
    if (!n) return null;
    const short = name.split(".").pop()!;
    const isLayers = short === "layers";
    const kids = isLayers ? [n.children[0]] : n.children;   // 18 katman aynı: ilkini göster
    const isOpen = open.has(name);
    return (
      <div key={name} style={{ marginLeft: d === 0 ? 0 : 16 }}>
        <div onClick={() => setSel(name)} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 6px", borderRadius: 6, cursor: "pointer", background: sel === name ? "var(--accent-soft)" : undefined }}>
          <span style={{ width: 14, color: "var(--fg-faint)" }}
            onClick={(e) => { e.stopPropagation(); const s = new Set(open); isOpen ? s.delete(name) : s.add(name); setOpen(s); }}>
            {kids.length ? (isOpen ? "▾" : "▸") : ""}
          </span>
          <span className="mono" style={{ fontSize: 13 }}>{d === 0 ? name : short}</span>
          {isLayers && <span className="pill" style={{ fontSize: 10.5 }}>×{n.children.length}</span>}
          <span className="small" style={{ color: n.cls === "Identity" ? "var(--warn)" : "var(--ax-I)" }}>{n.cls}</span>
          <span className="small muted mono" style={{ marginLeft: "auto" }}>{n.params ? fmtCount(n.params) : "0"}</span>
        </div>
        {isOpen && kids.map((k) => render(join(name, k), d + 1))}
      </div>
    );
  };

  const keys = node && node.name !== "(kök)" ? data.keys_ours.filter((k) => k.startsWith(node.name + ".")) : [];
  const what = node && WHAT[node.cls];
  return (
    <div className="grid2" style={{ fontFamily: "var(--sans)", alignItems: "start" }}>
      <div style={{ maxHeight: 460, overflowY: "auto" }}>{render(root, 0)}</div>
      {node && (
        <div style={{ background: "var(--bg-sunk)", borderRadius: 8, padding: 14, fontSize: 14 }}>
          <div className="mono" style={{ fontSize: 12, wordBreak: "break-all" }} title={node.name}>{node.name}</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--ax-I)", margin: "4px 0" }}>{node.cls}</div>
          {what && <p style={{ margin: "4px 0 10px" }}>{what[0]} {what[1] && <Link to={what[1]}>Bölüme git →</Link>}</p>}
          <table style={{ width: "100%", fontSize: 13 }}>
            <tbody>
              <tr><td className="muted">giriş</td><td>{node.in ? <S s={node.in.shape.join(",")} ax={axes(node.in.shape)} /> : "–"}</td></tr>
              <tr><td className="muted">çıkış</td><td>{node.out ? <S s={node.out.shape.join(",")} ax={axes(node.out.shape)} /> : "–"}</td></tr>
              <tr><td className="muted">parametre</td><td className="mono">{node.params.toLocaleString("tr-TR")}</td></tr>
              {Object.entries(node.param_shapes).map(([k, s]) => (
                <tr key={k}><td className="muted mono">.{k}</td><td><S s={s.join(",")} ax={axes(s)} /></td></tr>
              ))}
            </tbody>
          </table>
          {keys.length > 0 && keys.length <= 12 && (
            <div style={{ marginTop: 10 }}>
              <div className="small muted">state_dict anahtarları (HF checkpoint'iyle aynı olmalı):</div>
              {keys.map((k) => (
                <div key={k} className="mono" style={{ fontSize: 11.5, wordBreak: "break-all" }}>
                  {data.keys_ckpt.includes(k) ? <span style={{ color: "var(--ok)" }}>✓ </span> : <span style={{ color: "var(--warn)" }}>○ </span>}{k}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Şekildeki sayılardan eksen adı tahmini (renk için) */
export function axes(s: number[]): string {
  return s.map((n, i) => {
    if (i === 0 && s.length >= 3 && n === 1) return "B";
    if (n === 2048 || n === 1152 || n === 768) return "D";
    if (n === 16384 || n === 4304) return "I";
    if (n === 257216) return "V";
    if (n === 260 || n === 256 || n === 196) return "T";
    return "";
  }).join(",");
}

export function ParamShare() {
  const { data, error } = useData<ModelTree>("model_tree");
  if (!data) return <Loading what="parametreler" error={error} />;
  const g = (n: string) => data.nodes.find((x) => x.name === n)?.params ?? 0;
  const layers = data.nodes.filter((n) => /^language_model\.model\.layers\.\d+$/.test(n.name));
  const mlp = layers.reduce((s, l) => s + g(l.name + ".mlp"), 0);
  const norms = layers.reduce((s, l) => s + g(l.name + ".input_layernorm") + g(l.name + ".post_attention_layernorm"), 0) + g("language_model.model.norm");
  const emb = g("language_model.model.embed_tokens");
  const vision = g("vision_tower");
  const proj = g("multi_modal_projector");
  const t = data.config.text;
  const attn = layers.length * (t.hidden_size * t.num_attention_heads * t.head_dim * 2 + t.hidden_size * t.num_key_value_heads * t.head_dim * 2);
  const total = data.params_unique;
  const items = [
    { label: "SigLIP (görüntü)", value: vision, color: "var(--ax-T)" },
    { label: "projektör", value: proj, color: "var(--ax-I)" },
    { label: "embed_tokens = lm_head", value: emb, color: "var(--ax-V)" },
    { label: "18 × GemmaMLP", value: mlp, color: "var(--ax-D)" },
    { label: "RMSNorm'lar (37)", value: norms, color: "var(--ax-hd)" },
  ];
  return (
    <div style={{ fontFamily: "var(--sans)" }}>
      <div style={{ display: "flex", height: 34, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
        {items.map((i) => <div key={i.label} title={`${i.label}: ${fmtCount(i.value)}`} style={{ width: `${(i.value / (total + attn)) * 100}%`, background: i.color, minWidth: i.value ? 2 : 0 }} />)}
        <div title={`attention (henüz yok): ${fmtCount(attn)}`} style={{ width: `${(attn / (total + attn)) * 100}%`, background: "repeating-linear-gradient(45deg, var(--bg-sunk), var(--bg-sunk) 4px, var(--border-strong) 4px, var(--border-strong) 8px)" }} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))", gap: "6px 16px", marginTop: 10, fontSize: 13.5 }}>
        {items.map((i) => (
          <div key={i.label}><span style={{ color: i.color }}>■</span> {i.label}: <b className="mono">{fmtCount(i.value)}</b> <span className="muted">({((i.value / total) * 100).toFixed(1)}%)</span></div>
        ))}
        <div><span className="muted">▨</span> attention (henüz yok): <b className="mono">{fmtCount(attn)}</b> <span className="muted">henüz yok</span></div>
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>
        Şu an toplam <b className="mono">{fmtCount(total)}</b> tekil parametre (<code>model.parameters()</code> bağlı tensörü bir kez sayar).
        Attention gelince ≈ <b className="mono">{fmtCount(total + attn)}</b>, yani "3B".
      </div>
    </div>
  );
}
