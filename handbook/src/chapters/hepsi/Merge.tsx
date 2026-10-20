// Bölüm 6: birleştirme adım adım, config akışı, prefix-LM maskesi önizlemesi
import { useMemo, useState } from "react";
import { useData } from "../../lib/data";
import { Heatmap } from "../../components/charts";
import { Loading, S, Seg } from "../../components/ui";

// Küçük örnek: 4 görsel token + <bos> caption tr \n + 2 pad, D = 4
const IDS = [257152, 257152, 257152, 257152, 2, 21209, 799, 108, 0, 0];
const PIECE = ["<image>", "<image>", "<image>", "<image>", "<bos>", "caption", "▁tr", "\\n", "<pad>", "<pad>"];
const D = 4;
const kind = (id: number) => (id === 257152 ? "img" : id === 0 ? "pad" : "txt");
const COLOR = { img: "var(--ax-T)", txt: "var(--ax-V)", pad: "var(--fg-faint)", zero: "var(--bg-sunk)" } as const;

const STEPS = [
  { t: "input_ids", code: "input_ids  # [B, T]", s: "1, 10", ax: "B,T", note: <>Processor'dan gelen id'ler. Görsel konumlarda <code>&lt;image&gt;</code> = 257152 yer tutucusu var. Burada gösterim için sona 2 pad koyduk (senin modelin şimdilik padding'i reddediyor).</> },
  { t: "inputs_embeds", code: "embed_tokens(input_ids)", s: "1, 10, 4", ax: "B,T,D", note: <>Önce <b>bütün</b> id'ler tablodan geçiyor, <code>&lt;image&gt;</code> da dahil. O satırların embedding'i birazdan atılacak. Gerçek checkpoint'te &lt;image&gt; satırının normu ≈ 7.3, ama hiç kullanılmıyor.</> },
  { t: "maskeler", code: "text_mask, image_mask, pad_mask  # [B, T]", s: "1, 10", ax: "B,T", note: <>Her konum için üç boolean. Üçü birbirini dışlıyor ve birlikte bütün diziyi kaplıyor.</> },
  { t: "unsqueeze(-1).expand", code: "mask.unsqueeze(-1).expand(-1, -1, embed_dim)", s: "1, 10, 4", ax: "B,T,D", note: <><S s="B, T" /> → <S s="B, T, 1" /> → <S s="B, T, D" />. <code>expand</code> bellek kopyalamıyor, aynı değeri D boyunca "görünüm" olarak tekrarlıyor.</> },
  { t: "torch.where (metin)", code: "torch.where(text_mask_expanded, inputs_embeds, final_embedding)", s: "1, 10, 4", ax: "B,T,D", note: <>Sıfırlardan başlayan tensöre metin konumları <b>aynı konumdan</b> kopyalanıyor. <code>where</code> eleman eleman seçtiği için iki kaynağın şekli aynı olmalı. inputs_embeds de <S s="B, T, D" />.</> },
  { t: "masked_scatter (görüntü)", code: "final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)", s: "1, 10, 4", ax: "B,T,D", note: <>Görsel özellikler <S s="B, 4, D" /> ama hedef <S s="B, 10, D" />. Şekiller farklı olduğu için <code>where</code> kullanılamaz. <code>masked_scatter</code> kaynağı <b>düzleştirip sırayla</b> True konumlarına döküyor: 1. görsel vektör ilk &lt;image&gt;'e, 2.'si ikinciye… Sıra bu yüzden önemli.</> },
  { t: "torch.where (pad → 0)", code: "torch.where(pad_mask_expanded, zeros, final_embedding)", s: "1, 10, 4", ax: "B,T,D", note: <>Pad konumları zaten sıfırdı. Bu satır güvence. Sonuç Gemma'ya gidiyor.</> },
];

export function MergeSteps() {
  const [st, setSt] = useState(0);
  const cell = (r: number, c: number) => {
    const k = kind(IDS[r]);
    if (st === 0 || st === 2) return null;
    if (st === 1) return { c: k === "img" ? "var(--fg-faint)" : COLOR[k], label: k === "img" ? "e(img)" : "" };
    if (st === 3) return { c: COLOR.zero, label: "" };
    if (st === 4) return k === "txt" ? { c: COLOR.txt, label: "" } : { c: COLOR.zero, label: "0" };
    if (st >= 5) {
      if (k === "txt") return { c: COLOR.txt, label: "" };
      if (k === "img") return { c: COLOR.img, label: `v${r}` };
      return { c: COLOR.zero, label: "0" };
    }
    return null;
  };
  const S_ = STEPS[st];
  return (
    <div>
      <div className="row" style={{ gap: 6, marginBottom: 12 }}>
        {STEPS.map((x, i) => <button key={i} className={"btn" + (i === st ? " primary" : "")} onClick={() => setSt(i)} style={{ fontSize: 12.5 }}>{i + 1}. {x.t}</button>)}
      </div>
      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: `122px repeat(${IDS.length}, minmax(52px, 1fr))`, gap: 4, minWidth: 640, fontFamily: "var(--mono)", fontSize: 11.5 }}>
          <div className="muted">konum</div>
          {IDS.map((_, i) => <div key={i} className="muted" style={{ textAlign: "center" }}>{i}</div>)}
          <div className="muted">token</div>
          {IDS.map((id, i) => (
            <div key={i} style={{ textAlign: "center", padding: "3px 0", borderRadius: 4, background: `color-mix(in srgb, ${COLOR[kind(id)]} 18%, transparent)`, color: COLOR[kind(id)], fontWeight: 600 }}>{PIECE[i]}<div style={{ fontSize: 9.5, opacity: 0.8 }}>{id}</div></div>
          ))}
          {(st === 2 || st === 3) && (["text", "image", "pad"] as const).map((m) => (
            <div key={m} style={{ display: "contents" }}>
              <div style={{ color: m === "text" ? COLOR.txt : m === "image" ? COLOR.img : COLOR.pad }}>{m}_mask</div>
              {IDS.map((id, i) => {
                const on = (m === "text" && kind(id) === "txt") || (m === "image" && kind(id) === "img") || (m === "pad" && kind(id) === "pad");
                return <div key={i} style={{ textAlign: "center", borderRadius: 4, padding: "2px 0", background: on ? "var(--ok-soft)" : "var(--bg-sunk)", color: on ? "var(--ok)" : "var(--fg-faint)" }}>{on ? "True" : "False"}{st === 3 && <div style={{ fontSize: 9 }}>×{D}</div>}</div>;
              })}
            </div>
          ))}
          {st !== 0 && st !== 2 && Array.from({ length: D }, (_, c) => (
            <div key={c} style={{ display: "contents" }}>
              <div className="muted">{c === 0 ? (st === 1 ? "inputs_embeds" : "final_embedding") : ""} <span style={{ color: "var(--ax-D)" }}>d{c}</span></div>
              {IDS.map((_, r) => {
                const x = cell(r, c);
                return <div key={r} style={{ height: 22, borderRadius: 3, background: x?.c, color: "white", textAlign: "center", fontSize: 10, lineHeight: "22px", transition: "background .35s" }}>{x?.label && c === 0 ? x.label : ""}</div>;
              })}
            </div>
          ))}
        </div>
      </div>
      {st === 5 && (
        <div className="row small" style={{ gap: 6, marginTop: 10, alignItems: "center" }}>
          <span className="muted">kaynak <span className="mono">scaled_image_features [1, 4, 4]</span>, düzleştirilmiş sırayla:</span>
          {[0, 1, 2, 3].map((i) => <span key={i} className="pill" style={{ color: "var(--ax-T)" }}>v{i} → konum {i}</span>)}
        </div>
      )}
      <div style={{ marginTop: 12, fontSize: 15 }}>
        <div className="mono small" style={{ background: "var(--code-bg)", padding: "6px 10px", borderRadius: 6, border: "1px solid var(--border)", overflowX: "auto" }}>{S_.code}  <span className="muted">→</span> <S s={S_.s} ax={S_.ax} /></div>
        <p style={{ margin: "8px 0 0" }}>{S_.note}</p>
      </div>
    </div>
  );
}

export function ConfigFlow() {
  const { data, error } = useData<any>("model_tree");
  if (!data) return <Loading what="config" error={error} />;
  const c = data.config;
  const rows: [string, any, string, string][] = [
    ["vision_config.image_size / patch_size", `${c.vision.image_size} / ${c.vision.patch_size}`, "__post_init__ → text_config.num_image_tokens", "processor kaç <image> koyacak, merge kaç görsel vektör bekleyecek"],
    ["text_config.num_image_tokens", c.text.num_image_tokens, "PaliGemmaProcessor(num_image_tokens=…)", "dizinin ilk 256 konumu"],
    ["projection_dim", c.projection_dim, "__post_init__ → vision_config.projection_dim", "projektörün çıkış boyutu (= Gemma hidden)"],
    ["vision_config.hidden_size", c.vision.hidden_size, "projektörün giriş boyutu", "1152 → 2048"],
    ["image_token_index", c.image_token_index, "_merge_input_ids_with_image_features", "image_mask = input_ids == 257152"],
    ["pad_token_id", c.text.pad_token_id, "merge (pad_mask) + nn.Embedding(padding_idx)", "pad konumları sıfır"],
    ["text_config.hidden_size", c.text.hidden_size, "merge (÷√d) + GemmaModel (×√d)", "√2048 = 45.25"],
    ["text_config.head_dim", c.text.head_dim, "GemmaAttention (sırada)", "config'ten okunuyor, hidden/heads ile hesaplanmıyor"],
    ["text_config.num_key_value_heads", c.text.num_key_value_heads, "MQA (sırada)", "k_proj/v_proj: 2048 → 256"],
  ];
  return (
    <table>
      <thead><tr><th>config alanı</th><th>değer</th><th>nerede kullanılıyor</th><th>etkisi</th></tr></thead>
      <tbody>{rows.map(([k, v, w, e]) => (
        <tr key={k}><td className="mono" style={{ fontSize: 12.5 }}>{k}</td><td className="mono"><b>{String(v)}</b></td><td className="small mono">{w}</td><td className="small">{e}</td></tr>
      ))}</tbody>
    </table>
  );
}

/** Önizleme: prefix-LM maskesi. Prefix (görüntü + prompt) kendi içinde tam, üretilenler causal */
export function PrefixMask() {
  const [gen, setGen] = useState(4);
  const [mode, setMode] = useState<"prefix" | "causal">("prefix");
  const nImg = 8, nTxt = 4;
  const P = nImg + nTxt, T = P + gen;
  const vals = useMemo(() => {
    const a = new Float32Array(T * T);
    for (let q = 0; q < T; q++) for (let k = 0; k < T; k++) {
      const ok = mode === "causal" ? k <= q : (k < P ? true : k <= q) && !(q < P && k >= P);
      a[q * T + k] = ok ? 1 : 0;
    }
    return a;
  }, [gen, mode, T, P]);
  const name = (i: number) => (i < nImg ? `img${i}` : i < P ? ["<bos>", "caption", "▁tr", "\\n"][i - nImg] : `üretilen ${i - P + 1}`);
  return (
    <div>
      <div className="row" style={{ gap: 12, alignItems: "center", marginBottom: 10 }}>
        <Seg value={mode} onChange={setMode} options={[["prefix", "prefix-LM (PaliGemma)"], ["causal", "tam causal"]]} />
        <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>üretilen token <input type="range" min={0} max={8} value={gen} onChange={(e) => setGen(+e.target.value)} /> <b>{gen}</b></label>
      </div>
      <div style={{ maxWidth: 420 }}>
        <Heatmap values={vals} rows={T} cols={T} domain={[0, 1]} legend={false} rowLabel="sorgu (q)" colLabel="anahtar (k)"
          cellLabel={(q, k, v) => {
            const why = mode === "causal" ? (k <= q ? "geçmiş ya da kendisi" : "gelecek")
              : q < P && k < P ? "ikisi de prefix: tam attention"
              : q < P ? "prefix, üretilenleri göremez (henüz yoklar)"
              : k < P ? "üretilen token bütün prefix'i görür"
              : k <= q ? "causal: önceki üretilenler" : "gelecek token";
            return <span>{name(q)} → {name(k)}: <b>{v ? "görebilir" : "göremez"}</b><br /><span className="muted">{why}</span></span>;
          }} />
      </div>
      <p className="small muted" style={{ margin: "6px 0 0" }}>
        Küçültülmüş örnek: 8 görsel + 4 prompt token'ı (gerçekte 256 + 4). Açık hücre = q, k'yi görebilir.
        {mode === "prefix" ? " Prefix bloğu tamamen açık: görüntü ve prompt birbirini her yönde görüyor. Üretilen token'lar ise sadece geçmişi görüyor." : " Tam causal'da ilk görsel token prompt'u hiç göremiyor. PaliGemma'nın seçmediği yol."}
      </p>
    </div>
  );
}
