// Bölüm 3: _merge_input_ids_with_image_features, hücre hücre ve eleman takibiyle.
import { ReactNode } from "react";
import { rng } from "../../lib/tensor";
import { useWidth } from "../../components/charts";
import { Cell, CellCanvas } from "../../interact/cells";
import { S } from "../../components/ui";

const IDS = [257152, 257152, 257152, 257152, 2, 21209, 799, 108, 0, 0];
const PIECE = ["<image>", "<image>", "<image>", "<image>", "<bos>", "caption", "▁tr", "\\n", "<pad>", "<pad>"];
const T = IDS.length, D = 4, NI = 4;
const kind = (id: number) => (id === 257152 ? "img" : id === 0 ? "pad" : "txt");
const r = rng(31);
const EMB = IDS.map(() => Array.from({ length: D }, () => +(r.normal() * 0.8).toFixed(2)));     // embed_tokens(input_ids)
const IMG = Array.from({ length: NI }, () => Array.from({ length: D }, () => +(r.normal() * 1.2).toFixed(2)));   // scaled_image_features

const CODE = [
  ["input_ids", "B, T", "B,T"],
  ["inputs_embeds = embed_tokens(input_ids)", "B, T, D", "B,T,D"],
  ["text_mask, image_mask, pad_mask", "B, T", "B,T"],
  ["mask.unsqueeze(-1).expand(-1, -1, embed_dim)", "B, T, D", "B,T,D"],
  ["torch.where(text_mask_expanded, inputs_embeds, final_embedding)", "B, T, D", "B,T,D"],
  ["final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)", "B, T, D", "B,T,D"],
  ["torch.where(pad_mask_expanded, 0, final_embedding)", "B, T, D", "B,T,D"],
];

export function MergeScatter({ step }: { step: number }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const cs = Math.max(24, Math.min(40, (W - 120 - 140) / (T + NI * 0 + 3)));
  const g = 3;
  const xL = 92;
  const X = (t: number) => xL + t * (cs + g);
  const yTok = 26;
  const yA = yTok + 46;                          // inputs_embeds bandı
  const yM = yA + D * (cs + g) + 26;              // maske bandı
  const mh = step >= 3 ? 9 : cs * 0.55;           // maske satır yüksekliği (expand'de D alt satıra bölünüyor)
  const mBand = step >= 3 ? D * (mh + 2) + 10 : cs * 0.55 + 6;
  const yF = yM + 3 * mBand + 26;                 // final_embedding bandı
  const xS = X(T) + 36;                           // görsel özellik kaynağı (sağda)
  const H = yF + D * (cs + g) + 30;
  const cells: Cell[] = [];
  const colorOf = (k: string) => (k === "img" ? "var(--ax-T)" : k === "txt" ? "var(--ax-V)" : "var(--fg-faint)");

  // token satırı
  IDS.forEach((id, t) => {
    const k = kind(id);
    cells.push({ id: `tok${t}`, x: X(t), y: yTok, w: cs, h: 34, fill: `color-mix(in srgb, ${colorOf(k)} 18%, transparent)`, text: PIECE[t].length > 6 ? PIECE[t].slice(0, 5) + "…" : PIECE[t],
      tip: () => <><b>{PIECE[t]}</b> · id {id}<div className="muted">{k === "img" ? "görsel yer tutucu" : k === "pad" ? "dolgu" : "metin"}</div></> });
  });
  // inputs_embeds: bütün id'ler tablodan geçiyor (görsel yer tutucular da!)
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const k = kind(IDS[t]);
    const v = EMB[t][d];
    cells.push({ id: `a${t}${d}`, x: X(t), y: yA + d * (cs + g), w: cs, h: cs, v: k === "pad" ? 0 : v, text: cs > 30 ? v.toFixed(1) : "", opacity: step >= 1 ? (step >= 4 ? 0.3 : 1) : 0,
      tip: () => <>inputs_embeds[t={t}, d={d}] = <b>{v}</b>{k === "img" && <div className="muted">&lt;image&gt; satırının embedding'i: birazdan üzerine yazılacak</div>}</> });
  }
  // maskeler
  (["text", "image", "pad"] as const).forEach((m, mi) => {
    for (let t = 0; t < T; t++) {
      const k = kind(IDS[t]);
      const on = (m === "text" && k === "txt") || (m === "image" && k === "img") || (m === "pad" && k === "pad");
      for (let d = 0; d < D; d++) {
        const show = step >= 2 && step <= 6 && (d === 0 || step >= 3);
        cells.push({ id: `m${mi}${t}${d}`, x: X(t), y: yM + mi * mBand + (step >= 3 ? d * (mh + 2) : 0), w: cs, h: mh,
          fill: on ? (m === "text" ? "var(--ax-V)" : m === "image" ? "var(--ax-T)" : "var(--fg-muted)") : "var(--bg-sunk)", opacity: show ? (step >= 4 ? 0.45 : 1) : 0,
          text: step === 2 && cs > 30 ? (on ? "T" : "F") : "",
          tip: () => <>{m}_mask[t={t}]{step >= 3 ? `, d=${d}` : ""} = <b>{on ? "True" : "False"}</b>{step >= 3 && <div className="muted">expand: aynı değer D boyunca (kopya değil, görünüm)</div>}</> });
      }
    }
  });
  // final_embedding: sıfırlardan başlıyor
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const k = kind(IDS[t]);
    cells.push({ id: `z${t}${d}`, x: X(t), y: yF + d * (cs + g), w: cs, h: cs, fill: "var(--bg-sunk)", text: cs > 30 ? "0" : "", ring: step === 6 && k === "pad", opacity: step >= 4 && !(k === "txt") && !(k === "img" && step >= 5) ? 1 : 0,
      tip: () => <>final_embedding[t={t}, d={d}] = <b>0</b><div className="muted">{k === "pad" ? "pad konumu: sıfır kalıyor" : "henüz yazılmadı"}</div></> });
    // metin kopyaları: inputs_embeds'ten AYNI konuma (where)
    if (k === "txt") cells.push({ id: `w${t}${d}`, x: X(t), y: step >= 4 ? yF + d * (cs + g) : yA + d * (cs + g), w: cs, h: cs, v: EMB[t][d], text: cs > 30 ? EMB[t][d].toFixed(1) : "", opacity: step >= 4 ? 1 : 0, z: 4,
      tip: () => <>where(text_mask, inputs_embeds, …)<div>final[t={t}, d={d}] = inputs_embeds[t={t}, d={d}] = <b>{EMB[t][d]}</b></div></> });
  }
  // görsel özellikler: sağdaki kaynaktan <image> konumlarına SIRAYLA
  const imgSlots = IDS.map((id, t) => (kind(id) === "img" ? t : -1)).filter((t) => t >= 0);
  for (let i = 0; i < NI; i++) for (let d = 0; d < D; d++) {
    const t = imgSlots[i];
    const placed = step >= 5;
    cells.push({ id: `s${i}${d}`, x: placed ? X(t) : xS + i * (cs + g), y: yF + d * (cs + g), w: cs, h: cs, fill: "var(--ax-T)", text: cs > 30 ? IMG[i][d].toFixed(1) : `v${i}`,
      opacity: step >= 4 ? 1 : 0, delay: placed ? i * 0.25 : 0, z: 5,
      tip: () => <>scaled_image_features[0, {i}, {d}] = <b>{IMG[i][d]}</b><div className="muted">{placed ? `→ final[t=${t}, d=${d}] (${i + 1}. True konumu)` : `kaynak, düzleştirilmiş sırada ${i * D + d}. eleman`}</div></> });
  }
  const [code, shape, ax] = CODE[Math.min(step, CODE.length - 1)];
  const lbl = (y: number, text: ReactNode, color = "var(--fg-muted)") => <div style={{ position: "absolute", left: 0, top: y, width: xL - 10, textAlign: "right", fontSize: 12, color, lineHeight: 1.2 }}>{text}</div>;
  const labels = <>
    {lbl(yTok + 10, "input_ids")}
    {step >= 1 && lbl(yA + 4, <>inputs_embeds<br /><span className="mono" style={{ fontSize: 10.5 }}>[B, T, D]</span></>)}
    {step >= 2 && step <= 6 && (["text", "image", "pad"] as const).map((m, mi) => <div key={m}>{lbl(yM + mi * mBand, `${m}_mask`, m === "text" ? "var(--ax-V)" : m === "image" ? "var(--ax-T)" : "var(--fg-muted)")}</div>)}
    {step >= 4 && lbl(yF + 4, <>final_embedding<br /><span className="mono" style={{ fontSize: 10.5 }}>[B, T, D]</span></>, "var(--fg)")}
    {step === 4 && <div style={{ position: "absolute", left: xS, top: yF - 20, fontSize: 12, color: "var(--ax-T)" }}>scaled_image_features [1, {NI}, {D}]</div>}
  </>;
  return (
    <div ref={ref}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
        <code style={{ fontSize: 13 }}>{code}</code><S s={shape} ax={ax} />
      </div>
      <CellCanvas cells={cells} width={W} height={H} max={2.4} labels={labels} />
      <div className="small muted">Hücrelerin üstüne gel: her değerin nereden geldiği görünür.</div>
    </div>
  );
}
