// Chapter 4.12: generating with and without a KV cache, step by step, and the rectangular mask.
import { useState } from "react";
import { S, Seg, Stat } from "../../../components/ui";
import { useT } from "../../../i18n";

const IMG = 256;
const TXT = 4;                  // <bos> caption ▁en \n
const PREFIX = IMG + TXT;       // 260
const ANSWER = ["a", "▁sign", "▁near", "▁the", "▁entrance", "<eos>"];
const LAYERS = 18;
const KV_BYTES_PER_TOKEN = LAYERS * 2 * 1 * 256 * 2;   // layers × (K, V) × kv_heads × head_dim × bf16

/** Tokens drawn as a strip: a few image cells, an ellipsis, the prompt, then the answer so far */
function TokenStrip({ step, cached, tr }: { step: number; cached: boolean; tr: (en: any, trk: any) => any }) {
  const nAns = step;                       // answer tokens already in the sequence
  const cell = 22, gap = 3;
  const cells: { label: string; kind: "img" | "txt" | "ans" | "dots"; fresh: boolean }[] = [];
  const freshAll = !cached || step === 0;  // processed in this step
  for (let i = 0; i < 4; i++) cells.push({ label: "", kind: "img", fresh: freshAll });
  cells.push({ label: "…", kind: "dots", fresh: freshAll });
  ["<bos>", "caption", "▁en", "\\n"].forEach((p) => cells.push({ label: p, kind: "txt", fresh: freshAll }));
  for (let i = 0; i < nAns; i++) cells.push({ label: ANSWER[i], kind: "ans", fresh: !cached || i === nAns - 1 });
  const color = (k: string) => (k === "img" ? "var(--ax-T)" : k === "txt" ? "var(--ax-V)" : k === "ans" ? "var(--accent)" : "var(--fg-faint)");
  return (
    <div style={{ display: "flex", gap, flexWrap: "wrap", alignItems: "flex-end", minHeight: 52 }}>
      {cells.map((c, i) => (
        <div key={i} title={c.label} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
          <div style={{
            width: c.kind === "dots" ? 18 : cell, height: cell, borderRadius: 4,
            background: c.kind === "dots" ? "transparent" : color(c.kind),
            opacity: c.fresh ? 1 : 0.25, outline: c.fresh && c.kind !== "dots" ? "2px solid var(--fg)" : "none", outlineOffset: 1,
            display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-muted)", fontSize: 13,
            transition: "opacity .35s, outline-color .35s",
          }}>{c.kind === "dots" ? "…" : ""}</div>
          <div className="mono" style={{ fontSize: 9.5, color: "var(--fg-muted)", maxWidth: 44, minHeight: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.kind === "img" && i === 0 ? tr("image", "görüntü") : c.label}</div>
        </div>
      ))}
    </div>
  );
}

/** The score matrix drawn to scale: T_new rows × T_all columns */
function ScoreBox({ rows, cols, maxCols }: { rows: number; cols: number; maxCols: number }) {
  const W = 220, k = W / maxCols;
  const h = Math.max(3, rows * k), w = cols * k;
  return (
    <svg width={W + 2} height={W + 2} style={{ display: "block" }}>
      <rect x={1} y={1} width={W} height={W} fill="none" stroke="var(--border)" strokeDasharray="3 3" />
      <rect x={1} y={1} width={w} height={h} fill="var(--accent-soft)" stroke="var(--accent)" style={{ transition: "all .4s" }} />
    </svg>
  );
}

/** Generation with and without the cache, step by step */
export function KvCacheStepper() {
  const tr = useT();
  const [step, setStep] = useState(0);
  const [cached, setCached] = useState(true);
  const N = ANSWER.length;
  const Tall = PREFIX + step;                              // tokens in the sequence during this forward
  const Tnew = cached && step > 0 ? 1 : Tall;              // tokens processed in this forward
  const pastLen = cached && step > 0 ? Tall - 1 : 0;
  const processedSoFar = (s: number, c: boolean) => {
    let n = 0;
    for (let i = 0; i <= s; i++) n += c && i > 0 ? 1 : PREFIX + i;
    return n;
  };
  const siglipRuns = (s: number, c: boolean) => (c ? 1 : s + 1);
  const cacheKB = (Tall * KV_BYTES_PER_TOKEN) / 1024;

  return (
    <div style={{ fontFamily: "var(--sans)", fontSize: 14 }}>
      <div className="row" style={{ gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <Seg value={cached ? "c" : "n"} onChange={(v) => setCached(v === "c")} options={[["c", tr("with KV cache", "KV cache'li")], ["n", tr("without cache", "cache'siz")]]} />
        <button className="btn" disabled={step === 0} onClick={() => setStep(step - 1)}>←</button>
        <input type="range" min={0} max={N - 1} value={step} onChange={(e) => setStep(+e.target.value)} style={{ flex: 1, minWidth: 140 }} aria-label={tr("step", "adım")} />
        <button className="btn primary" disabled={step === N - 1} onClick={() => setStep(step + 1)}>→</button>
        <b style={{ minWidth: 150 }}>{step === 0 ? tr("step 0: prefill", "adım 0: prefill") : tr(`step ${step}: token ${step + 1}`, `adım ${step}: ${step + 1}. token`)}</b>
      </div>

      <div className="small muted" style={{ marginBottom: 6 }}>{tr("What goes into the model in this step (outlined):", "Bu adımda modele giren (çerçeveli):")}</div>
      <TokenStrip step={step} cached={cached} tr={tr} />

      <div className="grid2" style={{ marginTop: 18, alignItems: "start" }}>
        <div>
          <div className="small" style={{ marginBottom: 6 }}><b>{tr("Score matrix of one head", "Bir head'in skor matrisi")}</b> <span className="muted">· {tr("to scale", "ölçekli")}</span></div>
          <ScoreBox rows={Tnew} cols={Tall} maxCols={PREFIX + N} />
          <div className="mono small" style={{ marginTop: 6 }}><S s={`1, 8, ${Tnew}, ${Tall}`} ax="B,H,T,T" /></div>
          <div className="small muted" style={{ marginTop: 4 }}>
            {Tnew === 1 ? tr("One row: the new token against everything before it. The mask row is all zeros.", "Tek satır: yeni token, öncesindeki her şeye karşı. Maske satırı tamamen sıfır.")
              : tr("Square: every token against every token, with the prefix-LM mask.", "Kare: her token her token'a karşı, prefix-LM maskesiyle.")}
          </div>
        </div>
        <div>
          <div className="small" style={{ marginBottom: 6 }}><b>{tr("KV cache, per layer", "KV cache, katman başına")}</b> <span className="muted">· {tr(`${LAYERS} layers`, `${LAYERS} katman`)}</span></div>
          {cached ? (
            <>
              {[0, 1, 2].map((l) => (
                <div key={l} className="row" style={{ alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span className="mono small muted" style={{ width: 70, whiteSpace: "nowrap" }}>{tr("layer", "katman")} {l === 2 ? "17" : l}</span>
                  <div style={{ flex: 1, height: 14, background: "var(--bg-elev)", borderRadius: 3, position: "relative", overflow: "hidden" }}>
                    <div style={{ position: "absolute", inset: 0, width: `${(PREFIX / (PREFIX + N)) * 100}%`, background: "var(--ax-T)", opacity: 0.55 }} />
                    <div style={{ position: "absolute", top: 0, bottom: 0, left: `${(PREFIX / (PREFIX + N)) * 100}%`, width: `${(step / (PREFIX + N)) * 100}%`, background: "var(--accent)", transition: "width .35s" }} />
                  </div>
                </div>
              ))}
              <div className="small muted">K, V: <S s={`1, 1, ${Tall}, 256`} ax="B,H,T,hd" /> {tr("each, after RoPE and before repeat_kv", "her biri, RoPE'dan sonra ve repeat_kv'den önce")}</div>
            </>
          ) : (
            <div className="small muted" style={{ padding: "10px 0" }}>{tr("Nothing is kept. Every step recomputes K and V for the whole sequence.", "Hiçbir şey saklanmıyor. Her adım bütün dizinin K ve V'sini baştan hesaplıyor.")}</div>
          )}
        </div>
      </div>

      <div className="stats" style={{ marginTop: 16 }}>
        <Stat label={tr("tokens processed in this step", "bu adımda işlenen token")} value={Tnew} tone={Tnew === 1 ? "ok" : undefined} />
        <Stat label={tr("tokens processed so far", "şimdiye kadar işlenen token")} value={processedSoFar(step, cached).toLocaleString()} sub={tr(`other mode: ${processedSoFar(step, !cached).toLocaleString()}`, `öbür mod: ${processedSoFar(step, !cached).toLocaleString()}`)} />
        <Stat label={tr("SigLIP runs so far", "şimdiye kadar SigLIP çalışması")} value={siglipRuns(step, cached)} tone={cached ? "ok" : undefined} />
        <Stat label={tr("position of the newest token", "en yeni token'ın konumu")} value={Tall} sub={cached && step > 0 ? `past_len + 1 = ${pastLen} + 1` : tr("positions 1 … T", "konumlar 1 … T")} />
        {cached && <Stat label={tr("cache size (bf16)", "cache boyutu (bf16)")} value={`${(cacheKB / 1024).toFixed(2)} MB`} sub={tr(`${(KV_BYTES_PER_TOKEN / 1024).toFixed(0)} KB per token`, `token başına ${(KV_BYTES_PER_TOKEN / 1024).toFixed(0)} KB`)} />}
      </div>
    </div>
  );
}

/** make_causal_mask(T, past_len): rows are the new tokens, columns all tokens */
export function RectMask() {
  const tr = useT();
  const [past, setPast] = useState(5);
  const [tnew, setTnew] = useState(1);
  const cols = past + tnew, c = Math.min(26, 300 / Math.max(cols, 1));
  return (
    <div style={{ fontFamily: "var(--sans)", fontSize: 14 }}>
      <div className="row" style={{ gap: 18, flexWrap: "wrap", marginBottom: 12 }}>
        <label className="small">past_len <b className="mono">{past}</b><br />
          <input type="range" min={0} max={8} value={past} onChange={(e) => setPast(+e.target.value)} aria-label="past_len" /></label>
        <label className="small">T ({tr("new tokens", "yeni token")}) <b className="mono">{tnew}</b><br />
          <input type="range" min={1} max={6} value={tnew} onChange={(e) => setTnew(+e.target.value)} aria-label="T" /></label>
      </div>
      <svg width={cols * c + 60} height={tnew * c + 30} style={{ display: "block" }}>
        {Array.from({ length: tnew }, (_, i) => Array.from({ length: cols }, (_, j) => {
          const blocked = j > past + i;
          return <rect key={`${i}-${j}`} x={40 + j * c} y={20 + i * c} width={c - 2} height={c - 2} rx={2}
            fill={blocked ? "var(--bad)" : j < past ? "var(--ax-T)" : "var(--ok)"} opacity={blocked ? 0.35 : 0.8} />;
        }))}
        {Array.from({ length: tnew }, (_, i) => <text key={i} x={34} y={20 + i * c + c / 2 + 4} textAnchor="end" fontSize={11} fill="var(--fg-muted)" fontFamily="var(--mono)">{past + i}</text>)}
        <text x={40} y={13} fontSize={11} fill="var(--fg-muted)">{tr("columns: past (blue) + new (green), red = blocked", "sütunlar: geçmiş (mavi) + yeni (yeşil), kırmızı = yasak")}</text>
      </svg>
      <div className="small muted" style={{ marginTop: 6 }}>
        <S s={`1, 1, ${tnew}, ${cols}`} ax="B,H,T,T" />{" "}
        {tnew === 1 ? tr("A single row with nothing blocked: during generation the mask is all zeros.", "Hiçbir yeri yasak olmayan tek satır: üretimde maske tamamen sıfır.")
          : tr("Equal to the last rows of the square causal mask of size past_len + T.", "past_len + T boyutlu kare nedensel maskenin son satırlarıyla aynı.")}
      </div>
    </div>
  );
}
