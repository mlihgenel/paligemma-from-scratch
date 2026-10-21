// Bölüm 3: prefix-LM maskesi (kim kimi görüyor)
import { useMemo, useState } from "react";
import { Heatmap } from "../../components/charts";
import { Seg } from "../../components/ui";

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
