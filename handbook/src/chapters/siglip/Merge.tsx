// Part 3: the prefix-LM mask (who sees whom)
import { useMemo, useState } from "react";
import { Heatmap } from "../../components/charts";
import { Seg } from "../../components/ui";
import { useT } from "../../i18n";

/** Preview: the prefix-LM mask. Full within the prefix (image + prompt), causal for generated tokens */
export function PrefixMask() {
  const tr = useT();
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
  const name = (i: number) => (i < nImg ? `img${i}` : i < P ? ["<bos>", "caption", "▁tr", "\\n"][i - nImg] : tr(`generated ${i - P + 1}`, `üretilen ${i - P + 1}`));
  return (
    <div>
      <div className="row" style={{ gap: 12, alignItems: "center", marginBottom: 10 }}>
        <Seg value={mode} onChange={setMode} options={[["prefix", "prefix-LM (PaliGemma)"], ["causal", "tam causal"]]} />
        <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>{tr("generated tokens", "üretilen token")} <input type="range" min={0} max={8} value={gen} onChange={(e) => setGen(+e.target.value)} /> <b>{gen}</b></label>
      </div>
      <div style={{ maxWidth: 420 }}>
        <Heatmap values={vals} rows={T} cols={T} domain={[0, 1]} legend={false} rowLabel="sorgu (q)" colLabel="anahtar (k)"
          cellLabel={(q, k, v) => {
            const why = mode === "causal" ? (k <= q ? tr("past or itself", "geçmiş ya da kendisi") : tr("future", "gelecek"))
              : q < P && k < P ? "ikisi de prefix: tam attention"
              : q < P ? tr("the prefix cannot see generated tokens (they do not exist yet)", "prefix, üretilenleri göremez (henüz yoklar)")
              : k < P ? tr("a generated token sees the whole prefix", "üretilen token bütün prefix'i görür")
              : k <= q ? tr("causal: earlier generated tokens", "causal: önceki üretilenler") : tr("future token", "gelecek token");
            return <span>{name(q)} → {name(k)}: <b>{v ? tr("can see", "görebilir") : tr("cannot see", "göremez")}</b><br /><span className="muted">{why}</span></span>;
          }} />
      </div>
      <p className="small muted" style={{ margin: "6px 0 0" }}>
        {tr("A scaled-down example: 8 image + 4 prompt tokens (256 + 4 in reality). Open cell = q can see k.", "Küçültülmüş örnek: 8 görsel + 4 prompt token'ı (gerçekte 256 + 4). Açık hücre = q, k'yi görebilir.")}
        {mode === "prefix" ? tr(" The prefix block is fully open: image and prompt see each other in every direction. Generated tokens only see the past.", " Prefix bloğu tamamen açık: görüntü ve prompt birbirini her yönde görüyor. Üretilen token'lar ise sadece geçmişi görüyor.") : tr(" With fully causal attention the first image token never sees the prompt. The path PaliGemma did not take.", " Tam causal'da ilk görsel token prompt'u hiç göremiyor. PaliGemma'nın seçmediği yol.")}
      </p>
    </div>
  );
}
