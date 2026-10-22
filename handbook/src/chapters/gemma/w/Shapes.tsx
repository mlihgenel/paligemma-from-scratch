// 5.0: tensörün şekli, forward boyunca. Dikdörtgen: yükseklik T, genişlik son eksen (log ölçek).
import { motion } from "framer-motion";
import { S } from "../../../components/ui";

const STEPS = [
  { code: "input_ids", shape: [1, 260], ax: "B,T", dtype: "int64", last: 1, note: "token id'leri: 256 × <image> + <bos> caption ▁tr \\n" },
  { code: "embed_tokens(input_ids)", shape: [1, 260, 2048], ax: "B,T,D", dtype: "float32", last: 2048, note: "her id tablodan bir satır alıyor" },
  { code: "masked_scatter(image_mask, image_features)", shape: [1, 260, 2048], ax: "B,T,D", dtype: "float32", last: 2048, note: "256 konuma görsel özellikler ölçeksiz yazılıyor, metin satırları ×45.25 ile ölçeklenmiş" },
  { code: "language_model(…, scale_embeddings=False)", shape: [1, 260, 2048], ax: "B,T,D", dtype: "float32", last: 2048, note: "GemmaModel girişi: dizi zaten ölçekli, ikinci kez ×45.25 yok. Şekil aynı" },
  { code: "for decoder_layer in self.layers", shape: [1, 260, 2048], ax: "B,T,D", dtype: "float32", last: 2048, note: "18 katman, her biri aynı şekli koruyor" },
  { code: "self.norm(hidden_states)", shape: [1, 260, 2048], ax: "B,T,D", dtype: "float32", last: 2048, note: "akış tek bir kez normalize ediliyor" },
  { code: "self.lm_head(hidden_states)", shape: [1, 260, 257216], ax: "B,T,V", dtype: "float32", last: 257216, note: "her konum için sözlükteki her token'a bir skor" },
];

export function ShapeJourney({ step }: { step: number }) {
  const s = STEPS[Math.min(step, STEPS.length - 1)];
  const H = 240;
  const w = s.last === 1 ? 26 : 60 + Math.log2(s.last) * 26;
  const imgH = H - 22;                    // metin bandı okunsun diye büyütüldü (ölçekli değil)
  const imgColor = step === 0 ? "var(--ax-T)" : step >= 2 ? "var(--ax-I)" : "var(--fg-faint)";
  const txtColor = "var(--ax-V)";
  const n = s.shape.reduce((a, b) => a * b, 1);
  const mb = (n * (s.dtype === "int64" ? 8 : 4)) / 2 ** 20;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
        <code style={{ fontSize: 14 }}>{s.code}</code>
      </div>
      <div style={{ position: "relative", height: H + 70, display: "flex", alignItems: "flex-start", gap: 26 }}>
        <div style={{ position: "relative", height: H }}>
          {/* katman yığını (adım 4) */}
          {[3, 2, 1].map((k) => (
            <motion.div key={k} initial={false} animate={{ opacity: step === 4 ? 0.35 : 0, x: step === 4 ? k * 10 : 0, y: step === 4 ? -k * 10 : 0, width: w }}
              style={{ position: "absolute", top: 0, left: 0, height: H, border: "1px solid var(--border-strong)", borderRadius: 3, background: "var(--bg-sunk)" }} />
          ))}
          <motion.div initial={false} animate={{ width: w }} transition={{ type: "spring", stiffness: 120, damping: 20 }}
            style={{ position: "relative", height: H, borderRadius: 3, overflow: "hidden", boxShadow: step === 5 ? "0 0 0 2px var(--ax-hd)" : step === 3 ? "0 0 0 2px var(--accent)" : "none" }}>
            <motion.div initial={false} animate={{ backgroundColor: step >= 6 ? "var(--ax-V)" : imgColor, opacity: step >= 6 ? 0.55 : 0.85 }} style={{ height: imgH }} />
            <motion.div initial={false} animate={{ backgroundColor: txtColor, opacity: step >= 6 ? 0.55 : 0.95 }} style={{ height: H - imgH }} />
          </motion.div>
          <div style={{ position: "absolute", left: -46, top: 0, height: H, display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 11.5, color: "var(--fg-muted)", fontFamily: "var(--mono)", textAlign: "right", width: 40 }}>
            <span>0</span><span style={{ color: "var(--ax-T)" }}>T = 260</span><span>259</span>
          </div>
          <motion.div initial={false} animate={{ width: w }} style={{ position: "absolute", top: H + 8, left: 0, textAlign: "center", fontFamily: "var(--mono)", fontSize: 12, color: s.ax.endsWith("V") ? "var(--ax-V)" : s.ax.endsWith("D") ? "var(--ax-D)" : "var(--fg-muted)" }}>
            {s.last === 1 ? "—" : `${s.last.toLocaleString("tr-TR")} →`}
          </motion.div>
        </div>
        <div style={{ paddingTop: 4, minWidth: 180 }}>
          <div style={{ fontSize: 13, color: "var(--fg-muted)" }}>şekil</div>
          <div style={{ fontSize: 22, marginBottom: 10 }}><S s={s.shape.join(", ")} ax={s.ax} /></div>
          <div style={{ fontSize: 13, color: "var(--fg-muted)" }}>dtype · bellek</div>
          <div className="mono" style={{ fontSize: 15, marginBottom: 10 }}>{s.dtype} · {mb < 1 ? (mb * 1024).toFixed(1) + " KB" : mb.toFixed(1) + " MB"}</div>
          <div style={{ fontSize: 14, color: "var(--fg)", maxWidth: 260 }}>{s.note}</div>
          <div style={{ fontSize: 12, marginTop: 12, color: "var(--fg-muted)" }}>
            <span style={{ color: imgColor }}>■</span> görsel konumlar (256) · <span style={{ color: txtColor }}>■</span> metin (4, görünsün diye büyütüldü)
          </div>
        </div>
      </div>
    </div>
  );
}
