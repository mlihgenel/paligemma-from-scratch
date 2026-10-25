// Chapter 2.1: the contrastive loss (softmax vs sigmoid) and the cost of resolution
import { useMemo, useState } from "react";
import { useData } from "../../lib/data";
import { rng, sigmoid } from "../../lib/tensor";
import { useWidth, Bars } from "../../components/charts";
import { Cell, CellCanvas } from "../../interact/cells";
import { Loading, Seg, Stat } from "../../components/ui";
import { useLang, useT } from "../../i18n";

const N = 6;
const IMG_L = {
  en: ["🐈 cat", "🚦 sign", "🍞 bread", "🐟 fish", "🕌 mosque", "🚗 car"],
  tr: ["🐈 kedi", "🚦 tabela", "🍞 ekmek", "🐟 balık", "🕌 cami", "🚗 araba"],
};
const TXT_L = {
  en: ["\"a cat\"", "\"emergency exit\"", "\"bagel\"", "\"anchovy\"", "\"minaret\"", "\"automobile\""],
  tr: ["\"bir kedi\"", "\"acil çıkış\"", "\"simit\"", "\"hamsi\"", "\"minare\"", "\"otomobil\""],
};

export function ContrastMatrix() {
  const lang = useLang();
  const IMG = IMG_L[lang], TXT = TXT_L[lang];
  const tr = useT();
  const [ref, W] = useWidth<HTMLDivElement>();
  const [mode, setMode] = useState<"softmax" | "sigmoid">("softmax");
  const [hov, setHov] = useState<[number, number] | null>(null);
  const S = useMemo(() => { const r = rng(17); return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => +(i === j ? 2.2 + r.normal() * 0.5 : r.normal() * 0.9).toFixed(2))); }, []);
  const t = 1, b = -2;   // sigmoid: z_ij (t·s + b)
  const narrow = W < 460;                      // phone: smaller label area and text so the column headers do not overlap
  const x0 = narrow ? 92 : 150, y0 = 40;
  const cs = Math.max(narrow ? 40 : 34, Math.min(60, (W - x0 - 30) / N));
  const cells: Cell[] = [];
  const row = (i: number) => S[i].map((v) => Math.exp(v));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const s = S[i][j];
    let dep = false;
    if (hov) dep = mode === "softmax" ? (i === hov[0] || j === hov[1]) : (i === hov[0] && j === hov[1]);
    const isH = hov && hov[0] === i && hov[1] === j;
    cells.push({ id: `${i}-${j}`, x: x0 + j * (cs + 3), y: y0 + i * (cs + 3), w: cs, h: cs, v: s, text: s.toFixed(1),
      opacity: hov ? (dep ? 1 : 0.25) : 1, ring: !!isH,
      tip: () => {
        if (mode === "softmax") {
          const e = row(i); const sum = e.reduce((a, q) => a + q, 0);
          return <><b>{IMG[i]} ↔ {TXT[j]}</b><div>p = e^{s.toFixed(1)} / Σ {tr("row", "satır")} = <b>{(Math.exp(s) / sum).toFixed(3)}</b></div><div className="muted">{tr(`denominator: all ${N} cells of the row (and the same along the column)`, `payda: satırdaki ${N} hücrenin hepsi (ve sütun yönünde de aynısı)`)}</div></>;
        }
        const z = i === j ? 1 : -1;
        const p = sigmoid(z * (t * s + b));
        return <><b>{IMG[i]} ↔ {TXT[j]}</b><div>σ({z > 0 ? "+" : "−"}(s{b < 0 ? " − 2" : ""})) = <b>{p.toFixed(3)}</b></div><div className="muted">{i === j ? tr("matching pair: pulled to 1", "doğru çift: 1'e çekiliyor") : tr("non-matching pair: pushed to 0", "yanlış çift: 0'a itiliyor")} · {tr("depends on no other cell", "başka hücreye bağlı değil")}</div></>;
      } });
  }
  const labels = <>
    {IMG.map((m, i) => <div key={i} style={{ position: "absolute", left: 0, top: y0 + i * (cs + 3), width: x0 - 10, height: cs, textAlign: "right", lineHeight: `${cs}px`, fontSize: 13 }}>{m}</div>)}
    {TXT.map((m, j) => <div key={j} style={{ position: "absolute", left: x0 + j * (cs + 3), top: 0, width: cs, height: y0 - 4, fontSize: narrow ? 9.5 : 11.5, display: "flex", alignItems: "flex-end", justifyContent: "center", textAlign: "center", color: "var(--fg-muted)" }}>{m}</div>)}
  </>;
  const loss = mode === "softmax"
    ? -S.reduce((a, r, i) => a + (r[i] - Math.log(r.reduce((s, v) => s + Math.exp(v), 0))), 0) / N
    : -S.reduce((a, r, i) => a + r.reduce((s, v, j) => s + Math.log(sigmoid((i === j ? 1 : -1) * (t * v + b))), 0), 0) / N;
  // The matrix takes at most ~560 px (cells stay at 60 px); the numbers on the right fill the rest of the wide figure
  return (
    <div className="matrix-fig">
      <div ref={ref} className="matrix-main">
        <div style={{ marginBottom: 10 }}>
          <Seg value={mode} onChange={setMode} options={[["softmax", "softmax (CLIP)"], ["sigmoid", "sigmoid (SigLIP)"]]} />
        </div>
        <CellCanvas cells={cells} width={W} height={y0 + N * (cs + 3) + 6} max={3} labels={labels}
          onHover={(id) => setHov(id ? (id.split("-").map(Number) as [number, number]) : null)} />
      </div>
      <div className="matrix-side">
        <p className="small muted" style={{ margin: 0 }}>{tr("Hover a cell: which cells does its loss depend on?", "Bir hücrenin üstüne gel: kaybı hangi hücrelere bağlı?")}</p>
        <div className="stats col">
          <Stat label={tr("loss per batch", "batch başına kayıp")} value={loss.toFixed(3)} sub={mode === "softmax" ? tr("mean of the row and column softmaxes (rows here)", "satır ve sütun softmax'larının ortalaması (burada satır)") : tr("N² independent binary classifications", "N² bağımsız ikili sınıflandırma")} />
          <Stat label={tr("cells one cell depends on", "bir hücrenin bağımlı olduğu hücre")} value={mode === "softmax" ? `${2 * N - 1}` : "1"} sub={mode === "softmax" ? tr("its own row + column", "kendi satırı + sütunu") : tr("only itself", "sadece kendisi")} />
        </div>
      </div>
    </div>
  );
}

export function ResolutionCost() {
  const lang = useLang();
  const tr = useT();
  const { data, error } = useData<any>("pipeline");
  const [res, setRes] = useState(224);
  if (!data) return <Loading what={tr("image", "görüntü")} error={error} />;
  const n = res / 14;
  const tokens = n * n;
  const T = tokens + 8;
  const base = (256 + 8) ** 2;
  return (
    <div className="grid2" style={{ alignItems: "start" }}>
      <div>
        <Seg value={res} onChange={setRes} options={[[224, "224 px"], [448, "448 px"], [896, "896 px"]]} />
        <div style={{ position: "relative", width: "100%", maxWidth: 340, aspectRatio: "1", marginTop: 12, backgroundImage: `url(${import.meta.env.BASE_URL}data/img_${data.images[1].name}_224.png)`, backgroundSize: "cover", imageRendering: res === 224 ? "pixelated" : "auto", borderRadius: 4 }}>
          <div style={{ position: "absolute", inset: 0, backgroundImage: `linear-gradient(to right, rgba(255,255,255,.55) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,.55) 1px, transparent 1px)`, backgroundSize: `${100 / n}% ${100 / n}%` }} />
        </div>
        <div className="small muted" style={{ marginTop: 6 }}>{n} × {n} {tr("patches, each 14 × 14 pixels", "yama, her biri 14 × 14 piksel")}</div>
      </div>
      <div>
        <div className="stats" style={{ marginTop: 0 }}>
          <Stat label={tr("image tokens", "görsel token")} value={tokens.toLocaleString(lang === "tr" ? "tr-TR" : "en-US")} />
          <Stat label={tr("sequence length T", "dizi uzunluğu T")} value={T.toLocaleString(lang === "tr" ? "tr-TR" : "en-US")} sub={tr("+ ~8 prompt tokens", "+ ~8 prompt token'ı")} />
        </div>
        <div style={{ marginTop: 18 }}>
          <Bars items={[
            { label: tr("MLP, norms, projections (∝ T)", "MLP, norm, projeksiyon (∝ T)"), value: T / 264, color: "var(--ax-D)" },
            { label: tr("attention scores (∝ T²)", "attention skorları (∝ T²)"), value: T * T / base, color: "var(--ax-T)" },
          ]} fmt={(v) => `×${v.toFixed(1)}`} max={Math.max(1, (4104 * 4104) / base)} />
        </div>
        <p className="small muted" style={{ marginTop: 10 }}>{tr("Multiples of the 224 px values. Doubling the resolution makes the token count 4× and the attention matrix (T × T) ~16× larger.", "224 px'e göre kaç kat. Çözünürlük 2 katına çıkınca token sayısı 4 kat, attention matrisi (T × T) ~16 kat büyüyor.")}</p>
      </div>
    </div>
  );
}
