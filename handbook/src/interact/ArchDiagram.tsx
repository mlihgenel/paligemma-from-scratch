// Figure 1 of the paper (SigLIP → Linear projection → Gemma) as a timed animation.
// variant="full": the playable sequence on the landing page. variant="mini": the static "you are here" strip at the top of each page.
import { ReactNode, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { ArchKey } from "../registry";
import { useLang, useT, type L } from "../i18n";

const BASE = import.meta.env.BASE_URL + "data/";
const IMG = "c02";
// Real tokenizer output (pipeline.json, prompt "ocr")
const PROMPT = "ocr";
const PIECES = ["<bos>", "ocr", "\\n"];
// The answer our implementation generated for this photo with the real weights (prompt "ocr")
const OUT = ["ACİL", "ÇIKIŞ", "<eos>"];

// Timeline (seconds). Each phase has a caption and a shape chip.
const PHASES: { at: number; title: L<string>; text: L<ReactNode>; shape?: L<string> }[] = [
  { at: 0, title: { en: "Patches", tr: "Yamalar" }, shape: { en: "[1, 3, 224, 224]", tr: "[1, 3, 224, 224]" },
    text: { en: <>The photo is resized to 224 × 224 and cut into <b>256 patches</b> of 14 × 14 pixels.</>, tr: <>Fotoğraf 224 × 224'e getiriliyor ve 14 × 14 piksellik <b>256 yamaya</b> bölünüyor.</> } },
  { at: 2.2, title: { en: "SigLIP", tr: "SigLIP" }, shape: { en: "[1, 256, 1152]", tr: "[1, 256, 1152]" },
    text: { en: <>The image model SigLIP turns each patch into a vector of <b>1152 numbers</b>. Over 27 layers the patches look at each other and gather context.</>, tr: <>Görüntü modeli SigLIP her yamayı <b>1152 sayılık</b> bir vektöre çeviriyor. 27 katman boyunca yamalar birbirine bakıp bağlam topluyor.</> } },
  { at: 4.6, title: { en: "Projector", tr: "Projektör" }, shape: { en: "[1, 256, 2048]", tr: "[1, 256, 2048]" },
    text: { en: <>A single Linear layer moves every vector to <b>2048 dimensions</b>. Now there are 256 "visual words" the language model can read.</>, tr: <>Tek bir Linear katman her vektörü <b>2048 boyuta</b> taşıyor. Artık dil modelinin anlayacağı 256 "görsel kelime" var.</> } },
  { at: 6.8, title: { en: "Prompt", tr: "Komut" }, shape: { en: "[1, 259, 2048]", tr: "[1, 259, 2048]" },
    text: { en: <>The tokenizer splits the prompt (<code>{PROMPT}</code>) into pieces. Each piece gets its own 2048-dim vector and is appended <b>after</b> the image tokens.</>, tr: <>Komut (<code>{PROMPT}</code>) tokenizer ile parçalara ayrılıyor. Her parça kendi 2048'lik vektörünü alıp görsel token'ların <b>arkasına</b> ekleniyor.</> } },
  { at: 9.2, title: { en: "Gemma", tr: "Gemma" }, shape: { en: "[1, 259, 2048] → 18 layers", tr: "[1, 259, 2048] → 18 katman" },
    text: { en: <>The language model Gemma reads the whole sequence. Image and prompt see each other <b>fully</b> (the prefix), the answer only looks at the <b>past</b>.</>, tr: <>Dil modeli Gemma bütün diziyi okuyor. Görüntü ve komut birbirini <b>tam</b> görüyor (prefix), cevap sadece <b>geçmişe</b> bakıyor.</> } },
  { at: 11.6, title: { en: "Answer", tr: "Cevap" }, shape: { en: "last position → 257216 scores → argmax", tr: "son konum → 257216 skor → argmax" },
    text: { en: <>The answer is generated <b>token by token</b>. Each new token is appended to the sequence and fed back to the model. Generation stops at <code>&lt;eos&gt;</code>.</>, tr: <>Cevap <b>token token</b> üretiliyor. Her yeni token dizinin sonuna eklenip modele geri veriliyor. <code>&lt;eos&gt;</code> gelince üretim duruyor.</> } },
];
const END = 16.4;

const clamp = (x: number) => Math.max(0, Math.min(1, x));
const ease = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const seg = (t: number, a: number, b: number) => ease(clamp((t - a) / (b - a)));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

// Geometri (viewBox 1000 × 400)
const IM = { x: 24, y: 36, s: 128 };
const ENC = { x: 202, y: 44, w: 164, h: 112 };
const PROJ = { x: 398, y: 44, w: 36, h: 112 };
const PR = { x: 24, y: 244, w: 128, h: 60 };
const TOKZ = { x: 202, y: 244, w: 164, h: 60 };
const COL = { x: 466, sq: 16, gap: 5 };
const GEM = { x: 600, y: 44, w: 196, h: 140 };
const OUTX = 860;
const IMG_SLOTS = 7;
const imgY = (i: number) => 34 + i * (COL.sq + COL.gap);
const txtY = (i: number) => 214 + i * (COL.sq + COL.gap);
// Flying patches: a few patches picked from the 16 × 16 grid (row, column)
const FLY: [number, number][] = [[2, 3], [4, 11], [6, 6], [8, 13], [9, 2], [11, 8], [13, 5], [14, 12]];
const OUT_T = (k: number) => 11.9 + k * 1.45;

const LINK: Record<ArchKey, string> = { image: "/patches", siglip: "/big-picture", proj: "/bridge", prompt: "/bridge", tokenizer: "/bridge", gemma: "/gemma", output: "/gemma/tying" };

function Patch({ r, c, x, y, s, opacity = 1 }: { r: number; c: number; x: number; y: number; s: number; opacity?: number }) {
  return (
    <svg x={x - s / 2} y={y - s / 2} width={s} height={s} viewBox={`${c * 14} ${r * 14} 14 14`} opacity={opacity} style={{ overflow: "hidden" }}>
      <image href={`${BASE}img_${IMG}_224.png`} width={224} height={224} style={{ imageRendering: "pixelated" }} />
    </svg>
  );
}

function Arrow({ d, k = 1, flow = 0 }: { d: string; k?: number; flow?: number }) {
  return (
    <g opacity={0.25 + 0.75 * k}>
      <path d={d} fill="none" stroke="var(--fg-faint)" strokeWidth={1.4} markerEnd="url(#arch-arrow)" />
      {flow > 0 && flow < 1 && <path d={d} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="6 10" strokeDashoffset={-flow * 160} opacity={Math.sin(flow * Math.PI)} />}
    </g>
  );
}

function useReducedMotion() {
  const [r] = useState(() => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  return r;
}

/** Player: advances t with rAF and starts by itself once it becomes visible */
function usePlayer(enabled: boolean) {
  const reduced = useReducedMotion();
  const [t, setT] = useState(enabled && !reduced ? 0 : END);
  const [playing, setPlaying] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const started = useRef(false);
  useEffect(() => {
    if (!enabled || reduced || !ref.current) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started.current) { started.current = true; setPlaying(true); }
    }, { threshold: 0.4 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [enabled, reduced]);
  useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      setT((x) => {
        const n = x + dt;
        if (n >= END) { setPlaying(false); return END; }
        return n;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
  return { t, setT, playing, setPlaying, ref };
}

export function ArchDiagram({ variant = "full", active = [] }: { variant?: "full" | "mini"; active?: ArchKey[] }) {
  const full = variant === "full";
  const { t, setT, playing, setPlaying, ref } = usePlayer(full);
  const nav = useNavigate();
  const lang = useLang();
  const tr = useT();
  const phase = PHASES.reduce((k, p, i) => (t >= p.at ? i : k), 0);

  // Box highlight: the current phase in full mode, the page's boxes in mini mode
  const on = (k: ArchKey) => {
    if (!full) return active.length === 0 || active.includes(k) ? 1 : 0.22;
    const now: Record<number, ArchKey[]> = { 0: ["image"], 1: ["siglip"], 2: ["proj"], 3: ["prompt", "tokenizer"], 4: ["gemma"], 5: ["gemma", "output"] };
    return t >= END - 0.01 || now[phase].includes(k) ? 1 : 0.55;
  };
  const glow = (k: ArchKey) => (full ? on(k) === 1 && t < END - 0.01 : active.includes(k) && active.length < 7);
  const box = (k: ArchKey, children: ReactNode) => (
    <g opacity={on(k)} style={{ cursor: "pointer", transition: full ? undefined : "opacity .3s" }} onClick={() => nav(LINK[k])} role="link" aria-label={k}>
      {children}
    </g>
  );

  // Progress values
  const gridK = seg(t, 0.5, 1.6);
  const encK = seg(t, 2.2, 4.4);                      // yamalar SigLIP'e
  const projK = seg(t, 4.6, 6.6);                     // SigLIP → projector → column
  const imgTok = full ? projK : 1;
  const promptK = seg(t, 6.8, 7.6), tokzK = seg(t, 7.6, 9.0);
  const txtTok = full ? tokzK : 1;
  const gemK = seg(t, 9.2, 10.4);
  const maskK = full ? seg(t, 10.0, 11.2) : 1;
  const nOut = full ? OUT.filter((_, k) => t >= OUT_T(k)).length : OUT.length;

  const colTop = imgY(0) - 8;
  const colBot = txtY(PIECES.length + Math.max(0, nOut - 1)) + COL.sq + 8;

  const svg = (
    <svg viewBox={full ? "0 0 1000 400" : "0 8 1000 350"} width="100%" style={{ display: "block", fontFamily: "var(--sans)", overflow: "visible" }}>
      <defs>
        <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--fg-faint)" />
        </marker>
        <clipPath id="arch-img"><rect x={IM.x} y={IM.y} width={IM.s} height={IM.s} rx={4} /></clipPath>
      </defs>

      {/* ---- Image ---- */}
      {box("image", <>
        <image href={`${BASE}img_${IMG}_224.png`} x={IM.x} y={IM.y} width={IM.s} height={IM.s} clipPath="url(#arch-img)" />
        <g opacity={gridK * (1 - 0.6 * encK)}>
          {Array.from({ length: 15 }, (_, i) => {
            const p = IM.x + ((i + 1) * IM.s) / 16;
            const q = IM.y + ((i + 1) * IM.s) / 16;
            return <g key={i}><line x1={p} y1={IM.y} x2={p} y2={IM.y + IM.s} stroke="#fff" strokeOpacity={0.7} strokeWidth={0.6} /><line x1={IM.x} y1={q} x2={IM.x + IM.s} y2={q} stroke="#fff" strokeOpacity={0.7} strokeWidth={0.6} /></g>;
          })}
        </g>
        <text x={IM.x + IM.s / 2} y={IM.y + IM.s + 18} textAnchor="middle" fontSize={12} fill="var(--fg-muted)">{tr("image · 16 × 16 patches", "görüntü · 16 × 16 yama")}</text>
      </>)}
      <Arrow d={`M${IM.x + IM.s + 6},${IM.y + IM.s / 2} L${ENC.x - 22},${ENC.y + ENC.h / 2}`} k={full ? Math.max(0.3, seg(t, 1.6, 2.2)) : 1} flow={full ? clamp((t - 2.2) / 2.2) : 0} />

      {/* ---- SigLIP ---- */}
      {box("siglip", <>
        <rect x={ENC.x - 16} y={16} width={ENC.w + 32} height={186} rx={10} fill="var(--part-vision-soft)" stroke="var(--part-vision)" strokeDasharray="5 5" strokeWidth={1.2} />
        <rect x={ENC.x} y={ENC.y} width={ENC.w} height={ENC.h} rx={8} fill="var(--part-vision)" fillOpacity={glow("siglip") ? 0.3 : 0.16} stroke="var(--part-vision)" strokeWidth={glow("siglip") ? 2 : 1} />
        <text x={ENC.x + ENC.w / 2} y={ENC.y + 46} textAnchor="middle" fontSize={15} fontWeight={650} fill="var(--fg)">{tr("Image encoder", "Görüntü kodlayıcı")}</text>
        <text x={ENC.x + ENC.w / 2} y={ENC.y + 66} textAnchor="middle" fontSize={12} fill="var(--fg-muted)">{tr("27 layers · 1152 dims", "27 katman · 1152 boyut")}</text>
        <text x={ENC.x - 4} y={190} fontSize={12.5} fontWeight={700} fill="var(--part-vision)">SigLIP <tspan fontWeight={500} fill="var(--fg-muted)">{tr("· 400M image model", "· 400M görüntü modeli")}</tspan></text>
      </>)}

      {/* ---- Projector ---- */}
      <Arrow d={`M${ENC.x + ENC.w + 18},${ENC.y + ENC.h / 2} L${PROJ.x - 4},${ENC.y + ENC.h / 2}`} k={full ? Math.max(0.3, seg(t, 4.4, 4.8)) : 1} />
      {box("proj", <>
        <rect x={PROJ.x} y={PROJ.y} width={PROJ.w} height={PROJ.h} rx={6} fill="var(--part-bridge)" fillOpacity={glow("proj") ? 0.34 : 0.18} stroke="var(--part-bridge)" strokeWidth={glow("proj") ? 2 : 1} />
        <text transform={`translate(${PROJ.x + PROJ.w / 2 + 4},${PROJ.y + PROJ.h / 2}) rotate(-90)`} textAnchor="middle" fontSize={12.5} fontWeight={650} fill="var(--fg)">{tr("projector", "projektör")}</text>
        <text x={PROJ.x + PROJ.w / 2} y={PROJ.y + PROJ.h + 16} textAnchor="middle" fontSize={11} fill="var(--fg-muted)">Linear</text>
      </>)}
      <Arrow d={`M${PROJ.x + PROJ.w + 4},${ENC.y + ENC.h / 2} L${COL.x - 10},${ENC.y + ENC.h / 2}`} k={full ? Math.max(0.3, projK) : 1} />

      {/* ---- Prompt and tokenizer ---- */}
      {box("prompt", <>
        <rect x={PR.x} y={PR.y} width={PR.w} height={PR.h} rx={8} fill="var(--bg-elev)" stroke="var(--border-strong)" />
        <text x={PR.x + PR.w / 2} y={PR.y + 36} textAnchor="middle" fontSize={17} fontFamily="var(--mono)" fill="var(--fg)">"{PROMPT}"</text>
        <text x={PR.x + PR.w / 2} y={PR.y + PR.h + 18} textAnchor="middle" fontSize={12} fill="var(--fg-muted)">{tr("prompt", "komut (prompt)")}</text>
      </>)}
      <Arrow d={`M${PR.x + PR.w + 6},${PR.y + PR.h / 2} L${TOKZ.x - 6},${TOKZ.y + TOKZ.h / 2}`} k={full ? Math.max(0.3, promptK) : 1} />
      {box("tokenizer", <>
        <rect x={TOKZ.x} y={TOKZ.y} width={TOKZ.w} height={TOKZ.h} rx={8} fill="var(--part-lang)" fillOpacity={glow("tokenizer") ? 0.2 : 0.08} stroke="var(--part-lang)" strokeWidth={glow("tokenizer") ? 2 : 1} />
        <text x={TOKZ.x + TOKZ.w / 2} y={TOKZ.y + 27} textAnchor="middle" fontSize={14} fontWeight={650} fill="var(--fg)">Tokenizer</text>
        <text x={TOKZ.x + TOKZ.w / 2} y={TOKZ.y + 45} textAnchor="middle" fontSize={11.5} fill="var(--fg-muted)">SentencePiece + embedding</text>
      </>)}
      <Arrow d={`M${TOKZ.x + TOKZ.w + 6},${TOKZ.y + TOKZ.h / 2} L${COL.x - 10},${TOKZ.y + TOKZ.h / 2}`} k={full ? Math.max(0.3, tokzK) : 1} />

      {/* ---- Token column ---- */}
      <rect x={COL.x - 7} y={colTop} width={COL.sq + 14} height={colBot - colTop} rx={7} fill="var(--bg-sunk)" stroke={gemK > 0 && gemK < 1 ? "var(--accent)" : "var(--border)"} />
      {Array.from({ length: IMG_SLOTS }, (_, i) => {
        const k = clamp(imgTok * 1.6 - i * 0.09);
        return <g key={i} opacity={k}>
          {i === 4
            ? <text x={COL.x + COL.sq / 2} y={imgY(i) + 13} textAnchor="middle" fontSize={14} fill="var(--fg-faint)">⋮</text>
            : <rect x={COL.x} y={imgY(i)} width={COL.sq} height={COL.sq} rx={3} fill="var(--part-bridge)" fillOpacity={0.75} />}
        </g>;
      })}
      <text x={COL.x + COL.sq + 14} y={imgY(6) + 12} fontSize={11.5} fill="var(--fg-muted)" opacity={imgTok}>{tr("256 image", "256 görsel")}</text>
      {PIECES.map((p, i) => {
        const k = clamp(txtTok * 1.5 - i * 0.2);
        return <g key={p} opacity={k}>
          <rect x={COL.x} y={txtY(i)} width={COL.sq} height={COL.sq} rx={3} fill="var(--part-lang)" fillOpacity={0.7} />
          <text x={COL.x + COL.sq + 8} y={txtY(i) + 12} fontSize={11} fontFamily="var(--mono)" fill="var(--fg-muted)">{p}</text>
        </g>;
      })}
      {OUT.slice(0, Math.max(0, nOut - 1)).map((p, i) => {
        const k = full ? seg(t, OUT_T(i) + 0.55, OUT_T(i) + 1.25) : 1;
        return <g key={p} opacity={k}>
          <rect x={COL.x} y={txtY(PIECES.length + i)} width={COL.sq} height={COL.sq} rx={3} fill="var(--accent)" fillOpacity={0.75} />
          <text x={COL.x + COL.sq + 8} y={txtY(PIECES.length + i) + 12} fontSize={11} fontFamily="var(--mono)" fill="var(--accent)">{p}</text>
        </g>;
      })}

      {/* ---- Gemma ---- */}
      <Arrow d={`M${COL.x + COL.sq + 58},${GEM.y + GEM.h / 2} L${GEM.x - 4},${GEM.y + GEM.h / 2}`} k={full ? Math.max(0.3, gemK) : 1} flow={full ? clamp((t - 9.2) / 1.6) : 0} />
      {box("gemma", <>
        <rect x={GEM.x - 18} y={16} width={GEM.w + 36} height={330} rx={10} fill="var(--part-lang-soft)" stroke="var(--part-lang)" strokeDasharray="5 5" strokeWidth={1.2} />
        <rect x={GEM.x} y={GEM.y} width={GEM.w} height={GEM.h} rx={8} fill="var(--part-lang)" fillOpacity={glow("gemma") ? 0.28 : 0.15} stroke="var(--part-lang)" strokeWidth={glow("gemma") ? 2 : 1} />
        <text x={GEM.x + GEM.w / 2} y={GEM.y + 62} textAnchor="middle" fontSize={15} fontWeight={650} fill="var(--fg)">Transformer decoder</text>
        <text x={GEM.x + GEM.w / 2} y={GEM.y + 82} textAnchor="middle" fontSize={12} fill="var(--fg-muted)">{tr("18 layers · 2048 dims", "18 katman · 2048 boyut")}</text>
        <text x={GEM.x - 14} y={338} fontSize={12.5} fontWeight={700} fill="var(--part-lang)">Gemma <tspan fontWeight={500} fill="var(--fg-muted)">{tr("· 2B language model", "· 2B dil modeli")}</tspan></text>
        {full && <Mask x={GEM.x + GEM.w / 2 - 48} y={212} k={maskK} nOut={nOut} />}
      </>)}

      {/* ---- Output ---- */}
      <Arrow d={`M${GEM.x + GEM.w + 22},${GEM.y + GEM.h / 2} L${OUTX - 8},${GEM.y + GEM.h / 2}`} k={full ? Math.max(0.3, seg(t, 11.4, 11.9)) : 1} />
      {box("output", <>
        <text x={OUTX} y={34} fontSize={12} fill="var(--fg-muted)">{tr("answer", "cevap")}</text>
        {OUT.map((p, k) => {
          const a = full ? seg(t, OUT_T(k), OUT_T(k) + 0.45) : 1;
          const y = GEM.y + 22 + k * 34;
          return <g key={p} opacity={a} transform={`translate(${OUTX + (1 - a) * -14},${y})`}>
            <rect width={p.length * 9.5 + 22} height={26} rx={6} fill="var(--accent-soft)" stroke="var(--accent)" />
            <text x={11} y={17.5} fontSize={13.5} fontFamily="var(--mono)" fontWeight={600} fill="var(--accent)">{p}</text>
          </g>;
        })}
      </>)}

      {/* Feedback: every new token goes to the end of the column */}
      {full && <>
      <path d={`M${OUTX + 20},${GEM.y + 166} L${OUTX + 20},372 L${COL.x + COL.sq / 2},372 L${COL.x + COL.sq / 2},${colBot + 4}`} fill="none" stroke="var(--accent)" strokeOpacity={0.45} strokeDasharray="4 4" markerEnd="url(#arch-arrow)" opacity={full ? seg(t, 11.9, 12.4) : 1} />
      <text x={(OUTX + COL.x) / 2 + 40} y={390} textAnchor="middle" fontSize={11.5} fill="var(--accent)" opacity={full ? seg(t, 11.9, 12.4) : 1}>{tr("each new token is appended to the sequence and fed back to the model", "her yeni token dizinin sonuna eklenip modele geri veriliyor")}</text>
      </>}

      {/* ---- Flying pieces (full mode only) ---- */}
      {full && <Flyers t={t} />}
    </svg>
  );

  if (!full) {
    return (
      <div className="arch-mini" aria-label={tr("Where this chapter sits in the model", "Bu bölüm modelin neresinde")}>
        <div className="arch-mini-label">{tr("where this chapter sits in the model", "bu bölüm modelin neresinde")}</div>
        {svg}
      </div>
    );
  }

  const p = PHASES[phase];
  const done = t >= END - 0.01;
  return (
    <div ref={ref} className="arch-full">
      <div className="arch-stage">{svg}</div>
      <div className="arch-caption">
        <div className="arch-caption-text" key={phase}>
          <span className="arch-step-n">{phase + 1}</span>
          <div>
            <b>{p.title[lang]}.</b> {p.text[lang]}
            {p.shape && <div className="mono small muted" style={{ marginTop: 4 }}>{p.shape[lang]}</div>}
          </div>
        </div>
        <div className="arch-controls">
          <div className="arch-dots" role="tablist">
            {PHASES.map((q, i) => (
              <button key={i} role="tab" aria-selected={i === phase} className={(i === phase ? "on" : "") + (t > q.at ? " done" : "")} onClick={() => { setT(q.at + 0.001); setPlaying(true); }} title={q.title[lang]}>
                <i style={{ width: `${i === phase && !done ? clamp((t - q.at) / ((PHASES[i + 1]?.at ?? END) - q.at)) * 100 : t > q.at ? 100 : 0}%` }} />
                <span>{q.title[lang]}</span>
              </button>
            ))}
          </div>
          <button className="btn" onClick={() => { if (done) { setT(0); setPlaying(true); } else setPlaying(!playing); }}>
            {done ? tr("↻ Replay", "↻ Baştan") : playing ? tr("❚❚ Pause", "❚❚ Durdur") : tr("▶ Play", "▶ Oynat")}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Prefix-LM mask: rows look, columns are looked at. Full within the prefix, causal for the answer. */
function Mask({ x, y, k, nOut }: { x: number; y: number; k: number; nOut: number }) {
  const tr = useT();
  const P = 5, n = P + Math.max(1, nOut), c = 96 / (P + OUT.length);
  return (
    <g opacity={k}>
      <text x={x + 48} y={y - 8} fontSize={11} fill="var(--fg-muted)" textAnchor="middle">{tr("who sees whom", "kim kimi görüyor")}</text>
      {Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
        const see = j < P ? true : j <= i;
        const kind = i < P ? "prefix" : "suffix";
        return <rect key={i + "-" + j} x={x + j * c} y={y + i * c} width={c - 1.5} height={c - 1.5} rx={1.5}
          fill={!see ? "var(--bg-sunk)" : kind === "prefix" ? "var(--part-lang)" : "var(--accent)"} fillOpacity={see ? 0.6 : 1} />;
      }))}
      <text x={x + (P * c) / 2} y={y + (P + OUT.length) * c + 13} fontSize={10.5} textAnchor="middle" fill="var(--part-lang)">prefix</text>
      <text x={x + P * c + (OUT.length * c) / 2} y={y + (P + OUT.length) * c + 13} fontSize={10.5} textAnchor="middle" fill="var(--accent)">cevap</text>
    </g>
  );
}

/** Pieces that move during the animation: patches, vectors, prompt pieces, tokens fed back */
function Flyers({ t }: { t: number }) {
  const items: ReactNode[] = [];
  const cell = IM.s / 16;
  // 1) patches from the image to SigLIP
  FLY.forEach(([r, c], i) => {
    const k = seg(t, 2.2 + i * 0.12, 3.6 + i * 0.12);
    if (k <= 0 || k >= 1) return;
    const x0 = IM.x + (c + 0.5) * cell, y0 = IM.y + (r + 0.5) * cell;
    const x1 = ENC.x + 30 + (i % 4) * 34, y1 = ENC.y + 30 + Math.floor(i / 4) * 52;
    const lift = Math.sin(k * Math.PI) * -22;
    items.push(<Patch key={"p" + i} r={r} c={c} x={lerp(x0, x1, k)} y={lerp(y0, y1, k) + lift} s={lerp(cell, 20, Math.sin(k * Math.PI) * 0.8 + k * 0.2)} opacity={k < 0.85 ? 1 : (1 - k) / 0.15} />);
  });
  // 2) vectors from SigLIP through the projector into the column
  for (let i = 0; i < IMG_SLOTS; i++) {
    if (i === 4) continue;
    const k = seg(t, 4.6 + i * 0.1, 6.0 + i * 0.1);
    if (k <= 0 || k >= 1) continue;
    const x = lerp(ENC.x + ENC.w - 10, COL.x + COL.sq / 2, k);
    const y = lerp(ENC.y + ENC.h / 2, imgY(i) + COL.sq / 2, ease(clamp((k - 0.5) * 2)));
    const pastProj = x > PROJ.x + PROJ.w / 2;
    items.push(<rect key={"v" + i} x={x - 7} y={y - 7} width={14} height={14} rx={3} fill={pastProj ? "var(--part-bridge)" : "var(--part-vision)"} opacity={0.9} />);
  }
  // 3) prompt pieces: from the prompt card to the tokenizer, then into the column
  PIECES.forEach((p, i) => {
    const a = seg(t, 6.9 + i * 0.15, 7.7 + i * 0.15);
    const b = seg(t, 7.7 + i * 0.15, 8.9 + i * 0.15);
    if (a <= 0 || b >= 1) return;
    let x: number, y: number;
    if (b <= 0) { x = lerp(PR.x + 30 + i * 32, TOKZ.x + 40 + i * 42, a); y = lerp(PR.y - 14, TOKZ.y + TOKZ.h / 2, a); }
    else { x = lerp(TOKZ.x + 40 + i * 42, COL.x + COL.sq / 2, b); y = lerp(TOKZ.y + TOKZ.h / 2, txtY(i) + COL.sq / 2, b); }
    const chip = b <= 0;
    items.push(chip
      ? <g key={"t" + i} transform={`translate(${x},${y})`}><rect x={-20} y={-10} width={40} height={20} rx={4} fill="var(--bg-elev)" stroke="var(--part-lang)" /><text textAnchor="middle" y={4} fontSize={10.5} fontFamily="var(--mono)" fill="var(--part-lang)">{p}</text></g>
      : <rect key={"t" + i} x={x - 7} y={y - 7} width={14} height={14} rx={3} fill="var(--part-lang)" opacity={0.85} />);
  });
  // 4) the sequence flows into Gemma
  for (let i = 0; i < 6; i++) {
    const k = seg(t, 9.3 + i * 0.12, 10.3 + i * 0.12);
    if (k <= 0 || k >= 1) continue;
    const y0 = i < 3 ? imgY(i * 2) : txtY(i - 3);
    items.push(<circle key={"g" + i} cx={lerp(COL.x + COL.sq, GEM.x + 40 + i * 22, k)} cy={lerp(y0 + 8, GEM.y + 30, k)} r={4} fill={i < 3 ? "var(--part-bridge)" : "var(--part-lang)"} opacity={1 - k * 0.6} />);
  }
  // 5) generated tokens return to the end of the column
  OUT.slice(0, -1).forEach((_, k) => {
    const a = seg(t, OUT_T(k) + 0.45, OUT_T(k) + 1.25);
    if (a <= 0 || a >= 1) return;
    const pts: [number, number][] = [[OUTX + 20, GEM.y + 166], [OUTX + 20, 372], [COL.x + COL.sq / 2, 372], [COL.x + COL.sq / 2, txtY(PIECES.length + k) + 8]];
    const lens = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
    let d = a * lens.reduce((s, l) => s + l, 0), i = 0;
    while (i < lens.length - 1 && d > lens[i]) { d -= lens[i]; i++; }
    const q = d / lens[i];
    items.push(<circle key={"o" + k} cx={lerp(pts[i][0], pts[i + 1][0], q)} cy={lerp(pts[i][1], pts[i + 1][1], q)} r={5} fill="var(--accent)" />);
  });
  return <g style={{ pointerEvents: "none" }}>{items}</g>;
}
