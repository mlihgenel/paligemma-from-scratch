// Oyun alanı v2: rehberli tur (scrolly) ve serbest mod. Stage3D + instanced token sütunları + dokulu yamalar.
import { ReactNode, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useFrame, useLoader, ThreeEvent } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { easing } from "maath";
import * as THREE from "three";
import { useData } from "../../lib/data";
import { lineOf, SOURCES } from "../../lib/pysource";
import { Stage3D, Cubes, Cube, Tag, View, cssColor } from "../../interact/three";
import { Loading, S, Seg } from "../../components/ui";
import { useStore } from "../../store";
import { DIMS, N, NP, STAGES, idx } from "./stages";

const BASE = import.meta.env.BASE_URL + "data/";
const TW = 0.068;
const SHOW = (p: string) => p.replace("▁", "␣").replace("\n", "\\n");

export interface Prompt { prompt: string; length: number; text_ids: number[]; text_pieces: string[] }

/** Yamanın 14×14 pikseli: 224'lük görüntüden CSS ile kırpılır */
function PatchThumb({ img, r, c, size = 84 }: { img: string; r: number; c: number; size?: number }) {
  return <div style={{ width: size, height: size, backgroundImage: `url(${BASE}img_${img}_224.png)`, backgroundSize: `${size * N}px ${size * N}px`, backgroundPosition: `-${c * size}px -${r * size}px`, imageRendering: "pixelated", borderRadius: 3, outline: "1px solid var(--border)" }} />;
}

function Patches({ s, image, x0, colH, rowZ }: { s: number; image: string; x0: number; colH: number; rowZ: number }) {
  const tex = useLoader(THREE.TextureLoader, `${BASE}img_${image}_224.png`);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  const geos = useMemo(() => Array.from({ length: NP }, (_, i) => {
    const r = Math.floor(i / N), c = i % N;
    const g = new THREE.PlaneGeometry(1, 1);
    const u0 = c / N, u1 = (c + 1) / N, v1 = 1 - r / N, v0 = 1 - (r + 1) / N;
    g.setAttribute("uv", new THREE.Float32BufferAttribute([u0, v1, u1, v1, u0, v0, u1, v0], 2));
    return g;
  }), []);
  const meshes = useRef<(THREE.Mesh | null)[]>([]);
  const [hover, setHover] = useState<number | null>(null);
  const tgt = useMemo(() => new THREE.Vector3(), []);
  const tsc = useMemo(() => new THREE.Vector3(), []);
  const place = (i: number): [[number, number, number], number] => {
    const r = Math.floor(i / N), c = i % N;
    if (s <= idx("pre")) { const size = s === 0 ? 3.4 : 3.0; return [[-size / 2 + (c + 0.5) * (size / N), 1.7 + size / 2 - (r + 0.5) * (size / N), 0], size / N]; }
    if (s === idx("patch")) { const g = 0.22; return [[-(N * g) / 2 + (c + 0.5) * g, 1.75 + (N * g) / 2 - (r + 0.5) * g, 0], 0.19]; }
    return [[x0 + (i + 0.5) * TW, colH + 0.07, rowZ], 0.062];
  };
  useFrame((_, dt) => {
    meshes.current.forEach((m, i) => {
      if (!m) return;
      const [p, sc] = place(i);
      // dalga etkisi: yamalar sırayla yerine gitsin
      const lag = s === idx("tokens") ? 0.18 + (i / NP) * 0.5 : 0.25;
      tgt.set(...p); tsc.set(sc * (hover === i ? 1.25 : 1), sc * (hover === i ? 1.25 : 1), 1);
      easing.damp3(m.position, tgt, lag, dt);
      easing.damp3(m.scale, tsc, 0.15, dt);
    });
  });
  const h = hover !== null ? { r: Math.floor(hover / N), c: hover % N, p: place(hover)[0] } : null;
  return (
    <group>
      {geos.map((g, i) => (
        <mesh key={i} ref={(el) => (meshes.current[i] = el)} geometry={g}
          onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHover(i); }} onPointerOut={() => setHover(null)}>
          <meshBasicMaterial map={tex} side={THREE.DoubleSide} toneMapped={false} />
        </mesh>
      ))}
      {h && s >= idx("pre") && (
        <Html position={h.p} style={{ pointerEvents: "none" }} zIndexRange={[30, 0]}>
          <div className="tip" style={{ transform: "translate(14px, -50%)", display: "flex", gap: 10, alignItems: "center" }}>
            <PatchThumb img={image} r={h.r} c={h.c} />
            <div style={{ whiteSpace: "nowrap" }}>
              <div>yama ({h.r}, {h.c}) → <b>token {h.r * N + h.c}</b></div>
              <div className="muted">14 × 14 × 3 = 588 piksel değeri</div>
              <div className="muted">{s >= idx("tokens") ? "dizide " + (h.r * N + h.c) + ". konum" : "Conv2d bunu 1152 sayıya çeviriyor"}</div>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

export function JourneyScene({ stage: s, image, prompt, height = 480 }: { stage: number; image: string; prompt: Prompt; height?: number }) {
  const st = STAGES[s];
  const nText = prompt.text_ids.length;
  const T = NP + nText;
  const x0 = -(T * TW) / 2;
  const colH = s >= idx("head") ? 1.6 : s >= idx("proj") ? 0.95 : 0.8;
  const rowZ = st.z;
  const tokensOn = s >= idx("tokens"), textOn = s >= idx("merge");
  const Tc = cssColor("--ax-T"), Ic = cssColor("--ax-I"), Vc = cssColor("--ax-V");
  const imgColor = s >= idx("head") ? Vc : s >= idx("proj") ? Ic : Tc;
  const items: Cube[] = [];
  for (let i = 0; i < T; i++) {
    const isImg = i < NP;
    const on = isImg ? tokensOn : textOn;
    const piece = isImg ? "" : prompt.text_pieces[i - NP];
    items.push({
      id: "t" + i, pos: [x0 + (i + 0.5) * TW, on ? colH / 2 : (isImg ? colH / 2 : colH / 2), on ? rowZ : rowZ + (isImg ? 0 : 1.5)],
      size: [TW * 0.82, colH, 0.06], color: isImg ? imgColor : Vc, hidden: !on, dim: s === idx("div") && isImg,
      tip: () => isImg
        ? <div style={{ display: "flex", gap: 10, alignItems: "center" }}><PatchThumb img={image} r={Math.floor(i / N)} c={i % N} size={56} /><div><b>token {i}</b> · görsel<div className="muted">yama ({Math.floor(i / N)}, {i % N})</div><div className="mono">{st.shape(DIMS.real, T).slice(-1)[0]} sayı</div></div></div>
        : <><b>token {i}</b> · metin: <span className="mono">{SHOW(piece)}</span><div className="muted">id {prompt.text_ids[i - NP]}{piece === "<bos>" ? " · dizinin başı" : piece === "\n" ? " · prefix bitti işareti" : ""}</div></>,
    });
  }
  const W = T * TW + 0.6;
  const block = (z: number, h: number, color: string, active: boolean) => (
    <mesh position={[0, h / 2, z]}>
      <boxGeometry args={[W, h, z === -3.2 || z === -10.6 ? 1.0 : 0.04]} />
      <meshStandardMaterial color={color} transparent opacity={active ? 0.14 : 0.05} depthWrite={false} />
    </mesh>
  );
  const near = (z: number) => tokensOn && z - rowZ < 2.6 && z - rowZ > -1.2;
  const view: View = !tokensOn
    ? { pos: [1.2, 1.9, 6.4], target: [0, 1.7, 0] }
    : (s === idx("merge") || s === idx("scale"))
      ? { pos: [x0 + T * TW - 1.6, 1.6, rowZ + 4.2], target: [x0 + T * TW - 2.2, 0.6, rowZ] }
      : s === idx("head") ? { pos: [1.5, 3.6, rowZ + 16.0], target: [0, 0.8, rowZ] }
      : { pos: [0.8, 2.8, rowZ + 15.0], target: [0, 0.4, rowZ] };
  return (
    <Stage3D view={view} height={height} minDist={1.5} maxDist={40}>
      <Patches s={s} image={image} x0={x0} colH={colH} rowZ={rowZ} />
      <Cubes items={items} />
      {near(-3.2) && block(-3.2, 1.8, Tc, s === idx("enc"))}
      {near(-5.2) && block(-5.2, 1.6, Ic, s === idx("proj") || s === idx("div"))}
      {near(-7.2) && block(-7.2, 1.6, Vc, s === idx("merge") || s === idx("scale"))}
      {near(-10.6) && block(-10.6, 2.0, cssColor("--ax-D"), s === idx("dec"))}
      {near(-12.7) && block(-12.7, 1.6, cssColor("--ax-hd"), s === idx("norm"))}
      {near(-14.6) && block(-14.6, 2.6, Vc, s === idx("head"))}
      {tokensOn && <Tag position={[x0 - 0.5, colH / 2, rowZ]} size={11.5} color="var(--fg-muted)">{s >= idx("head") ? "257216" : s >= idx("proj") ? "2048" : "1152"}</Tag>}
      {tokensOn && <Tag position={[x0 + (NP * TW) / 2, -0.3, rowZ + 0.1]} size={12} color={s >= idx("proj") ? "var(--ax-I)" : "var(--ax-T)"}>256 görsel token</Tag>}
      {textOn && <Tag position={[x0 + (NP + nText / 2) * TW, colH + 0.42, rowZ]} size={12} color="var(--ax-V)">{prompt.text_pieces.map(SHOW).join(" · ")}</Tag>}
      {s === idx("pre") && <Tag position={[0, -0.05, 0]} size={12} color="var(--fg-muted)">224 × 224 · değerler [−1, 1]</Tag>}
      {s === idx("patch") && <Tag position={[0, -0.12, 0]} size={12} color="var(--fg-muted)">16 × 16 yama · üstüne gel</Tag>}
      {s === idx("head") && <Tag position={[x0 + (T - 0.5) * TW - 1.2, colH + 0.5, rowZ]} size={12} weight={700} color="var(--accent)">son konum → bir sonraki token</Tag>}
    </Stage3D>
  );
}

/** Görüntü ve prompt seçici (store'da paylaşılıyor) */
export function Pickers({ data }: { data: any }) {
  const { imageIdx, setImageIdx, promptIdx, setPromptIdx } = useStore();
  return (
    <div className="row" style={{ gap: 18, alignItems: "center", marginBottom: 10 }}>
      <div className="row" style={{ gap: 5 }}>
        {data.images.map((x: any, i: number) => (
          <button key={x.name} onClick={() => setImageIdx(i)} title={x.name} style={{ all: "unset", cursor: "pointer", borderRadius: 4, outline: i === imageIdx ? "2px solid var(--accent)" : "1px solid var(--border)", outlineOffset: 1, overflow: "hidden", width: 36, height: 36 }}>
            <img src={`${BASE}img_${x.name}.jpg`} alt={x.name} style={{ width: 36, height: 36, objectFit: "cover", display: "block" }} />
          </button>
        ))}
      </div>
      <select value={promptIdx} onChange={(e) => setPromptIdx(+e.target.value)} style={{ fontSize: 13.5, maxWidth: 260 }}>
        {data.prompts.map((p: Prompt, i: number) => <option key={i} value={i}>{p.prompt}</option>)}
      </select>
    </div>
  );
}

function InfoLine({ s, prompt, data }: { s: number; prompt: Prompt; data: any }) {
  const st = STAGES[s];
  const T = prompt.length;
  const im = data.images[useStore.getState().imageIdx];
  let shape = st.shape(DIMS.real, T);
  if (st.key === "raw") shape = [im.orig_size[1], im.orig_size[0], 3];
  const n = shape.reduce((a, b) => a * b, 1);
  const mb = (n * (st.dtype === "uint8" ? 1 : 4)) / 2 ** 20;
  const ln = st.code ? lineOf(st.code[0], st.code[1]) : -1;
  const line = st.code && ln > 0 ? SOURCES[st.code[0]].split("\n")[ln - 1].trim() : null;
  return (
    <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 14px", fontSize: 13.5, alignItems: "baseline" }}>
      <span className="muted">şekil</span><span><S s={shape.join(",")} ax={st.ax} /> <span className="muted mono" style={{ marginLeft: 8 }}>{st.dtype} · {mb < 1 ? (mb * 1024).toFixed(0) + " KB" : mb.toFixed(1) + " MB"}</span></span>
      <span className="muted">sınıf</span><span className="mono" style={{ color: "var(--ax-I)" }}>{st.cls}</span>
      {line && <><span className="muted">kod</span><span className="mono" style={{ fontSize: 12.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}><span className="muted">{st.code![0]}:{ln}</span> {line}</span></>}
    </div>
  );
}

/** Scrolly figürü */
export function TourFigure({ step }: { step: number }) {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx, promptIdx } = useStore();
  if (!data) return <Loading what="görüntüler" error={error} />;
  const prompt = data.prompts[promptIdx];
  return (
    <div>
      <Pickers data={data} />
      <JourneyScene stage={step} image={data.images[imageIdx].name} prompt={prompt} height={typeof window !== "undefined" && window.innerWidth < 900 ? 250 : 440} />
      <InfoLine s={step} prompt={prompt} data={data} />
    </div>
  );
}

/** Serbest mod: kaydırıcı ve oynat */
export function FreePlay() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx, promptIdx } = useStore();
  const [s, setS] = useState(idx("tokens"));
  if (!data) return <Loading what="görüntüler" error={error} />;
  const prompt = data.prompts[promptIdx];
  const st = STAGES[s];
  return (
    <div>
      <Pickers data={data} />
      <JourneyScene stage={s} image={data.images[imageIdx].name} prompt={prompt} height={520} />
      <div className="row" style={{ alignItems: "center", gap: 12, marginTop: 12 }}>
        <button className="btn" disabled={s === 0} onClick={() => setS(s - 1)}>←</button>
        <input type="range" min={0} max={STAGES.length - 1} value={s} onChange={(e) => setS(+e.target.value)} style={{ flex: 1 }} />
        <button className="btn" disabled={s === STAGES.length - 1} onClick={() => setS(s + 1)}>→</button>
        <span style={{ minWidth: 230, fontWeight: 600 }}>{s + 1}. {st.title}</span>
      </div>
      <InfoLine s={s} prompt={prompt} data={data} />
      <div style={{ marginTop: 8 }}><Link to={st.to}>Bu adımın bölümüne git →</Link></div>
    </div>
  );
}
