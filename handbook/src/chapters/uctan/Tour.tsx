// End to end: the guided tour (scrolly) and free mode. Stage3D + instanced token columns + textured patches.
import { ReactNode, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useFrame, useLoader, ThreeEvent } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useData } from "../../lib/data";
import { lineOf, SOURCES } from "../../lib/pysource";
import { Stage3D, Cubes, Cube, Tag, View, cssColor, easeInOut } from "../../interact/three";
import { Loading, S, Seg } from "../../components/ui";
import { useStore } from "../../store";
import { useLang, useT } from "../../i18n";
import { DIMS, N, NP, STAGES, idx } from "./stages";

const BASE = import.meta.env.BASE_URL + "data/";
const TW = 0.12;
const SHOW = (p: string) => p.replace("▁", "␣").replace("\n", "\\n");

export interface Prompt { prompt: string; length: number; text_ids: number[]; text_pieces: string[] }

/** The 14×14 pixels of a patch: cropped from the 224 px image with CSS */
function PatchThumb({ img, r, c, size = 84 }: { img: string; r: number; c: number; size?: number }) {
  return <div style={{ width: size, height: size, backgroundImage: `url(${BASE}img_${img}_224.png)`, backgroundSize: `${size * N}px ${size * N}px`, backgroundPosition: `-${c * size}px -${r * size}px`, imageRendering: "pixelated", borderRadius: 3, outline: "1px solid var(--border)" }} />;
}

const XAXIS = new THREE.Vector3(1, 0, 0);
const PATCH_VS = /* glsl */ `
attribute vec2 aOff;
varying vec2 vUv;
void main() {
  vUv = aOff + uv / 16.0;
  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
}`;
const PATCH_FS = /* glsl */ `
uniform sampler2D map;
varying vec2 vUv;
void main() {
  gl_FragColor = texture2D(map, vUv);
  #include <colorspace_fragment>
}`;

/** 256 yama tek bir InstancedMesh: her örnek dokunun kendi 1/16 × 1/16 dilimini gösteriyor.
 *  Durak değişince her yama bulunduğu yerden hedefine süreli (easeInOut) bir geçiş yapıyor, sırayla (dalga). */
function Patches({ s, image, place }: { s: number; image: string; place: (i: number, s: number) => [[number, number, number], number, number] }) {
  const t = useT();
  const tex = useLoader(THREE.TextureLoader, `${BASE}img_${image}_224.png`);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(1, 1);
    const off = new Float32Array(NP * 2);
    for (let i = 0; i < NP; i++) { off[2 * i] = (i % N) / N; off[2 * i + 1] = 1 - (Math.floor(i / N) + 1) / N; }
    g.setAttribute("aOff", new THREE.InstancedBufferAttribute(off, 2));
    return g;
  }, []);
  const mat = useMemo(() => new THREE.ShaderMaterial({ uniforms: { map: { value: tex } }, vertexShader: PATCH_VS, fragmentShader: PATCH_FS, side: THREE.DoubleSide }), [tex]);
  const ref = useRef<THREE.InstancedMesh>(null);
  const st = useMemo(() => {
    const cur = Array.from({ length: NP }, (_, i) => { const [p, sc, fl] = place(i, s); return { p: new THREE.Vector3(...p), sc, fl }; });
    return { cur, from: cur.map((c) => ({ ...c, p: c.p.clone() })), to: cur.map((c) => ({ ...c, p: c.p.clone() })), t0: new Float32Array(NP).fill(-1), stage: s };
  }, []);
  const [hover, setHover] = useState<number | null>(null);
  const m = useMemo(() => new THREE.Matrix4(), []), q = useMemo(() => new THREE.Quaternion(), []);
  const pv = useMemo(() => new THREE.Vector3(), []), sv = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ clock }) => {
    const mesh = ref.current;
    if (!mesh) return;
    const now = clock.elapsedTime;
    if (st.stage !== s) {
      // new stop: a row-by-row wave when entering/leaving the token sequence, almost together otherwise
      const wave = (s >= idx("tokens")) !== (st.stage >= idx("tokens"));
      st.stage = s;
      for (let i = 0; i < NP; i++) {
        const [p, sc, fl] = place(i, s);
        st.from[i].p.copy(st.cur[i].p); st.from[i].sc = st.cur[i].sc; st.from[i].fl = st.cur[i].fl;
        st.to[i].p.set(...p); st.to[i].sc = sc; st.to[i].fl = fl;
        st.t0[i] = now + (wave ? (i / NP) * 0.9 : (i / NP) * 0.15);
      }
    }
    for (let i = 0; i < NP; i++) {
      const c = st.cur[i];
      const k = st.t0[i] < 0 ? 1 : easeInOut(Math.min(1, Math.max(0, (now - st.t0[i]) / 1.0)));
      c.p.lerpVectors(st.from[i].p, st.to[i].p, k);
      // patches rise slightly on the way so it is clear they "fly" from one place to another
      const lift = Math.sin(k * Math.PI) * (st.from[i].p.distanceTo(st.to[i].p) > 1 ? 0.35 : 0);
      c.sc = st.from[i].sc + (st.to[i].sc - st.from[i].sc) * k;
      c.fl = st.from[i].fl + (st.to[i].fl - st.from[i].fl) * k;
      q.setFromAxisAngle(XAXIS, -c.fl * Math.PI / 2);
      const h = hover === i ? 1.6 : 1;
      pv.set(c.p.x, c.p.y + lift, c.p.z + (hover === i ? 0.02 : 0));
      sv.set(c.sc * h, c.sc * h, 1);
      m.compose(pv, q, sv);
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  });
  const hp = hover !== null ? { r: Math.floor(hover / N), c: hover % N, p: st.cur[hover].p.toArray() as [number, number, number] } : null;
  return (
    <group>
      <instancedMesh ref={ref} args={[geo, mat, NP]} frustumCulled={false}
        onPointerMove={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); if (e.instanceId !== undefined && e.instanceId !== hover) setHover(e.instanceId); }}
        onPointerOut={() => setHover(null)} />
      {hp && s >= idx("pre") && (
        <Html position={hp.p} style={{ pointerEvents: "none" }} zIndexRange={[30, 0]}>
          <div className="tip" style={{ transform: "translate(14px, -50%)", display: "flex", gap: 10, alignItems: "center" }}>
            <PatchThumb img={image} r={hp.r} c={hp.c} />
            <div style={{ whiteSpace: "nowrap" }}>
              <div>{t("patch", "yama")} ({hp.r}, {hp.c}) → <b>token {hp.r * N + hp.c}</b></div>
              <div className="muted">14 × 14 × 3 = 588 {t("pixel values", "piksel değeri")}</div>
              <div className="muted">{s >= idx("tokens") ? t(`position ${hp.r * N + hp.c} in the sequence`, `dizide ${hp.r * N + hp.c}. konum`) : t("Conv2d turns it into 1152 numbers", "Conv2d bunu 1152 sayıya çeviriyor")}</div>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

/** Dimension line: ticks at both ends, a label in the middle */
function Dim({ a, b, label, color, labelAt }: { a: [number, number, number]; b: [number, number, number]; label: ReactNode; color: string; labelAt?: [number, number, number] }) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
  const mid = va.clone().add(vb).multiplyScalar(0.5);
  const len = va.distanceTo(vb);
  const dir = vb.clone().sub(va).normalize();
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);
  const col = cssColor(color.slice(4, -1));
  return (
    <group>
      <mesh position={mid} quaternion={quat}><boxGeometry args={[len, 0.012, 0.012]} /><meshBasicMaterial color={col} /></mesh>
      {[va, vb].map((v, i) => <mesh key={i} position={v} quaternion={quat}><boxGeometry args={[0.012, 0.12, 0.012]} /><meshBasicMaterial color={col} /></mesh>)}
      <Tag position={labelAt ?? [mid.x, mid.y, mid.z]} size={12} weight={650} color={color}>{label}</Tag>
    </group>
  );
}

// Column height: proportional to the dim (the vocabulary size is cropped because it does not fit)
const COLH = { 1152: 0.7, 2048: 1.25, 257216: 2.2 } as const;

export function JourneyScene({ stage: s, image, prompt, height = 480 }: { stage: number; image: string; prompt: Prompt; height?: number }) {
  const st = STAGES[s];
  const t = useT();
  const nText = prompt.text_ids.length;
  const T = NP + nText;
  const x0 = -(NP * TW) / 2;                    // the first token sits a little left of the scene center
  const dimNow = s >= idx("head") ? 257216 : s >= idx("proj") ? 2048 : 1152;
  const colH = COLH[dimNow];
  const rowZ = st.z;
  const tokensOn = s >= idx("tokens"), textOn = s >= idx("merge");
  const Tc = cssColor("--ax-T"), Ic = cssColor("--ax-I"), Vc = cssColor("--ax-V");
  const imgColor = s >= idx("head") ? Vc : s >= idx("proj") ? Ic : Tc;
  const xOf = (i: number) => x0 + (i + 0.5) * TW;
  // Patch placement: an upright grid at the image stops, a flat tile on top of its column at the token stops
  const place = (i: number, ss: number): [[number, number, number], number, number] => {
    const r = Math.floor(i / N), c = i % N;
    if (ss <= idx("pre")) { const size = ss === 0 ? 3.4 : 3.0; return [[-size / 2 + (c + 0.5) * (size / N), 1.7 + size / 2 - (r + 0.5) * (size / N), 0], size / N, 0]; }
    if (ss === idx("patch")) { const g = 0.22; return [[-(N * g) / 2 + (c + 0.5) * g, 1.75 + (N * g) / 2 - (r + 0.5) * g, 0], 0.19, 0]; }
    const h = COLH[ss >= idx("head") ? 257216 : ss >= idx("proj") ? 2048 : 1152];
    return [[xOf(i), h + 0.012, STAGES[ss].z], TW * 0.94, 1];
  };
  const items: Cube[] = [];
  for (let i = 0; i < T; i++) {
    const isImg = i < NP;
    const on = isImg ? tokensOn : textOn;
    const piece = isImg ? "" : prompt.text_pieces[i - NP];
    items.push({
      id: "t" + i, pos: [xOf(i), colH / 2, on ? rowZ : rowZ + (isImg ? 0 : 1.5)],
      size: [TW * 0.8, colH, TW * 0.8], color: isImg ? imgColor : Vc, hidden: !on,
      tip: () => isImg
        ? <div style={{ display: "flex", gap: 10, alignItems: "center" }}><PatchThumb img={image} r={Math.floor(i / N)} c={i % N} size={56} /><div><b>token {i}</b> · {t("image", "görsel")}<div className="muted">{t("patch", "yama")} ({Math.floor(i / N)}, {i % N})</div><div className="mono">{st.shape(DIMS.real, T).slice(-1)[0]} {t("numbers", "sayı")}</div></div></div>
        : <><b>token {i}</b> · {t("text", "metin")}: <span className="mono">{SHOW(piece)}</span><div className="muted">id {prompt.text_ids[i - NP]}{piece === "<bos>" ? t(" · start of the sequence", " · dizinin başı") : piece === "\n" ? t(" · end-of-prefix marker", " · prefix bitti işareti") : ""}</div></>,
    });
  }
  const xEnd = x0 + T * TW;
  const W = T * TW + 0.6;
  const block = (z: number, h: number, color: string, active: boolean, label: string) => (
    <group>
      <mesh position={[(x0 + xEnd) / 2, h / 2, z]}>
        <boxGeometry args={[W, h, 0.9]} />
        <meshStandardMaterial color={color} transparent opacity={active ? 0.1 : 0.035} depthWrite={false} />
      </mesh>
      {active && <Tag position={[x0 - 0.3, h + 0.25, z]} size={12.5} weight={650} color="var(--fg)">{label}</Tag>}
    </group>
  );
  const near = (z: number) => tokensOn && z - rowZ < 2.6 && z - rowZ > -1.2;
  // Camera: a 3/4 view from the start of the sequence; columns shrink with distance (reading left to right)
  const front: View = { pos: [x0 - 1.0, colH + 1.25, rowZ + 2.3], target: [x0 + 1.3, colH * 0.3, rowZ] };
  const view: View = !tokensOn
    ? { pos: [1.2, 1.9, 6.4], target: [0, 1.7, 0] }
    : (s === idx("merge") || s === idx("head"))
      ? { pos: [xEnd + 1.0, colH + 1.25, rowZ + 2.3], target: [xEnd - 1.3, colH * 0.3, rowZ] }
      : front;
  const atEnd = s === idx("merge") || s === idx("head");
  return (
    <Stage3D view={view} height={height} minDist={1.5} maxDist={60}>
      <Patches s={s} image={image} place={place} />
      <Cubes items={items} tween={{ duration: 0.8, stagger: 0.0025 }} />
      {near(-3.2) && block(-3.2, 1.8, Tc, s === idx("enc"), "SigLIP encoder × 27")}
      {near(-5.6) && block(-5.6, 2.6, Ic, s === idx("proj"), t("projector", "projektör"))}
      {near(-7.6) && block(-7.6, 2.6, Vc, s === idx("merge"), t("merge", "birleştirme"))}
      {near(-10.6) && block(-10.6, 2.6, cssColor("--ax-D"), s === idx("dec"), "Gemma decoder × 18")}
      {near(-13) && block(-13, 2.6, cssColor("--ax-hd"), s === idx("norm"), t("final RMSNorm", "son RMSNorm"))}
      {near(-15) && block(-15, 3.8, Vc, s === idx("head"), "lm_head")}
      {tokensOn && !atEnd && <>
        <Dim a={[x0, -0.06, rowZ + TW]} b={[x0 + NP * TW, -0.06, rowZ + TW]} color={s >= idx("proj") ? "var(--ax-I)" : "var(--ax-T)"} label={t("256 image tokens →", "256 görsel token →")} labelAt={[x0 + 1.4, -0.2, rowZ + TW]} />
        {[0, 1, 2, 3].map((i) => <Tag key={i} position={[xOf(i), colH + 0.16, rowZ]} size={10.5} color="var(--fg-muted)">{i}</Tag>)}
        <Dim a={[x0 - 0.08, 0, rowZ + TW]} b={[x0 - 0.08, colH, rowZ + TW]} color="var(--ax-D)" label={`${dimNow} ${t("dims", "boyut")}`} labelAt={[x0 - 0.42, colH / 2, rowZ + TW]} />
      </>}
      {tokensOn && atEnd && <Dim a={[xEnd + 0.1, 0, rowZ + TW]} b={[xEnd + 0.1, colH, rowZ + TW]} color="var(--ax-D)" label={dimNow === 257216 ? t("257216 (cropped)", "257216 (kırpılmış)") : `${dimNow} ${t("dims", "boyut")}`} labelAt={[xEnd + 0.5, colH / 2, rowZ + TW]} />}
      {textOn && <Tag position={[x0 + (NP + nText / 2) * TW, colH + 0.5, rowZ]} size={12} color="var(--ax-V)">{prompt.text_pieces.map(SHOW).join(" · ")}</Tag>}
      {s === idx("pre") && <Tag position={[0, -0.05, 0]} size={12} color="var(--fg-muted)">224 × 224 · {t("values", "değerler")} [−1, 1]</Tag>}
      {s === idx("patch") && <Tag position={[0, -0.12, 0]} size={12} color="var(--fg-muted)">{t("16 × 16 patches · hover", "16 × 16 yama · üstüne gel")}</Tag>}
      {s === idx("head") && <Tag position={[xEnd - TW / 2 - 0.3, colH + 0.28, rowZ]} size={12} weight={700} color="var(--accent)">{t("last position → next token", "son konum → bir sonraki token")}</Tag>}
    </Stage3D>
  );
}

/** Image and prompt picker (shared through the store) */
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
  const t = useT();
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
      <span className="muted">{t("shape", "şekil")}</span><span><S s={shape.join(",")} ax={st.ax} /> <span className="muted mono" style={{ marginLeft: 8 }}>{st.dtype} · {mb < 1 ? (mb * 1024).toFixed(0) + " KB" : mb.toFixed(1) + " MB"}</span></span>
      <span className="muted">{t("class", "sınıf")}</span><span className="mono" style={{ color: "var(--ax-I)" }}>{st.cls}</span>
      {line && <><span className="muted">{t("code", "kod")}</span><span className="mono" style={{ fontSize: 12.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}><span className="muted">{st.code![0]}:{ln}</span> {line}</span></>}
    </div>
  );
}

/** The scrolly figure */
export function TourFigure({ step }: { step: number }) {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx, promptIdx } = useStore();
  const t = useT();
  if (!data) return <Loading what={t("images", "görüntüler")} error={error} />;
  const prompt = data.prompts[promptIdx];
  return (
    <div>
      <Pickers data={data} />
      <JourneyScene stage={step} image={data.images[imageIdx].name} prompt={prompt} height={typeof window !== "undefined" && window.innerWidth < 900 ? 250 : 440} />
      <InfoLine s={step} prompt={prompt} data={data} />
    </div>
  );
}

/** Free mode: slider and play */
export function FreePlay() {
  const { data, error } = useData<any>("pipeline");
  const { imageIdx, promptIdx } = useStore();
  const [s, setS] = useState(idx("tokens"));
  const lang = useLang();
  const t = useT();
  if (!data) return <Loading what={t("images", "görüntüler")} error={error} />;
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
        <span style={{ minWidth: 230, fontWeight: 600 }}>{s + 1}. {st.title[lang]}</span>
      </div>
      <InfoLine s={s} prompt={prompt} data={data} />
      <div style={{ marginTop: 8 }}><Link to={st.to}>{t("Go to this step's chapter →", "Bu adımın bölümüne git →")}</Link></div>
    </div>
  );
}
