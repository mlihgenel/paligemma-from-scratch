// 3D v2: yönlendirilmiş kamera (CameraControls), instanced küpler, raycast hover, yumuşak (damped) geçişler.
import { ReactNode, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, ThreeEvent } from "@react-three/fiber";
import { CameraControls, Html } from "@react-three/drei";
import { easing } from "maath";
import * as THREE from "three";

export interface View { pos: [number, number, number]; target?: [number, number, number] }

export function Stage3D({ children, view, height = 420, minDist = 2, maxDist = 30 }: {
  children: ReactNode; view: View; height?: number | string; minDist?: number; maxDist?: number;
}) {
  const ref = useRef<CameraControls | null>(null);
  const [moved, setMoved] = useState(false);
  // CameraControls Canvas içinde geç bağlanıyor: ilk bağlandığında hedefe animasyonsuz kur
  const bind = (c: CameraControls | null) => {
    if (c && !ref.current) { const t = view.target ?? [0, 0, 0]; c.setLookAt(...view.pos, ...t, false); }
    ref.current = c;
  };
  const go = (smooth = true) => {
    const t = view.target ?? [0, 0, 0];
    ref.current?.setLookAt(...view.pos, ...t, smooth);
    setMoved(false);
  };
  useEffect(() => { go(true); }, [JSON.stringify(view)]);
  return (
    <div style={{ height, position: "relative", borderRadius: 6, overflow: "hidden", background: "var(--stage-bg)" }}>
      <Canvas camera={{ position: view.pos, fov: 40 }} dpr={[1, 2]} gl={{ antialias: true, preserveDrawingBuffer: true }}>
        <ambientLight intensity={0.9} />
        <directionalLight position={[4, 8, 6]} intensity={1.1} />
        <directionalLight position={[-6, -3, -4]} intensity={0.25} />
        <Suspense fallback={null}>{children}</Suspense>
        <CameraControls ref={bind} makeDefault minDistance={minDist} maxDistance={maxDist}
          minPolarAngle={0.15} maxPolarAngle={Math.PI / 2 + 0.15} smoothTime={0.45} draggingSmoothTime={0.12}
          onStart={() => setMoved(true)} />
      </Canvas>
      {moved && <button className="btn" onClick={() => go(true)} style={{ position: "absolute", top: 8, right: 8, background: "var(--bg-elev)", fontSize: 12.5 }}>↺ görünümü sıfırla</button>}
      <div style={{ position: "absolute", bottom: 8, right: 12, pointerEvents: "none", fontSize: 11.5, color: "var(--fg-faint)" }}>sürükle: döndür · kaydır: yakınlaş · üstüne gel: değer</div>
    </div>
  );
}

export interface Cube {
  id: string;
  pos: [number, number, number];
  size?: [number, number, number];
  color: string;
  dim?: boolean;          // bağlam (soluk)
  hidden?: boolean;
  tip?: () => ReactNode;
}

/** Tek InstancedMesh: id sırası sabit kalmalı. Hedefe yumuşak geçiş, hover'da tooltip ve vurgu */
export function Cubes({ items, cell = 0.42 }: { items: Cube[]; cell?: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const n = items.length;
  const cur = useMemo(() => items.map((it) => ({ p: new THREE.Vector3(...it.pos), s: new THREE.Vector3(...(it.size ?? [cell, cell, cell])), c: new THREE.Color(it.color) })), [n]);
  const [hover, setHover] = useState<number | null>(null);
  const m = useMemo(() => new THREE.Matrix4(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);
  const tmpC = useMemo(() => new THREE.Color(), []);
  const tgtP = useMemo(() => new THREE.Vector3(), []);
  const tgtS = useMemo(() => new THREE.Vector3(), []);

  const dimC = useMemo(() => new THREE.Color(cssColor("--stage-bg")), []);
  useFrame((_, dt) => {
    const mesh = ref.current;
    if (!mesh) return;
    items.forEach((it, i) => {
      const c = cur[i];
      tgtP.set(...it.pos);
      const sz = it.hidden ? [0.0001, 0.0001, 0.0001] : (it.size ?? [cell, cell, cell]);
      tgtS.set(sz[0], sz[1], sz[2]);
      easing.damp3(c.p, tgtP, 0.28, dt);
      easing.damp3(c.s, tgtS, 0.28, dt);
      tmpC.set(it.color);
      if (it.dim) tmpC.lerp(dimC, 0.72);
      if (hover === i) tmpC.offsetHSL(0, 0, 0.12);
      easing.dampC(c.c, tmpC, 0.18, dt);
      m.compose(c.p, q, c.s);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, c.c);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  const onMove = (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); if (e.instanceId !== undefined && !items[e.instanceId]?.hidden) setHover(e.instanceId); };
  const h = hover !== null ? items[hover] : null;
  return (
    <group>
      <instancedMesh ref={ref} args={[undefined, undefined, n]} onPointerMove={onMove} onPointerOut={() => setHover(null)}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.6} metalness={0.02} />
      </instancedMesh>
      {h?.tip && hover !== null && (
        <Html position={cur[hover].p.toArray() as any} style={{ pointerEvents: "none" }} zIndexRange={[20, 0]}>
          <div className="tip" style={{ transform: "translate(14px, -50%)", whiteSpace: "nowrap" }}>{h.tip()}</div>
        </Html>
      )}
    </group>
  );
}

/** Sahne içi etiket (DOM) */
export function Tag({ position, children, color, size = 12, weight = 500 }: { position: [number, number, number]; children: ReactNode; color?: string; size?: number; weight?: number }) {
  return (
    <Html position={position} center style={{ pointerEvents: "none", whiteSpace: "nowrap" }} zIndexRange={[10, 0]}>
      <div style={{ fontFamily: "var(--sans)", fontSize: size, fontWeight: weight, color: color ?? "var(--fg)", textShadow: "0 0 3px var(--stage-bg), 0 0 6px var(--stage-bg)" }}>{children}</div>
    </Html>
  );
}

/** CSS değişkenini oku (three materyalleri için) */
export function cssColor(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}
