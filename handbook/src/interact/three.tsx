// 3D v2: yönlendirilmiş kamera (CameraControls), instanced küpler, raycast hover, yumuşak geçişler, eksen okları.
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
  // Ekranda olmayan sahne çizilmesin: aynı sayfadaki birkaç Canvas birbirini yavaşlatmasın
  const box = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    if (!box.current) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { rootMargin: "200px" });
    io.observe(box.current);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={box} style={{ height, position: "relative", borderRadius: 6, overflow: "hidden", background: "var(--stage-bg)" }}>
      <Canvas frameloop={visible ? "always" : "never"} camera={{ position: view.pos, fov: 40 }} dpr={[1, 2]} gl={{ antialias: true, preserveDrawingBuffer: true }}>
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

/** Süreli geçiş: hedef değişince baştan sona easeInOutCubic. stagger: i. küp i × stagger saniye geç başlar */
export interface Tween { duration?: number; stagger?: number }
export const easeInOut = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

/** Tek InstancedMesh: id sırası sabit kalmalı. Hedefe yumuşak geçiş, hover'da tooltip ve vurgu */
export function Cubes({ items, cell = 0.42, tween }: { items: Cube[]; cell?: number; tween?: Tween }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const n = items.length;
  const cur = useMemo(() => items.map((it) => {
    const p = new THREE.Vector3(...it.pos), s = new THREE.Vector3(...(it.size ?? [cell, cell, cell]));
    return { p, s, c: new THREE.Color(it.color), p0: p.clone(), s0: s.clone(), p1: p.clone(), s1: s.clone(), t0: -1 };
  }), [n]);
  const [hover, setHover] = useState<number | null>(null);
  const m = useMemo(() => new THREE.Matrix4(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);
  const tmpC = useMemo(() => new THREE.Color(), []);
  const tgtP = useMemo(() => new THREE.Vector3(), []);
  const tgtS = useMemo(() => new THREE.Vector3(), []);

  const dimC = useMemo(() => new THREE.Color(cssColor("--stage-bg")), []);
  useFrame((state, dt) => {
    const mesh = ref.current;
    if (!mesh) return;
    const now = state.clock.elapsedTime;
    items.forEach((it, i) => {
      const c = cur[i];
      tgtP.set(...it.pos);
      const sz = it.hidden ? [0.0001, 0.0001, 0.0001] : (it.size ?? [cell, cell, cell]);
      tgtS.set(sz[0], sz[1], sz[2]);
      if (tween) {
        // hedef değiştiyse bulunduğu yerden yeni bir geçiş başlat
        if (!c.p1.equals(tgtP) || !c.s1.equals(tgtS)) {
          c.p0.copy(c.p); c.s0.copy(c.s); c.p1.copy(tgtP); c.s1.copy(tgtS);
          c.t0 = now + i * (tween.stagger ?? 0);
        }
        const k = c.t0 < 0 ? 1 : easeInOut(Math.min(1, Math.max(0, (now - c.t0) / (tween.duration ?? 0.9))));
        c.p.lerpVectors(c.p0, c.p1, k);
        c.s.lerpVectors(c.s0, c.s1, k);
      } else {
        easing.damp3(c.p, tgtP, 0.28, dt);
        easing.damp3(c.s, tgtS, 0.28, dt);
      }
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
    mesh.computeBoundingSphere();   // hareket eden küpler hover'da bulunabilsin
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

export interface Axis {
  dir: [number, number, number];   // birim yön
  len: number;                     // dünya birimi
  color: string;                   // CSS değişkeni de olabilir: "var(--ax-T)"
  label: ReactNode;
  ticks?: number;                  // eksen boyunca hücre sayısı (her hücrenin ortasına bir çentik)
  labelOffset?: [number, number, number];
}

/** Eksen okları: bir köşeden çıkan renkli çizgi + ok başı + uçta etiket. Uzunluk ve köşe değişince yumuşakça kayar. */
export function Axes3D({ origin, axes }: { origin: [number, number, number]; axes: Axis[] }) {
  return <>{axes.map((a, i) => <AxisArrow key={i} origin={origin} {...a} />)}</>;
}

function AxisArrow({ origin, dir, len, color, label, ticks = 0, labelOffset = [0, 0, 0] }: Axis & { origin: [number, number, number] }) {
  const shaft = useRef<THREE.Mesh>(null), head = useRef<THREE.Mesh>(null), tag = useRef<THREE.Group>(null), tk = useRef<THREE.Group>(null);
  const st = useMemo(() => ({ o: new THREE.Vector3(...origin), len }), []);
  const d = useMemo(() => new THREE.Vector3(...dir).normalize(), [dir[0], dir[1], dir[2]]);
  const quat = useMemo(() => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d), [d]);
  const tgt = useMemo(() => new THREE.Vector3(), []);
  const col = useMemo(() => new THREE.Color(color.startsWith("var(") ? cssColor(color.slice(4, -1)) : color), [color]);
  const nT = Math.max(ticks, 1);
  useFrame((_, dt) => {
    tgt.set(...origin);
    easing.damp3(st.o, tgt, 0.3, dt);
    st.len += (len - st.len) * (1 - Math.exp(-dt / 0.12));
    const L = Math.max(0.001, st.len);
    if (shaft.current) { shaft.current.position.copy(st.o).addScaledVector(d, L / 2); shaft.current.scale.set(1, L, 1); shaft.current.quaternion.copy(quat); }
    if (head.current) { head.current.position.copy(st.o).addScaledVector(d, L + 0.06); head.current.quaternion.copy(quat); }
    if (tag.current) tag.current.position.copy(st.o).addScaledVector(d, L + 0.32).add(tgt.set(...labelOffset));
    if (tk.current) tk.current.children.forEach((m, k) => {
      m.position.copy(st.o).addScaledVector(d, ((k + 0.5) / nT) * L);
      m.visible = k < ticks;
    });
  });
  return (
    <group>
      <mesh ref={shaft}><cylinderGeometry args={[0.014, 0.014, 1, 8]} /><meshBasicMaterial color={col} /></mesh>
      <mesh ref={head}><coneGeometry args={[0.055, 0.14, 16]} /><meshBasicMaterial color={col} /></mesh>
      <group ref={tk}>{Array.from({ length: Math.max(ticks, 1) }, (_, k) => <mesh key={k}><sphereGeometry args={[0.03, 10, 10]} /><meshBasicMaterial color={col} /></mesh>)}</group>
      <group ref={tag}>
        <Html center style={{ pointerEvents: "none", whiteSpace: "nowrap" }} zIndexRange={[10, 0]}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 12.5, fontWeight: 650, color, textShadow: "0 0 3px var(--stage-bg), 0 0 6px var(--stage-bg)" }}>{label}</div>
        </Html>
      </group>
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
