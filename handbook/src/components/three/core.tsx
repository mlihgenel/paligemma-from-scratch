// Shared 3D pieces: stage, tensor box, label. Labels are DOM (drei Html) and use the theme colors.
import { ReactNode, Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Edges, Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useT } from "../../i18n";

export function Stage({ children, height = 380, camera = [6, 4, 8] as [number, number, number], fov = 40, autoRotate = false, target = [0, 0, 0] as [number, number, number], controls = true }: {
  children: ReactNode; height?: number; camera?: [number, number, number]; fov?: number; autoRotate?: boolean; target?: [number, number, number]; controls?: boolean;
}) {
  const t = useT();
  return (
    <div style={{ height, position: "relative", borderRadius: 8, overflow: "hidden", background: "var(--stage-bg)" }}>
      <Canvas camera={{ position: camera, fov }} dpr={[1, 2]} gl={{ antialias: true, preserveDrawingBuffer: true }}>
        <ambientLight intensity={0.75} />
        <directionalLight position={[5, 8, 6]} intensity={1.1} />
        <directionalLight position={[-6, -2, -4]} intensity={0.3} />
        <Suspense fallback={null}>{children}</Suspense>
        {controls && <OrbitControls makeDefault enableDamping autoRotate={autoRotate} autoRotateSpeed={0.6} target={target} enablePan={false} />}
      </Canvas>
      {controls && <div style={{ position: "absolute", bottom: 8, right: 12, pointerEvents: "none", fontFamily: "var(--sans)", fontSize: 11.5, color: "var(--fg-faint)" }}>{t("drag: rotate · scroll: zoom", "sürükle: döndür · kaydır: yakınlaş")}</div>}
    </div>
  );
}

/** Sahnedeki DOM etiketi */
export function Label({ position, children, color, size = 12, bold = false, center = true }: {
  position: [number, number, number]; children: ReactNode; color?: string; size?: number; bold?: boolean; center?: boolean;
}) {
  return (
    <Html position={position} center={center} style={{ pointerEvents: "none", whiteSpace: "nowrap" }} zIndexRange={[10, 0]}>
      <div style={{ fontFamily: "var(--sans)", fontSize: size, fontWeight: bold ? 700 : 500, color: color ?? "var(--fg)", textShadow: "0 0 4px var(--bg), 0 0 2px var(--bg)" }}>{children}</div>
    </Html>
  );
}

/** Animated sized box: smooth transition when size changes */
export function TBox({ size, position = [0, 0, 0], color = "#d0731d", opacity = 0.85, edges = true, onClick, highlight = false, wire = false }: {
  size: [number, number, number]; position?: [number, number, number]; color?: string; opacity?: number; edges?: boolean;
  onClick?: () => void; highlight?: boolean; wire?: boolean;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const target = useRef({ s: new THREE.Vector3(...size), p: new THREE.Vector3(...position) });
  target.current.s.set(...size);
  target.current.p.set(...position);
  useFrame((_, dt) => {
    const m = ref.current;
    if (!m) return;
    const k = 1 - Math.exp(-dt * 8);
    m.scale.lerp(target.current.s, k);
    m.position.lerp(target.current.p, k);
  });
  return (
    <mesh ref={ref} scale={size} position={position} onClick={onClick ? (e) => { e.stopPropagation(); onClick(); } : undefined}
      onPointerOver={onClick ? () => (document.body.style.cursor = "pointer") : undefined}
      onPointerOut={onClick ? () => (document.body.style.cursor = "") : undefined}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={color} transparent={wire || opacity < 1} depthWrite={!wire && opacity >= 1} opacity={wire ? 0.08 : opacity} roughness={0.55} metalness={0.05} emissive={highlight ? color : "#000"} emissiveIntensity={highlight ? 0.35 : 0} />
      {edges && <Edges color={highlight ? "#ffffff" : color} threshold={15} />}
    </mesh>
  );
}

/** Map a dim to a visual length: large dims on a log scale (so 2048 and 16384 fit on screen) */
export const vis = (n: number, k = 0.55) => Math.max(0.12, Math.log2(n + 1) * k);

/** Read a theme color in JS (three materials do not accept CSS variables) */
export function cssVar(name: string): string {
  if (typeof window === "undefined") return "#888";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}
