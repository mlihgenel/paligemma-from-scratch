// A single tooltip style: opens next to a given point inside a container and does not overflow the edges.
import { ReactNode, useLayoutEffect, useRef, useState } from "react";

export function Inspect({ x, y, children, containerWidth }: { x: number; y: number; children: ReactNode; containerWidth?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(220);
  useLayoutEffect(() => { if (ref.current) setW(ref.current.offsetWidth); });
  const cw = containerWidth ?? 10000;
  const left = x + 14 + w > cw ? Math.max(0, x - w - 14) : x + 14;
  return (
    <div ref={ref} className="tip" style={{ position: "absolute", left, top: Math.max(0, y - 10), pointerEvents: "none", zIndex: 5, maxWidth: 320, whiteSpace: "nowrap" }}>
      {children}
    </div>
  );
}
