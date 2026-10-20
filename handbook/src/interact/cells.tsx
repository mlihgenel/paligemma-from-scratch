// Hücre tuvali: her hücre kimliğini (id) koruyan, yaylı (spring) animasyonla yer değiştiren bir div.
// Aynı id'li hücre bir durumdan ötekine fiziksel olarak kayar: "hangi sayı nereye gitti" izlenebilir.
import { ReactNode, useRef, useState } from "react";
import { motion } from "framer-motion";
import * as d3 from "d3";
import { Inspect } from "./Inspect";

export interface Cell {
  id: string;
  x: number; y: number; w: number; h: number;
  v?: number;                 // renk için değer
  fill?: string;              // doğrudan renk (v yerine)
  text?: string;              // hücre içi yazı
  opacity?: number;
  ring?: boolean;             // vurgu halkası
  editable?: boolean;
  tip?: () => ReactNode;      // hover içeriği
  group?: string;             // hover'da aynı gruptaki hücreler de vurgulanır
  z?: number;
  delay?: number;             // saniye: sıralı akış için
}

const NEG = "#5b67d6", MID = "#f1efe9", POS = "#cf5a35";
const NEG_D = "#7c86e6", MID_D = "#2b2b30", POS_D = "#e07650";
export function valueColor(v: number, max: number, dark = false) {
  const t = Math.max(-1, Math.min(1, v / (max || 1)));
  const s = dark ? d3.interpolateRgbBasis([NEG_D, MID_D, POS_D]) : d3.interpolateRgbBasis([NEG, MID, POS]);
  return s((t + 1) / 2);
}
function inkFor(bg: string) {
  const c = d3.rgb(bg);
  return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b > 150 ? "#1b1b1f" : "#ffffff";
}
export const isDark = () => document.documentElement.dataset.theme === "dark";

export function CellCanvas({ cells, width, height, max = 1, onDrag, labels, highlightGroup, onHover }: {
  cells: Cell[]; width: number; height: number; max?: number;
  onDrag?: (id: string, dv: number) => void;
  labels?: ReactNode;
  highlightGroup?: string | null;
  onHover?: (id: string | null) => void;
}) {
  const [hover, setHover] = useState<{ c: Cell; x: number; y: number } | null>(null);
  const drag = useRef<{ id: string; y: number } | null>(null);
  const dark = isDark();
  const hg = hover?.c.group ?? highlightGroup ?? null;
  return (
    <div style={{ position: "relative", width, height, touchAction: "none" }} onPointerLeave={() => { setHover(null); onHover?.(null); }}
      onPointerMove={(e) => {
        if (drag.current && onDrag) {
          const dy = drag.current.y - e.clientY;
          drag.current.y = e.clientY;
          onDrag(drag.current.id, dy);
        }
      }}
      onPointerUp={() => (drag.current = null)}>
      {cells.map((c) => {
        const bg = c.fill ?? (c.v !== undefined ? valueColor(c.v, max, dark) : "transparent");
        const dim = hg && c.group && c.group !== hg;
        return (
          <motion.div key={c.id}
            initial={false}
            animate={{ left: c.x, top: c.y, width: c.w, height: c.h, opacity: (c.opacity ?? 1) * (dim ? 0.35 : 1), backgroundColor: bg }}
            transition={{ type: "spring", stiffness: 170, damping: 24, mass: 0.9, delay: c.delay ?? 0 }}
            onPointerEnter={(e) => {
              const r = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
              setHover({ c, x: e.clientX - r.left, y: e.clientY - r.top });
              onHover?.(c.id);
            }}
            onPointerMove={(e) => {
              const r = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
              setHover({ c, x: e.clientX - r.left, y: e.clientY - r.top });
            }}
            onPointerDown={(e) => { if (c.editable && onDrag) { drag.current = { id: c.id, y: e.clientY }; (e.target as HTMLElement).setPointerCapture?.(e.pointerId); } }}
            style={{
              position: "absolute", borderRadius: 3, zIndex: c.z ?? 1,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "var(--mono)", fontSize: Math.min(12, c.h * 0.42), fontVariantNumeric: "tabular-nums",
              color: bg === "transparent" || bg.startsWith("color-mix") || bg.includes("--bg") ? "var(--fg)" : bg.startsWith("var") ? "#ffffff" : inkFor(bg),
              outline: hover?.c.id === c.id ? "2px solid var(--fg)" : c.ring ? "2px solid var(--accent)" : "none",
              outlineOffset: 1, cursor: c.editable ? "ns-resize" : c.tip ? "crosshair" : "default", userSelect: "none",
            }}>
            {c.text}
          </motion.div>
        );
      })}
      {labels}
      {hover?.c.tip && <Inspect x={hover.x} y={hover.y} containerWidth={width}>{hover.c.tip()}</Inspect>}
    </div>
  );
}
