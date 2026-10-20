// Mimari haritası: içindekiler panelinde dikey, sayfa içinde yatay. Aktif bloklar vurgulanır.
import { Link } from "react-router-dom";

export type StageKey = "image" | "process" | "patch" | "encoder" | "proj" | "merge" | "scale" | "decoder" | "norm" | "head";

const STAGES: { key: StageKey; label: string; shape: string; to: string; group: "v" | "b" | "l" }[] = [
  { key: "image", label: "Görüntü", shape: "H×W×3", to: "/goruntu", group: "v" },
  { key: "process", label: "Ön işleme", shape: "3×224×224", to: "/goruntu", group: "v" },
  { key: "patch", label: "Yama + konum", shape: "256×1152", to: "/goruntu", group: "v" },
  { key: "encoder", label: "SigLIP ×27", shape: "256×1152", to: "/encoder", group: "v" },
  { key: "proj", label: "Projektör", shape: "256×2048", to: "/kopru", group: "b" },
  { key: "merge", label: "Birleştir ÷√d", shape: "T×2048", to: "/hepsi", group: "b" },
  { key: "scale", label: "×√d", shape: "T×2048", to: "/gemma/sqrt-d", group: "l" },
  { key: "decoder", label: "Decoder ×18", shape: "T×2048", to: "/gemma/akis", group: "l" },
  { key: "norm", label: "Son norm", shape: "T×2048", to: "/gemma/model", group: "l" },
  { key: "head", label: "lm_head", shape: "T×257216", to: "/gemma/tying", group: "l" },
];
const GROUP = { v: "SigLIP", b: "köprü", l: "Gemma" };

export function PipelineMinimap({ active = [], vertical = false, onNavigate }: { active?: StageKey[]; vertical?: boolean; onNavigate?: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: vertical ? "column" : "row", gap: vertical ? 0 : 2, fontFamily: "var(--sans)", overflowX: vertical ? undefined : "auto" }}>
      {STAGES.map((s, i) => {
        const on = active.includes(s.key);
        const newGroup = i === 0 || STAGES[i - 1].group !== s.group;
        return (
          <Link key={s.key} to={s.to} onClick={onNavigate} style={{
            background: "none", padding: vertical ? "5px 0 5px 14px" : "6px 8px", flex: vertical ? undefined : 1,
            display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline",
            borderLeft: vertical ? `2px solid ${on ? "var(--accent)" : "var(--border)"}` : undefined,
            borderTop: !vertical ? `2px solid ${on ? "var(--accent)" : "var(--border)"}` : undefined,
            marginTop: vertical && newGroup && i > 0 ? 10 : 0, color: on ? "var(--fg)" : "var(--fg-muted)",
          }}>
            <span style={{ fontSize: 13.5, fontWeight: on ? 600 : 400 }}>
              {vertical && newGroup && <span style={{ fontSize: 10.5, color: "var(--fg-faint)", textTransform: "uppercase", letterSpacing: "0.07em", marginRight: 8 }}>{GROUP[s.group]}</span>}
              {s.label}
            </span>
            <span className="mono" style={{ fontSize: 11, color: "var(--fg-faint)" }}>{s.shape}</span>
          </Link>
        );
      })}
    </div>
  );
}
