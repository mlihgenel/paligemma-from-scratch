// Mafs tabanlı 2D düzlemler (R3)
import { Mafs, Coordinates, Plot, Point, Line, Text, useMovablePoint } from "mafs";
import { geluErf, geluTanh, relu, silu } from "../../../lib/tensor";

export function GeluPlane() {
  const p = useMovablePoint([-0.8, 0], { constrain: "horizontal", color: "var(--accent)" });
  const x = p.x;
  const vals = [
    { name: "GELU (tanh) · Gemma", f: geluTanh, color: "var(--accent)" },
    { name: "SiLU · Llama", f: silu, color: "var(--ax-T)" },
    { name: "ReLU", f: relu, color: "var(--fg-faint)" },
  ];
  const diff = geluTanh(x) - geluErf(x);
  return (
    <div>
      <div style={{ maxWidth: 820, margin: "0 auto" }}>
        <Mafs height={400} viewBox={{ x: [-4.5, 3.5], y: [-0.8, 3.2] }} preserveAspectRatio="contain">
          <Coordinates.Cartesian subdivisions={false} />
          {vals.map((v) => <Plot.OfX key={v.name} y={v.f} color={v.color} weight={v.name.startsWith("GELU") ? 3 : 2} style={v.name.startsWith("ReLU") ? "dashed" : "solid"} />)}
          <Line.Segment point1={[x, -0.8]} point2={[x, 3.2]} color="var(--fg-faint)" style="dashed" weight={1} />
          {vals.map((v) => <Point key={v.name} x={x} y={v.f(x)} color={v.color} />)}
          <Text x={x} y={geluTanh(x)} attach="e" size={13} color="var(--accent)">{geluTanh(x).toFixed(3)}</Text>
          {p.element}
        </Mafs>
      </div>
      <div className="stats">
        {vals.map((v) => (
          <div className="stat" key={v.name}>
            <div className="lbl"><span style={{ color: v.color }}>■</span> {v.name}</div>
            <div className="val">{v.f(x).toFixed(4)}</div>
          </div>
        ))}
        <div className="stat">
          <div className="lbl">tanh − erf farkı</div>
          <div className="val" style={{ color: Math.abs(diff) > 1e-5 ? "var(--bad)" : undefined }}>{diff.toExponential(1)}</div>
          <div className="sub">atol = 1e-5'ten {Math.abs(diff) > 1e-5 ? "büyük → allclose düşer" : "küçük"}</div>
        </div>
      </div>
    </div>
  );
}
