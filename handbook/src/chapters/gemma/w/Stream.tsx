// 5.1 Residual akışı: 3D otoyol ve akış grafikleri
import { useMemo, useState } from "react";
import { QuadraticBezierLine } from "@react-three/drei";
import { useData } from "../../../lib/data";
import { rng } from "../../../lib/tensor";
import { Label, Stage, TBox, cssVar } from "../../../components/three/core";
import { LineChart } from "../../../components/charts";
import { Loading, Seg, Stat } from "../../../components/ui";
import type { GemmaToy } from "./types";

const DX = 1.5;   // aşamalar arası mesafe

export function StreamCompare({ show = ["normal", "residual_after_norm", "post_norm"] as (keyof GemmaToy["stream"])[], metric = "cos_to_embed" as "cos_to_embed" | "norms" }) {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [m, setM] = useState(metric);
  if (!data) return <Loading what="oyuncak model" error={error} />;
  const NAMES: Record<string, [string, string, boolean?]> = {
    normal: ["senin kodun (pre-norm)", "var(--ax-T)"],
    residual_after_norm: ["residual norm'dan sonra", "var(--bad)", true],
    post_norm: ["post-norm (orijinal Transformer)", "var(--ax-I)", true],
    no_sqrt_d: ["×√d olmadan", "var(--warn)", true],
  };
  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <Seg value={m} onChange={setM} options={[["cos_to_embed", "embedding'e benzerlik"], ["norms", "akış normu ‖h‖"]]} />
      </div>
      <LineChart height={240} xLabel="katman (0 = giriş)" yLabel={m === "norms" ? "‖h‖" : "cos(h, h₀)"}
        yDomain={m === "cos_to_embed" ? [0, 1] : undefined}
        series={show.map((k) => ({ name: NAMES[k][0], color: NAMES[k][1], dash: NAMES[k][2], values: data.stream[k][m] }))} />
    </div>
  );
}

export function UpdateSizes() {
  const { data, error } = useData<GemmaToy>("gemma_toy");
  if (!data) return <Loading what="oyuncak model" error={error} />;
  const s = data.stream.normal;
  return (
    <LineChart height={220} xLabel="katman" yLabel="‖güncelleme‖" fmtX={(x) => String(x + 1)}
      series={[
        { name: "attn dalının yazdığı ‖attn(norm(h))‖", color: "var(--fg-faint)", values: s.attn_update },
        { name: "mlp dalının yazdığı ‖mlp(norm(h))‖", color: "var(--ax-D)", values: s.mlp_update },
        { name: "akışın kendisi ‖h‖", color: "var(--ax-T)", values: s.norms.slice(1) },
      ]} />
  );
}
