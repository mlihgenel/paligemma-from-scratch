// 4.2 The residual stream: the 3D highway and stream charts
import { useMemo, useState } from "react";
import { QuadraticBezierLine } from "@react-three/drei";
import { useData } from "../../../lib/data";
import { rng } from "../../../lib/tensor";
import { Label, Stage, TBox, cssVar } from "../../../components/three/core";
import { LineChart } from "../../../components/charts";
import { Loading, Seg, Stat } from "../../../components/ui";
import type { GemmaToy } from "./types";
import { useT } from "../../../i18n";

const DX = 1.5;   // distance between stages

export function StreamCompare({ show = ["normal", "residual_after_norm", "post_norm"] as (keyof GemmaToy["stream"])[], metric = "cos_to_embed" as "cos_to_embed" | "norms" }) {
  const tr = useT();
  const { data, error } = useData<GemmaToy>("gemma_toy");
  const [m, setM] = useState(metric);
  if (!data) return <Loading what={tr("toy model", "oyuncak model")} error={error} />;
  const NAMES: Record<string, [string, string, boolean?]> = {
    normal: [tr("the real code (pre-norm)", "gerçek kod (pre-norm)"), "var(--ax-T)"],
    residual_after_norm: [tr("residual after the norm", "residual norm'dan sonra"), "var(--bad)", true],
    post_norm: [tr("post-norm (original Transformer)", "post-norm (orijinal Transformer)"), "var(--ax-I)", true],
    no_sqrt_d: [tr("without ×√d", "×√d olmadan"), "var(--warn)", true],
  };
  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <Seg value={m} onChange={setM} options={[["cos_to_embed", tr("similarity to the embedding", "embedding'e benzerlik")], ["norms", tr("stream norm ‖h‖", "akış normu ‖h‖")]]} />
      </div>
      <LineChart height={240} xLabel={tr("layer (0 = input)", "katman (0 = giriş)")} yLabel={m === "norms" ? "‖h‖" : "cos(h, h₀)"}
        yDomain={m === "cos_to_embed" ? [0, 1] : undefined}
        series={show.map((k) => ({ name: NAMES[k][0], color: NAMES[k][1], dash: NAMES[k][2], values: data.stream[k][m] }))} />
    </div>
  );
}

export function UpdateSizes() {
  const tr = useT();
  const { data, error } = useData<GemmaToy>("gemma_toy");
  if (!data) return <Loading what={tr("toy model", "oyuncak model")} error={error} />;
  const s = data.stream.normal;
  return (
    <LineChart height={220} xLabel={tr("layer", "katman")} yLabel={tr("‖update‖", "‖güncelleme‖")} fmtX={(x) => String(x + 1)}
      series={[
        { name: tr("written by the attn branch ‖attn(norm(h))‖", "attn dalının yazdığı ‖attn(norm(h))‖"), color: "var(--fg-faint)", values: s.attn_update },
        { name: tr("written by the mlp branch ‖mlp(norm(h))‖", "mlp dalının yazdığı ‖mlp(norm(h))‖"), color: "var(--ax-D)", values: s.mlp_update },
        { name: tr("the stream itself ‖h‖", "akışın kendisi ‖h‖"), color: "var(--ax-T)", values: s.norms.slice(1) },
      ]} />
  );
}
