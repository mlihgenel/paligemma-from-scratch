// 4.4 RMSNorm: step-by-step lab, 3D geometry, keepdim, real weights, bf16 order of operations
import { useMemo, useState } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import { useData, rowOf } from "../../../lib/data";
import { bf16Ulp, toBf16 } from "../../../lib/bf16";
import { fmtNum, layernorm, rmsnormSteps, rng } from "../../../lib/tensor";
import { Heatmap, Histogram } from "../../../components/charts";
import { Label, Stage, TBox, cssVar } from "../../../components/three/core";
import { Loading, Seg, Stat, S } from "../../../components/ui";
import type { GemmaToy, PgSlices } from "./types";
import { useT } from "../../../i18n";

const fmt = (x: number) => fmtNum(x, 4);

export function RealNormWeights() {
  const tr = useT();
  const { data, error } = useData<PgSlices>("pg_slices");
  const [which, setWhich] = useState<"input_layernorm" | "post_attention_layernorm">("input_layernorm");
  const [layer, setLayer] = useState(0);
  const mats = useMemo(() => {
    if (!data) return null;
    const m = data.norms[which];
    return { vals: Float32Array.from(m.data, (v) => 1 + v), rows: m.shape[0], cols: m.shape[1] };
  }, [data, which]);
  if (!data || !mats) return <Loading what={tr("real norm weights", "gerçek norm ağırlıkları")} error={error} />;
  const row = Array.from(rowOf(data.norms[which], layer), (v) => 1 + v);
  const fin = Array.from(data.norms.final.data, (v) => 1 + v);
  const neg = row.filter((v) => v < 0).length;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <Seg value={which} onChange={setWhich} options={[["input_layernorm", "input_layernorm"], ["post_attention_layernorm", "post_attention_layernorm"]]} />
        <span className="small muted">{tr("color = 1 + w, click a row", "renk = 1 + w, satıra tıkla")}</span>
      </div>
      <Heatmap values={mats.vals} rows={mats.rows} cols={mats.cols} height={200} diverging domain={[-4, 4]} rowLabel={tr("layer (0–17)", "katman (0–17)")} colLabel={tr("dim (0–2047)", "boyut (0–2047)")}
        highlight={{ r: layer }} onCell={(r) => setLayer(r)} cellLabel={(r, c, v) => tr(`layer ${r}, dim ${c}: 1+w = ${v.toFixed(3)}`, `katman ${r}, boyut ${c}: 1+w = ${v.toFixed(3)}`)} />
      <div className="grid2" style={{ marginTop: 14 }}>
        <div>
          <div className="small"><b>{tr("Layer", "Katman")} {layer}</b> · {which}: {tr("distribution of 1 + w", "1 + w dağılımı")}</div>
          <Histogram values={row} bins={60} color="var(--ax-hd)" marks={[{ x: 1, label: "w=0 → 1" }, { x: 0, label: "0", color: "var(--bad)" }]} />
        </div>
        <div>
          <div className="small"><b>{tr("Final norm", "Son norm")}</b> (model.norm): {tr("distribution of 1 + w", "1 + w dağılımı")}</div>
          <Histogram values={fin} bins={60} color="var(--ax-T)" marks={[{ x: 1, label: "w=0 → 1" }]} />
        </div>
      </div>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label={tr("mean scale (1+w)", "ortalama ölçek (1+w)")} value={(row.reduce((a, b) => a + b, 0) / row.length).toFixed(2)} />
        <Stat label={tr("largest scale", "en büyük ölçek")} value={Math.max(...row).toFixed(2)} />
        <Stat label={tr("1+w < 0 (flips the sign)", "1+w < 0 (işaret çeviren)")} value={neg} sub={`/ ${row.length} ${tr("dims", "boyut")}`} tone={neg ? "warn" : undefined} />
      </div>
    </div>
  );
}

export function OrderStats() {
  const tr = useT();
  const { data } = useData<GemmaToy>("gemma_toy");
  if (!data) return null;
  const r = data.rms_order;
  return <span className="pill bad">{r.n_diff} / {r.n} {tr("elements", "eleman")} ({((r.n_diff / r.n) * 100).toFixed(0)}%) {tr("differ, max diff", "farklı, max fark")} {r.maxdiff}</span>;
}

export function ZeroInitStats({ broken }: { broken: boolean }) {
  const tr = useT();
  const { data } = useData<GemmaToy>("gemma_toy");
  if (!data) return null;
  const z = data.rms_zero_init_trap;
  return (
    <div className="row" style={{ gap: 10 }}>
      <Stat label={tr("with w = 0: difference between correct and buggy", "w = 0 iken: doğru ile hatalı arası fark")} value={broken ? z.zero_w.ignores_weight.toFixed(2) : "–"} tone={broken ? "warn" : undefined} sub={broken ? tr("the test would PASS, the bug is invisible", "test GEÇERDİ, hata görünmez") : ""} />
      <Stat label={tr("difference with random w", "w rastgele iken fark")} value={broken ? z.random_w.ignores_weight.toFixed(2) : "–"} tone={broken ? "bad" : undefined} sub={broken ? tr("the test catches the bug", "test hatayı yakalar") : ""} />
    </div>
  );
}
