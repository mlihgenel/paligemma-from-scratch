// 5.3 RMSNorm: adım adım laboratuvar, 3D geometri, keepdim, gerçek ağırlıklar, bf16 işlem sırası
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

const fmt = (x: number) => fmtNum(x, 4);

export function RealNormWeights() {
  const { data, error } = useData<PgSlices>("pg_slices");
  const [which, setWhich] = useState<"input_layernorm" | "post_attention_layernorm">("input_layernorm");
  const [layer, setLayer] = useState(0);
  const mats = useMemo(() => {
    if (!data) return null;
    const m = data.norms[which];
    return { vals: Float32Array.from(m.data, (v) => 1 + v), rows: m.shape[0], cols: m.shape[1] };
  }, [data, which]);
  if (!data || !mats) return <Loading what="gerçek norm ağırlıkları" error={error} />;
  const row = Array.from(rowOf(data.norms[which], layer), (v) => 1 + v);
  const fin = Array.from(data.norms.final.data, (v) => 1 + v);
  const neg = row.filter((v) => v < 0).length;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <Seg value={which} onChange={setWhich} options={[["input_layernorm", "input_layernorm"], ["post_attention_layernorm", "post_attention_layernorm"]]} />
        <span className="small muted">renk = 1 + w, satıra tıkla</span>
      </div>
      <Heatmap values={mats.vals} rows={mats.rows} cols={mats.cols} height={200} diverging domain={[-4, 4]} rowLabel="katman (0–17)" colLabel="boyut (0–2047)"
        highlight={{ r: layer }} onCell={(r) => setLayer(r)} cellLabel={(r, c, v) => `katman ${r}, boyut ${c}: 1+w = ${v.toFixed(3)}`} />
      <div className="grid2" style={{ marginTop: 14 }}>
        <div>
          <div className="small"><b>Katman {layer}</b> · {which}: 1 + w dağılımı</div>
          <Histogram values={row} bins={60} color="var(--ax-hd)" marks={[{ x: 1, label: "w=0 → 1" }, { x: 0, label: "0", color: "var(--bad)" }]} />
        </div>
        <div>
          <div className="small"><b>Son norm</b> (model.norm): 1 + w dağılımı</div>
          <Histogram values={fin} bins={60} color="var(--ax-T)" marks={[{ x: 1, label: "w=0 → 1" }]} />
        </div>
      </div>
      <div className="row" style={{ gap: 10, marginTop: 10 }}>
        <Stat label="ortalama ölçek (1+w)" value={(row.reduce((a, b) => a + b, 0) / row.length).toFixed(2)} />
        <Stat label="en büyük ölçek" value={Math.max(...row).toFixed(2)} />
        <Stat label="1+w < 0 (işaret çeviren)" value={neg} sub={`/ ${row.length} boyut`} tone={neg ? "warn" : undefined} />
      </div>
    </div>
  );
}

export function OrderStats() {
  const { data } = useData<GemmaToy>("gemma_toy");
  if (!data) return null;
  const r = data.rms_order;
  return <span className="pill bad">{r.n_diff} / {r.n} eleman ({((r.n_diff / r.n) * 100).toFixed(0)}%) farklı, max fark {r.maxdiff}</span>;
}

export function ZeroInitStats({ broken }: { broken: boolean }) {
  const { data } = useData<GemmaToy>("gemma_toy");
  if (!data) return null;
  const z = data.rms_zero_init_trap;
  return (
    <div className="row" style={{ gap: 10 }}>
      <Stat label="w = 0 iken: doğru ile hatalı arası fark" value={broken ? z.zero_w.ignores_weight.toFixed(2) : "–"} tone={broken ? "warn" : undefined} sub={broken ? "test GEÇERDİ, hata görünmez" : ""} />
      <Stat label="w rastgele iken fark" value={broken ? z.random_w.ignores_weight.toFixed(2) : "–"} tone={broken ? "bad" : undefined} sub={broken ? "test hatayı yakalar" : ""} />
    </div>
  );
}
