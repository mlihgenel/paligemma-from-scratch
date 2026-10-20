import { describe, expect, it } from "vitest";
import golden from "./golden.json";
import { toBf16 } from "../lib/bf16";
import { geluErf, geluTanh, layernorm, rmsnorm, silu, softmax } from "../lib/tensor";

const rows = (flat: number[], n: number) => Array.from({ length: flat.length / n }, (_, i) => flat.slice(i * n, i * n + n));
const close = (a: number[], b: number[], atol = 1e-5) =>
  a.forEach((x, i) => expect(Math.abs(x - b[i]), `i=${i} ${x} vs ${b[i]}`).toBeLessThanOrEqual(atol + 1e-5 * Math.abs(b[i])));

describe("PyTorch altın değerleri", () => {
  const X = rows(golden.x, 8);
  it("RMSNorm (senin GemmaRMSNorm, float32)", () => close(X.flatMap((r) => rmsnorm(r, golden.w)), golden.rmsnorm));
  it("LayerNorm", () => close(X.flatMap((r) => layernorm(r)), golden.layernorm));
  it("softmax", () => close(X.flatMap(softmax), golden.softmax));
  it("gelu tanh / erf / silu", () => {
    close(golden.grid.map(geluTanh), golden.gelu_tanh);
    close(golden.grid.map(geluErf), golden.gelu_erf);
    close(golden.grid.map(silu), golden.silu);
  });
  it("bf16 yuvarlama bit bit aynı", () => {
    golden.bf16_in.forEach((x, i) => expect(Object.is(toBf16(x), golden.bf16_out[i]) || toBf16(x) === golden.bf16_out[i]).toBe(true));
  });
  it("RMSNorm bf16 (Gemma sırası) bit bit aynı", () => {
    const out = rows(golden.rms_bf16_x, 16).flatMap((r) => rmsnorm(r, golden.rms_bf16_w, 1e-6, "bf16", "gemma"));
    const diff = out.filter((x, i) => x !== golden.rms_bf16_out[i]).length;
    expect(diff).toBe(0);
  });
});
