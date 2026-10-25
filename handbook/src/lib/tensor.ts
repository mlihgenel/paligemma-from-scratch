// Small tensor helpers for the book's interactive math. Vectors are plain number[].
// All of them are checked against PyTorch golden values under src/tests.
import { fround, toBf16 } from "./bf16";

export type Vec = number[];
export type DType = "f32" | "bf16";

export const sum = (v: Vec) => v.reduce((a, b) => a + b, 0);
export const mean = (v: Vec) => sum(v) / v.length;
export const l2 = (v: Vec) => Math.sqrt(sum(v.map((x) => x * x)));
export const rms = (v: Vec) => Math.sqrt(mean(v.map((x) => x * x)));
export const dot = (a: Vec, b: Vec) => a.reduce((s, x, i) => s + x * b[i], 0);
export const cosine = (a: Vec, b: Vec) => dot(a, b) / (l2(a) * l2(b) + 1e-12);
export const std = (v: Vec) => {
  const m = mean(v);
  return Math.sqrt(mean(v.map((x) => (x - m) ** 2)));
};

/** mean(x²) in float32 arithmetic: every step is rounded to float32 with Math.fround */
function meanSqF32(v: Vec): number {
  let s = 0;
  for (const x of v) s = fround(s + fround(x * x));
  return fround(s / v.length);
}

export interface RmsSteps {
  x: Vec; sq: Vec; meanSq: number; plusEps: number; rsqrt: number; normed: Vec; scale: Vec; out: Vec;
}

/**
 * Gemma RMSNorm, senin GemmaRMSNorm.forward'ınla aynı sıra:
 *   out = (x.float() * rsqrt(mean(x²) + eps)) * (1 + w.float())  -> sonra girdinin dtype'ına
 * order="llama": önce dtype'a dön, sonra (1+w) ile çarp (Llama sırası).
 */
export function rmsnormSteps(x: Vec, w: Vec, eps = 1e-6, dtype: DType = "f32", order: "gemma" | "llama" = "gemma"): RmsSteps {
  const cast = dtype === "bf16" ? toBf16 : fround;
  const xf = x.map(fround);
  const sq = xf.map((v) => fround(v * v));
  const meanSq = meanSqF32(xf);
  const plusEps = fround(meanSq + eps);
  const r = fround(1 / fround(Math.sqrt(plusEps)));   // torch.rsqrt on CPU: float32 sqrt, then 1/x
  const normed = xf.map((v) => fround(v * r));
  const scale = w.map((v) => fround(1 + fround(v)));
  let out: Vec;
  if (order === "gemma") out = normed.map((v, i) => cast(fround(v * scale[i])));
  else out = normed.map((v, i) => cast(cast(v) * cast(scale[i])));
  return { x: xf, sq, meanSq, plusEps, rsqrt: r, normed, scale, out };
}

export const rmsnorm = (x: Vec, w: Vec, eps = 1e-6, dtype: DType = "f32", order: "gemma" | "llama" = "gemma") =>
  rmsnormSteps(x, w, eps, dtype, order).out;

export function layernorm(x: Vec, eps = 1e-6, gamma?: Vec, beta?: Vec): Vec {
  const m = mean(x);
  const v = mean(x.map((a) => (a - m) ** 2));
  const r = 1 / Math.sqrt(v + eps);
  return x.map((a, i) => (a - m) * r * (gamma ? gamma[i] : 1) + (beta ? beta[i] : 0));
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
export const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)));

// Abramowitz-Stegun 7.1.26 is not precise enough; erf uses a series / continued-fraction combination
export function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  if (a < 2.5) {
    // Taylor serisi
    let term = a, total = a;
    for (let n = 1; n < 60; n++) {
      term *= (-a * a) / n;
      total += term / (2 * n + 1);
    }
    return (s * 2 * total) / Math.sqrt(Math.PI);
  }
  // asymptotic erfc expansion for large |x|
  let t = 1, sumv = 1;
  for (let n = 1; n < 8; n++) {
    t *= -(2 * n - 1) / (2 * a * a);
    sumv += t;
  }
  const erfc = (Math.exp(-a * a) / (a * Math.sqrt(Math.PI))) * sumv;
  return s * (1 - erfc);
}
export const geluErf = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
export const silu = (x: number) => x / (1 + Math.exp(-x));
export const relu = (x: number) => Math.max(0, x);

export function softmax(v: Vec): Vec {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = sum(e);
  return e.map((x) => x / s);
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** y = W x, W: [out][in] */
export const matvec = (W: Vec[], x: Vec): Vec => W.map((row) => dot(row, x));

/** Reproducible random numbers (mulberry32) */
export function rng(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(next(), 1e-12), v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { next, normal };
}

export function histogram(v: ArrayLike<number>, bins: number, lo?: number, hi?: number) {
  let a = lo ?? Infinity, b = hi ?? -Infinity;
  if (lo === undefined || hi === undefined) for (let i = 0; i < v.length; i++) { a = Math.min(a, v[i]); b = Math.max(b, v[i]); }
  const counts = new Array(bins).fill(0);
  const w = (b - a) / bins || 1;
  for (let i = 0; i < v.length; i++) {
    const k = Math.min(bins - 1, Math.max(0, Math.floor((v[i] - a) / w)));
    counts[k]++;
  }
  return { counts, lo: a, hi: b, width: w };
}

/** A number in readable form: 2753597168 -> "2.75 B" */
export function fmtCount(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " K";
  return String(n);
}
export const fmtShape = (s: number[]) => "[" + s.join(", ") + "]";
export const fmtNum = (x: number, d = 3) =>
  Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d);
