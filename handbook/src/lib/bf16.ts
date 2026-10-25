// bfloat16 emulation. bf16 = the top 16 bits of float32: 1 sign, 8 exponent, 7 mantissa bits.
// float32 -> bf16 conversion uses "round to nearest, ties to even", like PyTorch.
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export const fround = Math.fround;

export function toBf16(x: number): number {
  f32[0] = x;
  let u = u32[0];
  if ((u & 0x7f800000) === 0x7f800000 && (u & 0x7fffff) !== 0) return NaN;
  const lsb = (u >>> 16) & 1;
  u = (u + 0x7fff + lsb) >>> 0;
  u32[0] = u & 0xffff0000;
  return f32[0];
}

/** The bits of a number: float32 (32) or bf16 (16) */
export function bits(x: number, kind: "f32" | "bf16"): { sign: string; exp: string; mant: string } {
  f32[0] = kind === "bf16" ? toBf16(x) : x;
  const s = u32[0].toString(2).padStart(32, "0");
  return kind === "bf16"
    ? { sign: s[0], exp: s.slice(1, 9), mant: s.slice(9, 16) }
    : { sign: s[0], exp: s.slice(1, 9), mant: s.slice(9) };
}

/** The gap between x's neighboring bf16 values (1 ulp) */
export function bf16Ulp(x: number): number {
  const b = toBf16(Math.abs(x));
  if (b === 0) return 0;
  const e = Math.floor(Math.log2(b));
  return Math.pow(2, e - 7);
}
