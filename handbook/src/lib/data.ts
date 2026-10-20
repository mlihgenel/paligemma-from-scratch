// public/data/*.json yükleyici. {"$bin", "shape", "dtype"} nesnelerini typed array'e çevirir.
import { useEffect, useState } from "react";

export interface Arr { data: Float32Array | Uint8Array | Int32Array; shape: number[] }

const BASE = import.meta.env.BASE_URL + "data/";
const cache = new Map<string, Promise<any>>();

function f16ToF32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}

async function resolve(node: any): Promise<any> {
  if (Array.isArray(node)) return Promise.all(node.map(resolve));
  if (node && typeof node === "object") {
    if ("$bin" in node) {
      const buf = await (await fetch(BASE + node.$bin)).arrayBuffer();
      const data = node.dtype === "f16" ? f16ToF32(new Uint16Array(buf))
        : node.dtype === "u8" ? new Uint8Array(buf)
        : node.dtype === "i32" ? new Int32Array(buf) : new Float32Array(buf);
      return { data, shape: node.shape } as Arr;
    }
    const out: any = {};
    await Promise.all(Object.entries(node).map(async ([k, v]) => { out[k] = await resolve(v); }));
    return out;
  }
  return node;
}

export function loadData<T = any>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(name, fetch(BASE + name + ".json").then((r) => {
      if (!r.ok) throw new Error(`${name}.json bulunamadı. handbook/export scriptlerini çalıştır.`);
      return r.json();
    }).then(resolve));
  }
  return cache.get(name)!;
}

export function useData<T = any>(name: string): { data: T | null; error: string | null } {
  const [state, set] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    loadData<T>(name).then((d) => alive && set({ data: d, error: null }), (e) => alive && set({ data: null, error: String(e.message ?? e) }));
    return () => { alive = false; };
  }, [name]);
  return state;
}

/** Arr'dan bir satır (son eksen boyunca) */
export function rowOf(a: Arr, ...idx: number[]): Float32Array {
  const shape = a.shape;
  const last = shape[shape.length - 1];
  let off = 0, stride = a.data.length;
  for (let i = 0; i < idx.length; i++) { stride /= shape[i]; off += idx[i] * stride; }
  return (a.data as Float32Array).subarray(off, off + (idx.length === shape.length - 1 ? last : stride));
}
