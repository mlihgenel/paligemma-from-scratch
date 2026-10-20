// ../src/*.py dosyalarını derleme anında ham metin olarak okur ve sembol (sınıf/metot) aralığını bulur.
// Satır numarası sabit yazılmadığı için src/ değiştikçe kitap kendiliğinden güncel kalır.
const files = import.meta.glob(["../../../src/*.py", "../../../checks/*.py"], { query: "?raw", import: "default", eager: true }) as Record<string, string>;

// Anahtar dosya adı ("gemma.py", "check_gemma.py"); klasörü DIRS'te
export const SOURCES: Record<string, string> = Object.fromEntries(
  Object.entries(files).map(([p, s]) => [p.split("/").pop()!, s]),
);
export const DIRS: Record<string, string> = Object.fromEntries(
  Object.keys(files).map((p) => { const a = p.split("/"); return [a.pop()!, a.pop()!]; }),
);

export interface Snippet { file: string; start: number; end: number; lines: string[]; found: boolean }

const indentOf = (l: string) => l.length - l.trimStart().length;

/**
 * symbol: "GemmaRMSNorm" (sınıf ya da fonksiyon), "GemmaRMSNorm.forward" (metot).
 * withComments: tanımın hemen üstündeki yorum satırlarını da dahil et.
 */
export function findSymbol(file: string, symbol: string, withComments = true): Snippet {
  const src = SOURCES[file];
  if (!src) return { file, start: 0, end: 0, lines: [`# ${file} bulunamadı`], found: false };
  const lines = src.split("\n");
  const parts = symbol.split(".");
  let lo = 0, hi = lines.length, baseIndent = -1, start = -1;
  for (const part of parts) {
    const re = new RegExp(`^\\s*(class|def)\\s+${part}\\b`);
    start = -1;
    for (let i = lo; i < hi; i++) if (re.test(lines[i]) && (baseIndent < 0 || indentOf(lines[i]) > baseIndent)) { start = i; break; }
    if (start < 0) return { file, start: 0, end: 0, lines: [`# ${symbol} ${file} içinde bulunamadı`], found: false };
    baseIndent = indentOf(lines[start]);
    // Gövde: girintisi tanımdan büyük ya da boş olan satırlar. Çok satırlı imzalar için ")" ile biten satıra kadar devam
    let end = start + 1;
    let inSig = !/:\s*(#.*)?$/.test(lines[start]);
    while (end < hi) {
      const l = lines[end];
      if (inSig) { if (/:\s*(#.*)?$/.test(l)) inSig = false; end++; continue; }
      if (l.trim() === "" || indentOf(l) > baseIndent) { end++; continue; }
      break;
    }
    while (end > start + 1 && lines[end - 1].trim() === "") end--;
    lo = start; hi = end;
  }
  let s = start;
  if (withComments) while (s > 0 && lines[s - 1].trim().startsWith("#") && indentOf(lines[s - 1]) === baseIndent) s--;
  return { file, start: s + 1, end: hi, lines: lines.slice(s, hi).map((l) => l.replace(/\s+$/, "")), found: true };
}

/** Bir dosyada bir metnin geçtiği satır numarası (1 tabanlı), yoksa -1 */
export function lineOf(file: string, needle: string): number {
  const i = (SOURCES[file] ?? "").split("\n").findIndex((l) => l.includes(needle));
  return i < 0 ? -1 : i + 1;
}
