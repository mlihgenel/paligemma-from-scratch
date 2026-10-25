// Reads ../src/*.py as raw text at build time and finds a symbol's (class/method) line range.
// Line numbers are not hard-coded, so the book stays up to date as src/ changes.
const files = import.meta.glob(["../../../src/*.py", "../../../checks/*.py"], { query: "?raw", import: "default", eager: true }) as Record<string, string>;

// Keyed by file name ("gemma.py", "check_gemma.py"); the folder is in DIRS
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
  if (!src) return { file, start: 0, end: 0, lines: [`# ${file} not found`], found: false };
  const lines = src.split("\n");
  const parts = symbol.split(".");
  let lo = 0, hi = lines.length, baseIndent = -1, start = -1;
  for (const part of parts) {
    const re = new RegExp(`^\\s*(class|def)\\s+${part}\\b`);
    start = -1;
    for (let i = lo; i < hi; i++) if (re.test(lines[i]) && (baseIndent < 0 || indentOf(lines[i]) > baseIndent)) { start = i; break; }
    if (start < 0) return { file, start: 0, end: 0, lines: [`# ${symbol} not found in ${file}`], found: false };
    baseIndent = indentOf(lines[start]);
    // Body: lines indented more than the definition, or empty. Multi-line signatures continue up to the line ending with ")"
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

/** Line number (1-based) where a text occurs in a file, or -1 */
export function lineOf(file: string, needle: string): number {
  const i = (SOURCES[file] ?? "").split("\n").findIndex((l) => l.includes(needle));
  return i < 0 ? -1 : i + 1;
}
