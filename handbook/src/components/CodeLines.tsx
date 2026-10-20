// Scrolly figürleri için sade kod paneli: bir sembolün satırları, aktif satır(lar) vurgulu.
import { useEffect, useMemo, useRef } from "react";
import { DIRS, findSymbol } from "../lib/pysource";

export function CodeLines({ file, symbol, active = [], maxHeight = 420, withComments = false }: { file: string; symbol: string; active?: string[]; maxHeight?: number; withComments?: boolean }) {
  const snip = useMemo(() => findSymbol(file, symbol, withComments), [file, symbol, withComments]);
  const box = useRef<HTMLDivElement>(null);
  // "metin::n" → metnin n. geçtiği satır (0 tabanlı); düz metin → bütün geçtiği satırlar
  const hits = useMemo(() => {
    const on = new Set<number>();
    for (const a of active) {
      if (!a) continue;
      const [needle, nth] = a.split("::");
      let k = 0;
      snip.lines.forEach((l, i) => { if (l.includes(needle)) { if (nth === undefined || +nth === k) on.add(i); k++; } });
    }
    return on;
  }, [active.join("|"), snip]);
  useEffect(() => {
    const el = box.current?.querySelector("[data-on='true']") as HTMLElement | null;
    if (el && box.current) box.current.scrollTo({ top: el.offsetTop - box.current.clientHeight / 3, behavior: "smooth" });
  }, [active.join("|")]);
  return (
    <div>
      <div className="small mono muted" style={{ marginBottom: 6 }}>{DIRS[file] ?? "src"}/<b style={{ color: "var(--fg)" }}>{file}</b> · {symbol}</div>
      <div ref={box} style={{ maxHeight, overflow: "auto", background: "var(--code-bg)", borderRadius: 6, padding: "10px 0", fontFamily: "var(--mono)", fontSize: 12.6, lineHeight: 1.7, position: "relative" }}>
        {snip.lines.map((l, i) => {
          const on = hits.has(i);
          const comment = l.trim().startsWith("#");
          return (
            <div key={i} data-on={on} style={{ display: "flex", whiteSpace: "pre", padding: "0 14px 0 0", background: on ? "color-mix(in srgb, var(--accent) 14%, transparent)" : undefined, borderLeft: `2px solid ${on ? "var(--accent)" : "transparent"}`, transition: "background .3s" }}>
              <span style={{ width: 40, textAlign: "right", paddingRight: 10, color: "var(--fg-faint)", flex: "none" }}>{snip.start + i}</span>
              <span style={{ color: comment ? "var(--fg-faint)" : on ? "var(--fg)" : "var(--fg-muted)", fontWeight: on ? 600 : 400 }}>{l || " "}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
