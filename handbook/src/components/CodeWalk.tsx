// Senin src/ kodun, canlı. Not'lu satırlara tıklayınca açıklama açılır ve (varsa) görsel o adıma geçer.
import { ReactNode, useEffect, useMemo, useState } from "react";
import type { HighlighterGeneric } from "shiki";
import { DIRS, findSymbol } from "../lib/pysource";

let hlPromise: Promise<HighlighterGeneric<any, any>> | null = null;
function highlighter() {
  if (!hlPromise) hlPromise = import("shiki").then((m) => m.createHighlighter({ themes: ["github-light", "github-dark"], langs: ["python"] })) as any;
  return hlPromise!;
}

export interface Note { match: string; text: ReactNode; step?: string; title?: string }

interface Props {
  file: string;
  symbol: string;
  notes?: Note[];
  active?: string;                       // dışarıdan seçili adım
  onStep?: (step: string) => void;
  withComments?: boolean;
  maxHeight?: number;
  caption?: ReactNode;
}

export function CodeWalk({ file, symbol, notes = [], active, onStep, withComments = true, maxHeight, caption }: Props) {
  const snip = useMemo(() => findSymbol(file, symbol, withComments), [file, symbol, withComments]);
  const [tokens, setTokens] = useState<any[][] | null>(null);
  const [sel, setSel] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    highlighter().then((hl) => {
      if (!alive) return;
      const r = hl.codeToTokens(snip.lines.join("\n"), { lang: "python", themes: { light: "github-light", dark: "github-dark" } });
      setTokens(r.tokens);
    });
    return () => { alive = false; };
  }, [snip]);

  // Her not, metni içeren ilk satıra bağlanır
  const noteAt = useMemo(() => {
    const m = new Map<number, Note>();
    const missing: Note[] = [];
    for (const n of notes) {
      const i = snip.lines.findIndex((l, k) => l.includes(n.match) && !m.has(k));
      if (i >= 0) m.set(i, n); else missing.push(n);
    }
    return { m, missing };
  }, [notes, snip]);

  // Dışarıdan gelen adım -> o adımın satırını seç
  useEffect(() => {
    if (!active) return;
    for (const [i, n] of noteAt.m) if (n.step === active) { setSel(i); return; }
  }, [active, noteAt]);

  const selNote = sel !== null ? noteAt.m.get(sel) : undefined;
  const hasNotes = notes.length > 0;

  return (
    <figure className="fig codewalk wide" style={{ margin: "24px 0" }}>
      <div className="cw">
        <div className="cw-head">
          <span className="mono"><span className="muted">{DIRS[file] ?? "src"}/</span><b>{file}</b> <span className="muted">· {symbol}</span></span>
          <span className="muted mono">{snip.found ? `satır ${snip.start}–${snip.end}` : "bulunamadı"}</span>
        </div>
        <div className={hasNotes ? "cw-grid" : ""}>
          <pre className="cw-code" style={{ maxHeight }}>
            {(tokens ?? snip.lines.map((l) => [{ content: l }])).map((line, i) => {
              const note = noteAt.m.get(i);
              const on = sel === i;
              return (
                <div key={i} className={"cw-line" + (note ? " has-note" : "") + (on ? " on" : "")}
                  onClick={() => { if (note) { setSel(i); note.step && onStep?.(note.step); } }}>
                  <span className="cw-ln">{snip.start + i}</span>
                  <span className="cw-mark">{note ? "●" : ""}</span>
                  <span className="cw-src">
                    {line.length === 0 ? " " : line.map((t: any, k: number) => <span key={k} style={t.htmlStyle}>{t.content}</span>)}
                  </span>
                </div>
              );
            })}
          </pre>
          {hasNotes && (
            <div className="cw-note">
              {selNote ? (
                <>
                  <div className="eyebrow">{selNote.title ?? `satır ${snip.start + (sel ?? 0)}`}</div>
                  <div style={{ marginTop: 6 }}>{selNote.text}</div>
                  <div className="row" style={{ gap: 6, marginTop: 14 }}>
                    <button className="btn" disabled={!prevNote(noteAt.m, sel)} onClick={() => go(prevNote(noteAt.m, sel))}>← Önceki</button>
                    <button className="btn" disabled={!nextNote(noteAt.m, sel)} onClick={() => go(nextNote(noteAt.m, sel))}>Sonraki →</button>
                  </div>
                </>
              ) : (
                <div className="muted">
                  <span style={{ color: "var(--accent)" }}>●</span> işaretli satırlara tıkla ya da{" "}
                  <button className="btn" style={{ color: "var(--accent)", padding: "2px 4px" }} onClick={() => go(nextNote(noteAt.m, -1))}>baştan oku →</button>
                </div>
              )}
              {noteAt.missing.length > 0 && (
                <div className="small" style={{ color: "var(--warn)", marginTop: 12 }}>
                  Kodun değişmiş olabilir: {noteAt.missing.length} açıklama satırını bulamadı ({noteAt.missing.map((n) => `"${n.match}"`).join(", ")}).
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );

  function go(i: number | null | undefined) {
    if (i === null || i === undefined) return;
    setSel(i);
    const n = noteAt.m.get(i);
    n?.step && onStep?.(n.step);
  }
}

function sortedKeys(m: Map<number, Note>) { return [...m.keys()].sort((a, b) => a - b); }
function nextNote(m: Map<number, Note>, cur: number | null) { return sortedKeys(m).find((k) => k > (cur ?? -1)) ?? null; }
function prevNote(m: Map<number, Note>, cur: number | null) { const ks = sortedKeys(m).filter((k) => k < (cur ?? 0)); return ks.length ? ks[ks.length - 1] : null; }
