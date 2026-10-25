// Scrollytelling: scrolling steps on the left, a sticky figure on the right.
// The figure receives the active step's index and the in-step progress (0–1).
import { Children, ReactElement, ReactNode, isValidElement, useEffect, useRef, useState } from "react";

export function Step({ children }: { children: ReactNode; title?: string }) {
  return <>{children}</>;
}

export function Scrolly({ children, figure, label }: {
  children: ReactNode;
  figure: (step: number, progress: number) => ReactNode;
  label?: ReactNode;
}) {
  const steps = Children.toArray(children).filter(isValidElement) as ReactElement[];
  const refs = useRef<(HTMLDivElement | null)[]>([]);
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const line = window.innerHeight * 0.55;
      let a = 0;
      refs.current.forEach((el, i) => { if (el && el.getBoundingClientRect().top < line) a = i; });
      const el = refs.current[a];
      if (el) {
        const r = el.getBoundingClientRect();
        setProgress(Math.min(1, Math.max(0, (line - r.top) / Math.max(1, r.height))));
      }
      setActive(a);
    };
    const on = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => { window.removeEventListener("scroll", on); window.removeEventListener("resize", on); cancelAnimationFrame(raf); };
  }, [steps.length]);

  return (
    <section className="scrolly wide">
      <div className="scrolly-steps">
        {steps.map((s, i) => (
          <div key={i} ref={(el) => (refs.current[i] = el)} className={"scrolly-step" + (i === active ? " on" : "")}>
            <div className="scrolly-step-inner">
              <span className="scrolly-n">{i + 1}</span>
              {s}
            </div>
          </div>
        ))}
      </div>
      <div className="scrolly-fig">
        <div className="scrolly-fig-inner">
          <div className="scrolly-meta">
            {label && <span>{label}</span>}
            <span className="scrolly-dots" aria-hidden>
              {steps.map((_, i) => <i key={i} className={i === active ? "on" : i < active ? "done" : ""} />)}
            </span>
          </div>
          {figure(active, progress)}
        </div>
      </div>
    </section>
  );
}
