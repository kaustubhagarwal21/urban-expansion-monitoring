import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type TourSlide } from "../api";

export default function Welcome({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [slides, setSlides] = useState<TourSlide[]>([]);
  const [i, setI] = useState(0);
  const nav = useNavigate();

  useEffect(() => {
    if (open && slides.length === 0) api.tour().then((d) => setSlides(d.slides)).catch(() => {});
    if (open) setI(0);
  }, [open, slides.length]);

  if (!open || slides.length === 0) return null;
  const s = slides[i];
  const last = i === slides.length - 1;

  const go = (route: string) => { onClose(); nav(route); };

  return (
    <div className="welcome-ov" onClick={onClose}>
      <div className="welcome" onClick={(e) => e.stopPropagation()}>
        <button className="w-skip" onClick={onClose}>skip ✕</button>

        <div className="w-kicker">{s.kicker}</div>
        <h2 className="w-title">{s.title}</h2>

        {s.body && <p className="w-body">{s.body}</p>}
        {s.points && (
          <ol className="w-points">
            {s.points.map((p, k) => (
              <li key={k}><span className="w-num">{k + 1}</span>{p}</li>
            ))}
          </ol>
        )}

        {s.goto && (
          <button className="btn primary w-goto" onClick={() => go(s.goto!.route)}>{s.goto.label} →</button>
        )}

        <div className="w-foot">
          <button className="btn w-nav" disabled={i === 0} onClick={() => setI((x) => Math.max(0, x - 1))}>← back</button>
          <div className="w-dots">
            {slides.map((_, k) => (
              <span key={k} className={"w-dot" + (k === i ? " on" : "")} onClick={() => setI(k)} />
            ))}
          </div>
          {last ? (
            <button className="btn primary w-nav" onClick={onClose}>Got it</button>
          ) : (
            <button className="btn primary w-nav" onClick={() => setI((x) => Math.min(slides.length - 1, x + 1))}>next →</button>
          )}
        </div>
      </div>

      <style>{`
        .welcome-ov { position: fixed; inset: 0; z-index: 100; background: rgba(4,7,11,0.82); backdrop-filter: blur(8px); display: flex; align-items: center; justify-content: center; padding: 24px; animation: fadeUp 0.3s ease; }
        .welcome { position: relative; width: 100%; max-width: 560px; background: linear-gradient(180deg, var(--panel), var(--ink-2)); border: 1px solid var(--cyan); border-radius: 18px; padding: 34px 34px 22px; box-shadow: 0 30px 80px -20px rgba(0,0,0,0.9), 0 0 60px -24px var(--cyan); }
        .w-skip { position: absolute; top: 16px; right: 18px; background: transparent; border: 0; color: var(--text-faint); font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em; }
        .w-skip:hover { color: var(--text); }
        .w-kicker { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--cyan); }
        .w-title { font-family: var(--font-display); font-size: 28px; line-height: 1.15; margin: 10px 0 16px; }
        .w-body { color: var(--text-dim); font-size: 15px; line-height: 1.7; margin: 0; }
        .w-points { list-style: none; margin: 4px 0 0; padding: 0; }
        .w-points li { display: flex; gap: 12px; align-items: flex-start; color: var(--text-dim); font-size: 14.5px; line-height: 1.55; padding: 9px 0; border-bottom: 1px solid var(--line); }
        .w-points li:last-child { border-bottom: 0; }
        .w-num { flex: none; width: 24px; height: 24px; border-radius: 50%; background: rgba(63,212,196,0.15); color: var(--cyan); font-family: var(--font-mono); font-size: 12px; display: flex; align-items: center; justify-content: center; margin-top: 1px; }
        .w-goto { margin-top: 18px; }
        .w-foot { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--line); }
        .w-nav { min-width: 84px; }
        .w-dots { display: flex; gap: 8px; }
        .w-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--panel-3); cursor: pointer; transition: all 0.15s; }
        .w-dot.on { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
      `}</style>
    </div>
  );
}
