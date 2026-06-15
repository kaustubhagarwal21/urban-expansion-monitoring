import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type PresenterPayload } from "../api";
import { Loading, SectionHead, useApi } from "../ui";

export default function Presenter() {
  const { data } = useApi<PresenterPayload>(api.presenter, []);
  const [step, setStep] = useState(0);

  const cumulative = useMemo(() => {
    if (!data) return [];
    let acc = 0;
    return data.beats.map((b) => {
      const start = acc;
      acc += b.min;
      return start;
    });
  }, [data]);

  if (!data) return <Loading label="loading presenter mode" />;

  const beat = data.beats[step];
  const start = cumulative[step] ?? 0;

  return (
    <div>
      <SectionHead eyebrow={`Talk Script · ~${data.total_minutes} min`} title="Presenter mode">
        Your on-stage teleprompter. Step through the {data.beats.length} beats — each tells you which page to be
        on, what to click, and what to say. Click the page name to jump there in another view.
      </SectionHead>

      {/* timeline */}
      <div className="timeline reveal">
        {data.beats.map((b, i) => (
          <button key={b.n} className={"tl-step" + (i === step ? " on" : "") + (i < step ? " done" : "")} onClick={() => setStep(i)} title={b.page}>
            <span className="tl-n">{b.n}</span>
          </button>
        ))}
      </div>

      {/* current beat */}
      <div className="panel pad reveal beat" style={{ marginTop: 20 }}>
        <div className="beat-head">
          <div>
            <div className="eyebrow">Beat {beat.n} of {data.beats.length} · {start.toFixed(1)}–{(start + beat.min).toFixed(1)} min</div>
            <h2 style={{ fontSize: 26, marginTop: 8 }}>
              <Link to={beat.route} style={{ color: "var(--cyan)" }}>{beat.page} ↗</Link>
            </h2>
          </div>
          <div className="beat-time mono">{beat.min} min</div>
        </div>

        <div className="beat-grid">
          <div className="beat-do">
            <span className="t">do — click</span>
            <p>{beat.click}</p>
          </div>
          <div className="beat-say">
            <span className="t">say</span>
            <p>“{beat.say}”</p>
          </div>
        </div>

        <div className="beat-nav">
          <button className="btn" disabled={step === 0} onClick={() => setStep((s) => Math.max(0, s - 1))}>← prev</button>
          <span className="mono faint" style={{ fontSize: 12 }}>{step + 1} / {data.beats.length}</span>
          <button className="btn primary" disabled={step === data.beats.length - 1} onClick={() => setStep((s) => Math.min(data.beats.length - 1, s + 1))}>next →</button>
        </div>
      </div>

      {/* tips */}
      <div className="panel pad reveal" style={{ marginTop: 18 }}>
        <div className="eyebrow" style={{ marginBottom: 12 }}>Stage tips</div>
        <ul className="tips">
          {data.tips.map((t, i) => <li key={i}>{t}</li>)}
        </ul>
      </div>

      <style>{`
        .timeline { display: flex; gap: 8px; flex-wrap: wrap; }
        .tl-step { width: 40px; height: 40px; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--panel-2); color: var(--text-dim); font-family: var(--font-mono); transition: all 0.15s; }
        .tl-step:hover { border-color: var(--cyan); color: var(--text); }
        .tl-step.on { background: var(--cyan); color: #04110f; border-color: var(--cyan); font-weight: 600; }
        .tl-step.done { color: var(--cyan); border-color: var(--cyan); }
        .tl-n { font-size: 14px; }
        .beat-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding-bottom: 18px; border-bottom: 1px solid var(--line); }
        .beat-time { color: var(--text-faint); font-size: 13px; }
        .beat-grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 22px; padding: 22px 0; }
        @media (max-width: 920px) { .beat-grid { grid-template-columns: 1fr; } }
        .beat-do, .beat-say { }
        .beat-do .t, .beat-say .t { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 8px; }
        .beat-do .t { color: var(--amber); }
        .beat-say .t { color: var(--cyan); }
        .beat-do p { color: var(--text-dim); font-size: 14px; line-height: 1.6; margin: 0; }
        .beat-say p { font-family: var(--font-display); font-size: 18px; line-height: 1.55; margin: 0; color: var(--text); }
        .beat-nav { display: flex; align-items: center; justify-content: space-between; padding-top: 18px; border-top: 1px solid var(--line); }
        .tips { margin: 0; padding-left: 18px; color: var(--text-dim); font-size: 13.5px; line-height: 1.8; }
      `}</style>
    </div>
  );
}
