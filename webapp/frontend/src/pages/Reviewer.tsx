import { useMemo, useState } from "react";
import { api, SEVERITY_QA_COLORS, type ReviewerPayload } from "../api";
import { Loading, SectionHead, useApi } from "../ui";

const ORDER = ["CRITICAL", "MODERATE", "MINOR"];

export default function Reviewer() {
  const { data } = useApi<ReviewerPayload>(api.reviewer, []);
  const [open, setOpen] = useState<number | null>(1);
  const [sev, setSev] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const q = query.trim().toLowerCase();

  const list = useMemo(() => {
    if (!data) return [];
    return data.qa.filter((x) => {
      if (sev && x.severity !== sev) return false;
      if (!q) return true;
      return (
        x.question.toLowerCase().includes(q) ||
        x.plain.toLowerCase().includes(q) ||
        x.one_liner.toLowerCase().includes(q) ||
        x.points.some((p) => p.toLowerCase().includes(q))
      );
    });
  }, [data, sev, q]);

  if (!data) return <Loading label="loading reviewer playbook" />;

  const counts = ORDER.map((s) => ({ s, n: data.qa.filter((x) => x.severity === s).length }));

  return (
    <div>
      <SectionHead eyebrow="Q&A Playbook" title="Anticipated reviewer questions">
        Every likely question at CHANDICON, with a one-line plain answer, the supporting points, and a punchy
        “if pressed” comeback. Click any question to expand it.
      </SectionHead>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginBottom: 18 }}>
        <div className="seg">
          <button className={!sev ? "on" : ""} onClick={() => setSev(null)}>ALL {data.qa.length}</button>
          {counts.map((c) => (
            <button key={c.s} className={sev === c.s ? "on" : ""} onClick={() => setSev(c.s)}
              style={{ color: sev === c.s ? SEVERITY_QA_COLORS[c.s] : undefined }}>
              {c.s} {c.n}
            </button>
          ))}
        </div>
        <input className="rev-search" placeholder="Search questions…" value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>

      <div className="grid" style={{ gap: 12 }}>
        {list.map((x) => {
          const isOpen = open === x.id;
          const color = SEVERITY_QA_COLORS[x.severity];
          return (
            <div key={x.id} className={"qa panel" + (isOpen ? " open" : "")} onClick={() => setOpen(isOpen ? null : x.id)}>
              <div className="qa-head">
                <span className="qa-sev" style={{ background: color }}>{x.severity[0]}</span>
                <div style={{ flex: 1 }}>
                  <div className="qa-q">{x.question}</div>
                  {!isOpen && <div className="qa-plain mono">{x.plain}</div>}
                </div>
                <span className="qa-toggle">{isOpen ? "–" : "+"}</span>
              </div>

              {isOpen && (
                <div className="qa-body" onClick={(e) => e.stopPropagation()}>
                  <div className="qa-aside mono">Likely from: {x.asker}</div>
                  <div className="qa-answer"><span className="plain-tag">plain answer</span>{x.plain}</div>
                  <div className="eyebrow" style={{ margin: "16px 0 8px" }}>Supporting points</div>
                  <ul className="qa-points">
                    {x.points.map((p, i) => <li key={i}>{p}</li>)}
                  </ul>
                  <div className="qa-oneliner">
                    <span className="ol-tag">if pressed →</span> {x.one_liner}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* venue tips */}
      <div className="panel pad reveal" style={{ marginTop: 22 }}>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Venue-specific framing</div>
        <table className="table">
          <thead><tr><th>Venue</th><th>What they want</th><th>Lead with</th></tr></thead>
          <tbody>
            {data.venue_tips.map((v) => (
              <tr key={v.venue}>
                <td className="mono" style={{ color: "var(--cyan)" }}>{v.venue}</td>
                <td className="muted">{v.want}</td>
                <td>{v.tip}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <style>{`
        .rev-search { flex: 1; min-width: 220px; padding: 10px 14px; border-radius: 9px; background: var(--panel-2); border: 1px solid var(--line-strong); color: var(--text); font-family: var(--font-mono); font-size: 12px; }
        .rev-search:focus { outline: none; border-color: var(--cyan); }
        .rev-search::placeholder { color: var(--text-faint); }
        .qa { padding: 16px 18px; cursor: pointer; transition: border-color 0.15s; }
        .qa:hover { border-color: var(--line-strong); }
        .qa.open { border-color: var(--cyan); }
        .qa-head { display: flex; align-items: flex-start; gap: 14px; }
        .qa-sev { flex: none; width: 22px; height: 22px; border-radius: 6px; color: #080b10; font-family: var(--font-mono); font-weight: 600; font-size: 12px; display: flex; align-items: center; justify-content: center; margin-top: 2px; }
        .qa-q { font-family: var(--font-display); font-size: 17px; line-height: 1.35; }
        .qa-plain { font-size: 12px; color: var(--text-faint); margin-top: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .qa-toggle { font-family: var(--font-mono); font-size: 20px; color: var(--text-faint); line-height: 1; }
        .qa-body { padding: 16px 0 4px 36px; cursor: default; }
        .qa-aside { font-size: 11px; color: var(--text-faint); margin-bottom: 12px; }
        .qa-answer { font-size: 14.5px; line-height: 1.6; color: var(--text); }
        .plain-tag { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--cyan); margin-bottom: 6px; }
        .qa-points { margin: 0; padding-left: 18px; color: var(--text-dim); font-size: 13.5px; line-height: 1.7; }
        .qa-oneliner { margin-top: 16px; padding: 12px 14px; border-radius: 9px; background: rgba(63,212,196,0.08); border: 1px solid var(--line-strong); font-size: 14px; line-height: 1.5; }
        .ol-tag { font-family: var(--font-mono); font-size: 11px; color: var(--cyan); letter-spacing: 0.08em; }
      `}</style>
    </div>
  );
}
