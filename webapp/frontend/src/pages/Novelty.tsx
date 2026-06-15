import { api, type NoveltyPayload } from "../api";
import { Glossarize, Loading, SectionHead, useApi } from "../ui";

const TAG_COLOR: Record<string, string> = {
  "First-of-its-kind": "#3fd4c4",
  "Surprising finding": "#f6a93b",
  "Systems": "#9b8cf0",
  "Method": "#57b6ef",
  "Rigor": "#5fcf8c",
  "Insight": "#e8c34a",
};

export default function Novelty() {
  const { data } = useApi<NoveltyPayload>(api.novelty, []);
  if (!data) return <Loading label="loading novelty" />;

  return (
    <div>
      <SectionHead eyebrow="The Contribution" title="What makes this novel">
        Not a new architecture — a complete, India-specific, statistically-rigorous system. Below: the pitch,
        the distinct novelty pillars (each with the “no prior work has this” angle and the evidence), and how
        it sits against the state of the art.
      </SectionHead>

      {/* pitch */}
      <div className="panel pad reveal" style={{ marginBottom: 22, borderLeft: "3px solid var(--cyan)" }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>30-second pitch</div>
        <p className="display" style={{ fontSize: 20, lineHeight: 1.5, margin: 0 }}><Glossarize text={data.pitch} /></p>
      </div>

      {/* pillars */}
      <div className="grid cols-2">
        {data.pillars.map((p, i) => (
          <div key={p.title} className="panel pad reveal nov" style={{ animationDelay: `${0.04 * i}s` }}>
            <span className="nov-tag" style={{ color: TAG_COLOR[p.tag] || "var(--cyan)", borderColor: (TAG_COLOR[p.tag] || "#3fd4c4") + "66" }}>{p.tag}</span>
            <h3 style={{ fontSize: 19, margin: "12px 0 8px" }}>{p.title}</h3>
            <p className="nov-what"><Glossarize text={p.what} /></p>
            <div className="nov-why"><span className="t">no prior work</span><Glossarize text={p.why_novel} /></div>
            <div className="nov-ev"><span className="t">evidence</span>{p.evidence}</div>
          </div>
        ))}
      </div>

      {/* positioning */}
      <div className="panel pad reveal" style={{ marginTop: 22 }}>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Positioning vs. state of the art</div>
        <table className="table">
          <tbody>
            {data.positioning.map((p) => (
              <tr key={p.area}>
                <td className="mono" style={{ color: "var(--cyan)", whiteSpace: "nowrap", verticalAlign: "top", width: 180 }}>{p.area}</td>
                <td className="muted">{p.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <style>{`
        .nov { display: flex; flex-direction: column; }
        .nov-tag { align-self: flex-start; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; padding: 4px 10px; border-radius: 999px; border: 1px solid; }
        .nov-what { color: var(--text); font-size: 14px; line-height: 1.6; margin: 0 0 14px; }
        .nov-why, .nov-ev { font-size: 13px; line-height: 1.55; color: var(--text-dim); padding-top: 12px; border-top: 1px solid var(--line); margin-top: auto; }
        .nov-ev { margin-top: 12px; }
        .nov-why .t, .nov-ev .t { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-faint); margin-bottom: 5px; }
      `}</style>
    </div>
  );
}
