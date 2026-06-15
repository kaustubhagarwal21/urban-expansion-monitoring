import { api, type LimitationsPayload } from "../api";
import { Glossarize, Loading, SectionHead, useApi } from "../ui";

const TAG_COLOR: Record<string, string> = {
  Scope: "#57b6ef",
  Data: "#f6a93b",
  Method: "#9b8cf0",
  Evaluation: "#3fd4c4",
  Ethics: "#ef5d5d",
};

export default function Limitations() {
  const { data } = useApi<LimitationsPayload>(api.limitations, []);
  if (!data) return <Loading label="loading limitations" />;

  return (
    <div>
      <SectionHead eyebrow="Owning the Limits" title="Limitations & future work">
        Honest scoping builds trust. Each limitation is paired with how the paper handles it today and what
        comes next — so when a reviewer raises one, you've already answered it.
      </SectionHead>

      <div className="grid cols-2">
        {data.limitations.map((l, i) => {
          const c = TAG_COLOR[l.tag] || "#3fd4c4";
          return (
            <div key={l.title} className="panel pad reveal lim" style={{ animationDelay: `${0.04 * i}s`, borderTop: `2px solid ${c}` }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <h3 style={{ fontSize: 18 }}>{l.title}</h3>
                <span className="lim-tag" style={{ color: c, borderColor: c + "55" }}>{l.tag}</span>
              </div>
              <p className="lim-what"><Glossarize text={l.what} /></p>
              <div className="lim-row"><span className="t" style={{ color: "var(--green)" }}>how we handle it</span><Glossarize text={l.mitigation} /></div>
              <div className="lim-row"><span className="t" style={{ color: "var(--cyan)" }}>future work</span><Glossarize text={l.future} /></div>
            </div>
          );
        })}
      </div>

      <style>{`
        .lim-tag { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid; flex: none; }
        .lim-what { color: var(--text); font-size: 14px; line-height: 1.55; margin: 10px 0 14px; }
        .lim-row { font-size: 13px; line-height: 1.6; color: var(--text-dim); padding-top: 12px; border-top: 1px solid var(--line); }
        .lim-row .t { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px; }
      `}</style>
    </div>
  );
}
