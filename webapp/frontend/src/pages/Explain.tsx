import { useMemo, useState } from "react";
import { api, type ExplainPayload } from "../api";
import { Loading, SectionHead, useApi } from "../ui";

export default function Explain() {
  const { data } = useApi<ExplainPayload>(api.explain, []);
  const [active, setActive] = useState<string>("abstract");
  const [query, setQuery] = useState("");

  const q = query.trim().toLowerCase();

  const filtered = useMemo(() => {
    if (!data) return [];
    if (!q) return data.sections;
    return data.sections
      .map((s) => ({
        ...s,
        blocks: s.blocks.filter((b) => b.q.toLowerCase().includes(q) || b.p.toLowerCase().includes(q)),
      }))
      .filter((s) => s.blocks.length > 0 || s.title.toLowerCase().includes(q));
  }, [data, q]);

  const glossaryHits = useMemo(() => {
    if (!data) return [];
    if (!q) return data.glossary;
    return data.glossary.filter((g) => g.t.toLowerCase().includes(q) || g.p.toLowerCase().includes(q));
  }, [data, q]);

  if (!data) return <Loading label="loading paper companion" />;

  return (
    <div>
      <SectionHead eyebrow="Presenter Companion" title="Paper, explained line by line">
        Every key line of the IEEE paper paired with a plain-English translation — your live teleprompter for
        CHANDICON. Use the search to jump to any word or phrase; hover the glossary for any term.
      </SectionHead>

      <input
        className="explain-search"
        placeholder="Search any word or line from the paper…  (e.g. LOCO, focal loss, Transition, MC Dropout)"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className="grid" style={{ gridTemplateColumns: "210px 1fr", alignItems: "start", marginTop: 18 }}>
        {/* section rail */}
        <nav className="explain-rail panel pad">
          <div className="eyebrow" style={{ marginBottom: 12 }}>Sections</div>
          {data.sections.map((s) => (
            <a
              key={s.id}
              href={`#sec-${s.id}`}
              className={"rail-item" + (active === s.id ? " on" : "")}
              onClick={() => setActive(s.id)}
            >
              <span className="mono rn">{s.number}</span> {s.title.split(" — ")[0]}
            </a>
          ))}
          <a href="#glossary" className="rail-item" onClick={() => setActive("glossary")}>
            <span className="mono rn">★</span> Glossary
          </a>
        </nav>

        {/* content */}
        <div className="grid" style={{ gap: 18 }}>
          {q && filtered.length === 0 && glossaryHits.length === 0 && (
            <div className="panel pad muted">No matches for “{query}”.</div>
          )}

          {filtered.map((s) => (
            <section key={s.id} id={`sec-${s.id}`} className="panel pad reveal">
              <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
                <span className="mono" style={{ color: "var(--cyan)", fontSize: 13 }}>{s.number}</span>
                <h3 style={{ fontSize: 22 }}>{s.title}</h3>
              </div>
              <p className="faint" style={{ margin: "0 0 18px", fontSize: 13 }}>{s.summary}</p>
              {s.blocks.map((b, i) => (
                <div key={i} className="block">
                  <blockquote className="paper-quote">{b.q}</blockquote>
                  <div className="plain"><span className="plain-tag">in plain words</span>{b.p}</div>
                </div>
              ))}
            </section>
          ))}

          {/* glossary */}
          {(!q || glossaryHits.length > 0) && (
            <section id="glossary" className="panel pad reveal">
              <div className="eyebrow" style={{ marginBottom: 14 }}>Glossary · {glossaryHits.length} terms</div>
              <div className="gloss-grid">
                {glossaryHits.map((g) => (
                  <div key={g.t} className="gloss">
                    <div className="gt">{g.t}</div>
                    <div className="gp">{g.p}</div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      <style>{`
        .explain-search {
          width: 100%; padding: 13px 16px; border-radius: 11px;
          background: var(--panel-2); border: 1px solid var(--line-strong); color: var(--text);
          font-family: var(--font-mono); font-size: 13px;
        }
        .explain-search:focus { outline: none; border-color: var(--cyan); box-shadow: 0 0 0 1px var(--cyan); }
        .explain-search::placeholder { color: var(--text-faint); }
        .explain-rail { position: sticky; top: 90px; }
        .rail-item { display: block; padding: 8px 10px; border-radius: 8px; font-size: 13px; color: var(--text-dim); transition: all 0.15s; }
        .rail-item:hover { background: rgba(120,165,190,0.06); color: var(--text); }
        .rail-item.on { color: var(--cyan); }
        .rail-item .rn { color: var(--text-faint); margin-right: 6px; font-size: 11px; }
        .block { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; padding: 16px 0; border-top: 1px solid var(--line); }
        @media (max-width: 920px) { .block { grid-template-columns: 1fr; } }
        .block:first-of-type { border-top: 0; }
        .paper-quote {
          margin: 0; padding: 0 0 0 16px; border-left: 2px solid var(--line-strong);
          font-family: var(--font-display); font-size: 15px; line-height: 1.55; color: var(--text);
        }
        .plain { font-size: 14px; line-height: 1.65; color: var(--text-dim); }
        .plain-tag { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--cyan); margin-bottom: 6px; }
        .gloss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 28px; }
        @media (max-width: 920px) { .gloss-grid { grid-template-columns: 1fr; } }
        .gloss { padding-bottom: 12px; border-bottom: 1px solid var(--line); }
        .gt { font-family: var(--font-mono); font-size: 13px; color: var(--cyan); margin-bottom: 4px; }
        .gp { font-size: 13px; color: var(--text-dim); line-height: 1.55; }
      `}</style>
    </div>
  );
}
