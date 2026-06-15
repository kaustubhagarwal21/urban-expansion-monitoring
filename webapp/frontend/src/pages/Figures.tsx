import { useMemo, useState } from "react";
import { api, type FigureItem, type FiguresPayload } from "../api";
import { Glossarize, Loading, ReadThis, SectionHead, useApi } from "../ui";

export default function Figures() {
  const { data } = useApi<FiguresPayload>(api.figures, []);
  const [zoom, setZoom] = useState<FigureItem | null>(null);

  const groups = useMemo(() => {
    if (!data) return [];
    const order = ["Architecture", "Data", "Results", "Explainability", "Analysis"];
    const by: Record<string, FigureItem[]> = {};
    data.figures.forEach((f) => (by[f.group] ||= []).push(f));
    return order.filter((g) => by[g]).map((g) => ({ group: g, items: by[g] }));
  }, [data]);

  if (!data) return <Loading label="loading figures" />;

  return (
    <div>
      <SectionHead eyebrow="Results Gallery" title="Paper figures">
        Every figure is a real result from the study — click any to enlarge. Use these as backup slides or to
        answer a question with the exact chart.
      </SectionHead>

      <ReadThis>
        Figures are grouped by theme (Architecture → Data → Results → Explainability → Analysis), roughly the
        order of the paper. <b>Click any figure to open it large</b> — the zoomed view explains <b>how to read
        the chart</b> and gives the <b>one-line takeaway</b>, so you can narrate any figure even if a question
        comes from left field.
      </ReadThis>

      {groups.map((grp, gi) => (
        <div key={grp.group} className="reveal" style={{ marginBottom: 26, animationDelay: `${0.04 * gi}s` }}>
          <div className="eyebrow" style={{ marginBottom: 14 }}>{grp.group}</div>
          <div className="fig-grid">
            {grp.items.map((f) => (
              <button key={f.file} className="fig-card" onClick={() => setZoom(f)}>
                <div className="fig-img"><img src={api.figure(f.file)} alt={f.title} loading="lazy" /></div>
                <div className="fig-meta">
                  <div className="fig-title">{f.title}</div>
                  <div className="fig-cap">{f.caption}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <div className="lb-inner" onClick={(e) => e.stopPropagation()}>
            <button className="lb-close" onClick={() => setZoom(null)}>✕</button>
            <img src={api.figure(zoom.file)} alt={zoom.title} />
            <div className="lb-cap">
              <b>{zoom.title}</b>
              <span>{zoom.caption}</span>
              <div className="lb-detail">
                <div className="lb-box"><span className="lb-tag">How to read it</span><Glossarize text={zoom.read} /></div>
                <div className="lb-box take"><span className="lb-tag">Takeaway</span><Glossarize text={zoom.takeaway} /></div>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .fig-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        @media (max-width: 1000px) { .fig-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 640px) { .fig-grid { grid-template-columns: 1fr; } }
        .fig-card { text-align: left; padding: 0; background: linear-gradient(180deg, var(--panel), var(--ink-2)); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; transition: all 0.18s; display: flex; flex-direction: column; }
        .fig-card:hover { border-color: var(--cyan); transform: translateY(-3px); box-shadow: var(--shadow); }
        .fig-img { background: #fff; aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .fig-img img { width: 100%; height: 100%; object-fit: contain; }
        .fig-meta { padding: 12px 14px; }
        .fig-title { font-family: var(--font-display); font-size: 15px; }
        .fig-cap { color: var(--text-dim); font-size: 12px; line-height: 1.5; margin-top: 5px; }
        .lightbox { position: fixed; inset: 0; z-index: 50; background: rgba(4,7,11,0.86); backdrop-filter: blur(6px); display: flex; align-items: center; justify-content: center; padding: 40px; animation: fadeUp 0.25s ease; }
        .lb-inner { position: relative; max-width: 1000px; width: 100%; }
        .lb-inner img { width: 100%; max-height: 76vh; object-fit: contain; background: #fff; border-radius: 12px; border: 1px solid var(--line-strong); }
        .lb-close { position: absolute; top: -14px; right: -14px; width: 36px; height: 36px; border-radius: 50%; background: var(--panel-2); border: 1px solid var(--line-strong); color: var(--text); font-size: 16px; }
        .lb-close:hover { border-color: var(--cyan); color: var(--cyan); }
        .lb-cap { margin-top: 14px; text-align: center; }
        .lb-cap b { font-family: var(--font-display); font-size: 18px; }
        .lb-cap span { display: block; color: var(--text-dim); font-size: 13px; margin-top: 6px; max-width: 70ch; margin-left: auto; margin-right: auto; line-height: 1.6; }
        .lb-detail { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; max-width: 820px; margin: 14px auto 0; text-align: left; }
        @media (max-width: 720px) { .lb-detail { grid-template-columns: 1fr; } }
        .lb-box { background: var(--panel-2); border: 1px solid var(--line-strong); border-radius: 10px; padding: 11px 13px; color: var(--text-dim); font-size: 13px; line-height: 1.6; }
        .lb-box.take { background: rgba(63,212,196,0.08); }
        .lb-tag { display: block; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--cyan); margin-bottom: 6px; }
      `}</style>
    </div>
  );
}
