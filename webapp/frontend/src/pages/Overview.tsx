import { Link } from "react-router-dom";
import { api } from "../api";
import { CountUp, Loading, pct, useApi } from "../ui";

const STAGE_ROUTES: Record<string, string> = {
  data: "/classify",
  classify: "/classify",
  timeseries: "/growth",
  forecast: "/forecast",
  alerts: "/alerts",
};

export default function Overview() {
  const { data, error } = useApi(api.overview, []);
  if (error) return <p className="muted">Backend offline — start the FastAPI server. ({error})</p>;
  if (!data) return <Loading label="loading mission overview" />;

  const h = data.headline;
  const stats = [
    { k: "Best model", v: h.best_model, num: null as number | null, u: `${pct(h.best_oa)} ± ${(h.best_oa_std * 100).toFixed(1)}% OA` },
    { k: "Deep models", v: "", num: h.n_models, u: "benchmarked, 3 seeds each" },
    { k: "Cities", v: "", num: h.n_cities, u: "Mumbai · Delhi NCR · Bangalore" },
    { k: "Live alerts", v: "", num: h.n_alerts, u: "encroachment events flagged" },
  ];

  return (
    <div>
      {/* hero */}
      <div className="reveal" style={{ marginBottom: 30 }}>
        <div className="eyebrow">{data.subtitle}</div>
        <h1 className="grad-text" style={{ fontSize: 46, lineHeight: 1.04, margin: "12px 0 0", maxWidth: "18ch" }}>{data.title}</h1>
        <p className="muted" style={{ maxWidth: "66ch", marginTop: 14, lineHeight: 1.65 }}>
          An end-to-end system that reads decades of satellite imagery, classifies every patch of land,
          tracks how Indian metros sprawl over time, forecasts where they grow next, and raises alerts when
          that growth threatens protected ecological zones.
        </p>
      </div>

      {/* headline stats */}
      <div className="grid cols-4" style={{ marginBottom: 34 }}>
        {stats.map((s, i) => (
          <div key={s.k} className="panel stat reveal" style={{ animationDelay: `${0.05 * i}s` }}>
            <div className="k">{s.k}</div>
            <div className="v">{s.num != null ? <CountUp value={s.num} /> : s.v}</div>
            <div className="u">{s.u}</div>
          </div>
        ))}
      </div>

      {/* pipeline */}
      <div className="eyebrow reveal" style={{ marginBottom: 14 }}>How the system works · click a stage to explore</div>
      <div className="pipeline">
        {data.pipeline.map((st, i) => (
          <Link to={STAGE_ROUTES[st.id] || "/overview"} key={st.id} className="stage reveal" style={{ animationDelay: `${0.08 * i}s` }}>
            <div className="stage-idx mono">{String(i + 1).padStart(2, "0")}</div>
            <h3>{st.name}</h3>
            <p>{st.desc}</p>
            <div className="stage-tech">
              {st.tech.map((t) => (
                <span key={t} className="tag">{t}</span>
              ))}
            </div>
            {i < data.pipeline.length - 1 && <div className="stage-arrow" style={{ animationDelay: `${0.18 * i}s` }}>↓</div>}
          </Link>
        ))}
      </div>

      {/* official paper figure */}
      <div className="panel pad reveal" style={{ marginTop: 30 }}>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Paper Figure 1 · Five-pillar architecture</div>
        <img src={api.figure("fig1_architecture.png")} alt="Five-pillar framework architecture"
          style={{ width: "100%", borderRadius: 10, border: "1px solid var(--line)", background: "#fff" }} />
      </div>

      <style>{`
        .pipeline { display: grid; gap: 14px; }
        .stage {
          position: relative; display: block;
          background: linear-gradient(180deg, var(--panel), var(--ink-2));
          border: 1px solid var(--line); border-radius: var(--radius);
          padding: 20px 24px; transition: all 0.2s ease;
        }
        .stage:hover { border-color: var(--cyan); transform: translateX(6px); box-shadow: var(--shadow); }
        .stage-idx { position: absolute; top: 18px; right: 22px; font-size: 30px; color: var(--line-strong); }
        .stage h3 { font-size: 21px; }
        .stage p { color: var(--text-dim); margin: 8px 0 14px; max-width: 78ch; line-height: 1.6; font-size: 14px; }
        .stage-tech { display: flex; gap: 8px; flex-wrap: wrap; }
        .stage-arrow {
          position: absolute; left: 50%; bottom: -16px;
          color: var(--cyan); font-size: 18px; z-index: 2;
          animation: flowDown 1.7s ease-in-out infinite;
        }
        @keyframes flowDown {
          0%, 100% { transform: translate(-50%, -3px); opacity: 0.35; }
          50% { transform: translate(-50%, 3px); opacity: 1; text-shadow: 0 0 10px var(--cyan); }
        }
      `}</style>
    </div>
  );
}
