import { useMemo, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type Forecasts } from "../api";
import { InfoTip, Loading, ReadThis, SectionHead, useApi } from "../ui";

function ChartTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  return (
    <div className="rc-tip">
      <div className="t">YEAR {label}</div>
      <div className="row"><span style={{ color: "#3fd4c4" }}>forecast</span><span>{Math.round(row.mean).toLocaleString()} km²</span></div>
      <div className="row"><span className="faint">95% CI</span><span>{Math.round(row.band[0]).toLocaleString()}–{Math.round(row.band[1]).toLocaleString()}</span></div>
    </div>
  );
}

export default function Forecast() {
  const { data } = useApi<Forecasts>(api.forecasts, []);
  const [city, setCity] = useState<string | null>(null);

  const cities = data ? Object.keys(data) : [];
  const active = city || cities[0];

  const rows = useMemo(() => {
    if (!data || !active) return [];
    return data[active].map((p) => ({ year: p.year, mean: p.mean, band: [p.ci_lower, p.ci_upper] }));
  }, [data, active]);

  if (!data) return <Loading label="loading forecasts" />;

  const last = rows[rows.length - 1];
  const first = rows[0];
  const growth = last && first ? ((last.mean - first.mean) / first.mean) * 100 : 0;
  const uncertainty = last ? last.band[1] - last.band[0] : 0;

  return (
    <div>
      <SectionHead eyebrow="Stage 04 · Pillar IV — Predictive Modelling" title="Sprawl forecast to 2035">
        A bidirectional LSTM with multi-head temporal attention projects each city's built-up area forward,
        fusing the satellite-derived time series with Census, GDP and policy signals. Monte-Carlo dropout
        yields the 95% confidence band.
      </SectionHead>

      <ReadThis>
        The <span style={{ color: "var(--cyan)" }}>solid line</span> is the model's best guess for each city's
        built-up area; the <span style={{ color: "var(--cyan)" }}>shaded band</span> is the 95% confidence range
        from <InfoTip term="MC Dropout" />. A wider band = more uncertainty, which naturally grows for years
        further out. <b>This band is the novelty</b> — no prior Indian forecasting study reports uncertainty.
        Switch cities with the buttons; hover the chart for exact values.
      </ReadThis>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 18 }} className="reveal">
        {cities.map((c) => (
          <button key={c} className={"btn" + (c === active ? " primary" : "")} onClick={() => setCity(c)}>
            {c.replace("_", " ")}
          </button>
        ))}
      </div>

      <div className="grid cols-3" style={{ marginBottom: 18 }}>
        <div className="panel stat reveal"><div className="k">2035 projection</div><div className="v">{last ? Math.round(last.mean).toLocaleString() : "—"} <small>km²</small></div></div>
        <div className="panel stat reveal" style={{ animationDelay: "0.05s" }}><div className="k">Growth {first?.year}→2035</div><div className="v" style={{ color: growth >= 0 ? "var(--green)" : "var(--red)" }}>{growth >= 0 ? "+" : ""}{growth.toFixed(1)}<small>%</small></div></div>
        <div className="panel stat reveal" style={{ animationDelay: "0.1s" }}><div className="k">2035 uncertainty</div><div className="v">±{Math.round(uncertainty / 2).toLocaleString()} <small>km²</small></div></div>
      </div>

      <div className="panel pad reveal" style={{ animationDelay: "0.15s" }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>{active?.replace("_", " ")} · forecast trajectory</div>
        <ResponsiveContainer width="100%" height={420}>
          <ComposedChart data={rows} margin={{ top: 10, right: 24, left: 8, bottom: 6 }}>
            <defs>
              <linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3fd4c4" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#3fd4c4" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(120,165,190,0.1)" vertical={false} />
            <XAxis dataKey="year" stroke="#5a6e7c" tick={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
            <YAxis stroke="#5a6e7c" tick={{ fontFamily: "var(--font-mono)", fontSize: 12 }} domain={["auto", "auto"]}
              label={{ value: "built-up area (km²)", angle: -90, position: "insideLeft", fill: "#5a6e7c", style: { fontFamily: "var(--font-mono)", fontSize: 11 } }} />
            <Tooltip content={<ChartTip />} />
            <Area type="monotone" dataKey="band" stroke="none" fill="url(#band)" isAnimationActive />
            <Line type="monotone" dataKey="mean" stroke="#3fd4c4" strokeWidth={2.6} dot={{ r: 3, fill: "#3fd4c4", strokeWidth: 0 }} activeDot={{ r: 6 }} />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="legend" style={{ marginTop: 12 }}>
          <span><i style={{ background: "#3fd4c4" }} />mean forecast</span>
          <span><i style={{ background: "rgba(63,212,196,0.25)" }} />95% confidence band (MC dropout)</span>
        </div>
      </div>
    </div>
  );
}
