import { useMemo } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type TimeSeries } from "../api";
import { Loading, ReadThis, SectionHead, useApi } from "../ui";

const CITY_COLORS: Record<string, string> = {
  Mumbai: "#3fd4c4",
  Delhi_NCR: "#f6a93b",
  Bangalore: "#9b8cf0",
};

function ChartTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rc-tip">
      <div className="t">YEAR {label}</div>
      {payload.map((p: any) => (
        <div className="row" key={p.dataKey}>
          <span style={{ color: p.color }}>{p.dataKey.replace("_", " ")}</span>
          <span>{Math.round(p.value).toLocaleString()} km²</span>
        </div>
      ))}
    </div>
  );
}

export default function Growth() {
  const { data } = useApi<TimeSeries>(api.timeseries, []);

  const { rows, cities } = useMemo(() => {
    if (!data) return { rows: [], cities: [] as string[] };
    const cities = Object.keys(data);
    const years = new Set<number>();
    cities.forEach((c) => data[c].forEach((p) => years.add(p.year)));
    const rows = [...years]
      .sort((a, b) => a - b)
      .map((year) => {
        const row: any = { year };
        cities.forEach((c) => {
          const pt = data[c].find((p) => p.year === year);
          if (pt) row[c] = pt.urban_area_km2;
        });
        return row;
      });
    return { rows, cities };
  }, [data]);

  if (!data) return <Loading label="loading urban time series" />;

  const summary = cities.map((c) => {
    const s = data[c];
    const first = s[0], last = s[s.length - 1];
    const delta = ((last.urban_area_km2 - first.urban_area_km2) / first.urban_area_km2) * 100;
    return { city: c, first, last, delta };
  });

  return (
    <div>
      <SectionHead eyebrow="Stage 03 · Longitudinal Aggregation" title="Urban growth, 1990 → 2023">
        Every classified patch is summed per city per year to reconstruct the built-up footprint over three
        decades. These curves are the input to the forecasting model. Sources: Landsat (pre-2017) and
        Sentinel-2 (2017+).
      </SectionHead>

      <ReadThis>
        Each line is one city's <b>built-up area in km²</b> over time — a rising line means the city is sprawling.
        The cards above show the latest value and the total change since 1990. Hover any point for the exact number.
        (Delhi and Bangalore look flatter because their tight study boxes were already nearly fully urban; Mumbai,
        hemmed in by sea and a national park, varies more.)
      </ReadThis>

      <div className="grid cols-3" style={{ marginBottom: 18 }}>
        {summary.map((s, i) => (
          <div key={s.city} className="panel stat reveal" style={{ animationDelay: `${0.05 * i}s` }}>
            <div className="k">{s.city.replace("_", " ")}</div>
            <div className="v">
              {Math.round(s.last.urban_area_km2).toLocaleString()} <small>km²</small>
            </div>
            <div className="u" style={{ color: s.delta >= 0 ? "var(--green)" : "var(--red)" }}>
              {s.delta >= 0 ? "▲" : "▼"} {Math.abs(s.delta).toFixed(1)}% since {s.first.year}
            </div>
          </div>
        ))}
      </div>

      <div className="panel pad reveal" style={{ animationDelay: "0.15s" }}>
        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={rows} margin={{ top: 10, right: 24, left: 8, bottom: 6 }}>
            <CartesianGrid stroke="rgba(120,165,190,0.1)" vertical={false} />
            <XAxis dataKey="year" stroke="#5a6e7c" tick={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
            <YAxis
              stroke="#5a6e7c"
              tick={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
              label={{ value: "built-up area (km²)", angle: -90, position: "insideLeft", fill: "#5a6e7c", style: { fontFamily: "var(--font-mono)", fontSize: 11 } }}
            />
            <Tooltip content={<ChartTip />} />
            <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
            {cities.map((c) => (
              <Line
                key={c}
                type="monotone"
                dataKey={c}
                name={c.replace("_", " ")}
                stroke={CITY_COLORS[c] || "#3fd4c4"}
                strokeWidth={2.4}
                dot={{ r: 3, strokeWidth: 0, fill: CITY_COLORS[c] }}
                activeDot={{ r: 6 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
