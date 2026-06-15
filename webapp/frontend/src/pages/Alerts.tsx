import { useMemo, useState } from "react";
import { api, SEVERITY_COLORS, type Alert, type AlertsPayload } from "../api";
import { Loading, ReadThis, SectionHead, useApi } from "../ui";

const W = 1000, H = 720, PAD = 0.1;
const SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"];
const SEV_R: Record<string, number> = { CRITICAL: 9, HIGH: 7, MEDIUM: 5.5, LOW: 4.5, NONE: 4 };

export default function Alerts() {
  const { data } = useApi<AlertsPayload>(api.alerts, []);
  const [sevFilter, setSevFilter] = useState<string | null>(null);
  const [hover, setHover] = useState<Alert | null>(null);
  const [selected, setSelected] = useState<Alert | null>(null);

  const project = useMemo(() => {
    if (!data?.alerts.length) return null;
    const lats = data.alerts.map((a) => a.coordinates.lat);
    const lons = data.alerts.map((a) => a.coordinates.lon);
    let minLat = Math.min(...lats), maxLat = Math.max(...lats);
    let minLon = Math.min(...lons), maxLon = Math.max(...lons);
    const dLat = (maxLat - minLat) * PAD || 1, dLon = (maxLon - minLon) * PAD || 1;
    minLat -= dLat; maxLat += dLat; minLon -= dLon; maxLon += dLon;
    return {
      x: (lon: number) => ((lon - minLon) / (maxLon - minLon)) * W,
      y: (lat: number) => (1 - (lat - minLat) / (maxLat - minLat)) * H,
      minLat, maxLat, minLon, maxLon,
    };
  }, [data]);

  const cityCentroids = useMemo(() => {
    if (!data) return [];
    const groups: Record<string, { lat: number; lon: number; n: number }> = {};
    data.alerts.forEach((a) => {
      const g = (groups[a.city] ||= { lat: 0, lon: 0, n: 0 });
      g.lat += a.coordinates.lat; g.lon += a.coordinates.lon; g.n++;
    });
    return Object.entries(groups).map(([city, g]) => ({ city, lat: g.lat / g.n, lon: g.lon / g.n }));
  }, [data]);

  if (!data) return <Loading label="loading alert engine" />;

  const r = data.report;
  const shown = data.alerts.filter((a) => !sevFilter || a.severity === sevFilter);
  const stats = [
    { k: "Total alerts", v: r.total_alerts },
    { k: "Critical", v: r.by_severity?.CRITICAL || 0, c: SEVERITY_COLORS.CRITICAL },
    { k: "High", v: r.by_severity?.HIGH || 0, c: SEVERITY_COLORS.HIGH },
    { k: "Protected-zone hits", v: r.protected_zone_violations, c: "var(--violet)" },
  ];

  return (
    <div>
      <SectionHead eyebrow="Stage 05 · Pillar V — Encroachment Alerts" title="Real-time encroachment monitor">
        Predicted expansion is cross-checked against India's regulatory zones — CRZ, forest reserves,
        wetlands, lake buffers. Each marker is a flagged event, sized and coloured by severity and routed to
        the relevant authority.
      </SectionHead>

      <ReadThis>
        Each dot on the map is a flagged encroachment event, positioned by its real coordinates. <b>Bigger, redder
        dots = more severe.</b> Dots ringed in violet sit inside a protected zone. Use the severity buttons to filter,
        hover a dot for details, or click one to highlight it in the feed on the right. The stat cards summarise the
        whole simulation. (For the same data on a 3D Earth, see the <b>Globe</b> page.)
      </ReadThis>

      <div className="grid cols-4" style={{ marginBottom: 18 }}>
        {stats.map((s, i) => (
          <div key={s.k} className="panel stat reveal" style={{ animationDelay: `${0.05 * i}s` }}>
            <div className="k">{s.k}</div>
            <div className="v" style={{ color: s.c as string }}>{s.v}</div>
          </div>
        ))}
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1.5fr 1fr", alignItems: "start" }}>
        {/* MAP */}
        <div className="panel pad reveal">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div className="eyebrow">Geospatial scope · {shown.length} events</div>
            <div className="seg">
              <button className={!sevFilter ? "on" : ""} onClick={() => setSevFilter(null)}>ALL</button>
              {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((s) => (
                <button key={s} className={sevFilter === s ? "on" : ""} onClick={() => setSevFilter(s)}>{s}</button>
              ))}
            </div>
          </div>

          <div style={{ position: "relative" }}>
            <svg viewBox={`0 0 ${W} ${H}`} className="geomap">
              {/* graticule */}
              {project && Array.from({ length: 7 }).map((_, i) => {
                const gx = (i / 6) * W, gy = (i / 6) * H;
                return (
                  <g key={i}>
                    <line x1={gx} y1={0} x2={gx} y2={H} className="grat" />
                    <line x1={0} y1={gy} x2={W} y2={gy} className="grat" />
                  </g>
                );
              })}
              {/* city labels */}
              {project && cityCentroids.map((c) => (
                <text key={c.city} x={project.x(c.lon)} y={project.y(c.lat) - 16} className="city-label" textAnchor="middle">
                  {c.city.replace("_", " ").toUpperCase()}
                </text>
              ))}
              {/* alerts */}
              {project && shown.map((a) => {
                const cx = project.x(a.coordinates.lon), cy = project.y(a.coordinates.lat);
                const color = SEVERITY_COLORS[a.severity] || "#5a6e7c";
                const isSel = selected?.id === a.id;
                return (
                  <g key={a.id} onMouseEnter={() => setHover(a)} onMouseLeave={() => setHover(null)} onClick={() => setSelected(a)} style={{ cursor: "pointer" }}>
                    {(isSel || a.in_protected_zone) && <circle cx={cx} cy={cy} r={SEV_R[a.severity] + 7} fill="none" stroke={a.in_protected_zone ? "#9b8cf0" : color} strokeWidth={1} opacity={0.6} className={isSel ? "pulsing" : ""} />}
                    <circle cx={cx} cy={cy} r={SEV_R[a.severity] || 5} fill={color} fillOpacity={0.85} stroke="#080b10" strokeWidth={1} style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
                  </g>
                );
              })}
            </svg>
            {hover && project && (
              <div className="map-tip" style={{ left: `${(project.x(hover.coordinates.lon) / W) * 100}%`, top: `${(project.y(hover.coordinates.lat) / H) * 100}%` }}>
                <b style={{ color: SEVERITY_COLORS[hover.severity] }}>{hover.severity}</b> · {hover.alert_type.replace(/_/g, " ")}<br />
                <span className="faint">{hover.city.replace("_", " ")} · {hover.coordinates.lat.toFixed(3)}, {hover.coordinates.lon.toFixed(3)}</span>
              </div>
            )}
          </div>
          <div className="legend" style={{ marginTop: 14 }}>
            {SEV_ORDER.map((s) => (<span key={s}><i style={{ background: SEVERITY_COLORS[s], borderRadius: "50%" }} />{s}</span>))}
            <span><i style={{ border: "1.5px solid #9b8cf0", background: "transparent", borderRadius: "50%" }} />protected zone</span>
          </div>
        </div>

        {/* FEED */}
        <div className="panel pad reveal" style={{ animationDelay: "0.05s", maxHeight: 760, overflow: "auto" }}>
          <div className="eyebrow" style={{ marginBottom: 12 }}>Alert feed</div>
          {shown.map((a) => (
            <button key={a.id} className={"feed-item" + (selected?.id === a.id ? " on" : "")} onClick={() => setSelected(a)}>
              <span className="sev" style={{ background: SEVERITY_COLORS[a.severity] }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="ft">{a.alert_type.replace(/_/g, " ")}</div>
                <div className="fm mono">
                  {a.city.replace("_", " ")} · {a.id}
                  {a.in_protected_zone && <span style={{ color: "var(--violet)" }}> · ⚠ {a.zone_name || "protected"}</span>}
                </div>
              </div>
              <span className="fconf mono">{(parseFloat(a.confidence) * 100).toFixed(0)}%</span>
            </button>
          ))}
        </div>
      </div>

      <style>{`
        .geomap { width: 100%; height: auto; background: radial-gradient(circle at 50% 40%, rgba(63,212,196,0.05), transparent 70%), var(--ink); border: 1px solid var(--line); border-radius: 12px; display: block; }
        .grat { stroke: rgba(120,165,190,0.1); stroke-width: 1; }
        .city-label { fill: var(--text-faint); font-family: var(--font-mono); font-size: 13px; letter-spacing: 0.12em; }
        .map-tip { position: absolute; transform: translate(-50%, -130%); background: var(--panel-2); border: 1px solid var(--line-strong); border-radius: 9px; padding: 8px 11px; font-family: var(--font-mono); font-size: 11px; pointer-events: none; white-space: nowrap; box-shadow: var(--shadow); z-index: 4; }
        .feed-item { width: 100%; display: flex; align-items: center; gap: 11px; padding: 11px 10px; background: transparent; border: 0; border-bottom: 1px solid var(--line); text-align: left; transition: background 0.15s; }
        .feed-item:hover { background: rgba(120,165,190,0.05); }
        .feed-item.on { background: rgba(63,212,196,0.08); }
        .feed-item .sev { width: 9px; height: 9px; border-radius: 50%; flex: none; }
        .feed-item .ft { font-size: 13px; text-transform: capitalize; }
        .feed-item .fm { font-size: 11px; color: var(--text-faint); margin-top: 2px; }
        .feed-item .fconf { font-size: 12px; color: var(--text-dim); }
      `}</style>
    </div>
  );
}
