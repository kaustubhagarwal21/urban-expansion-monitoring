import { useEffect, useMemo, useRef, useState } from "react";
import Globe from "react-globe.gl";
import { api, SEVERITY_COLORS, type Alert, type AlertsPayload } from "../api";
import { Loading, ReadThis, SectionHead, useApi } from "../ui";

const GlobeAny = Globe as any;
const INDIA_POV = { lat: 21, lng: 80, altitude: 1.9 };

export default function GlobeView() {
  const { data } = useApi<AlertsPayload>(api.alerts, []);
  const [countries, setCountries] = useState<any>(null);
  const globeRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 620 });
  const [spin, setSpin] = useState(true);
  const [selected, setSelected] = useState<any>(null);

  useEffect(() => {
    fetch("/countries-110m.geojson").then((r) => r.json()).then(setCountries).catch(() => {});
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const apply = () => setSize({ w: el.clientWidth, h: Math.max(560, Math.min(720, el.clientWidth * 0.62)) });
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!countries || !data) return;
    const id = setTimeout(() => {
      if (!globeRef.current) return;
      globeRef.current.pointOfView(INDIA_POV, 0);
      const c = globeRef.current.controls();
      c.autoRotate = spin;
      c.autoRotateSpeed = 0.45;
      c.enableZoom = true;
      c.minDistance = 101.5; // allow zooming right down to the surface
      c.maxDistance = 600;
      c.zoomSpeed = 1.2;
    }, 60);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [countries, data]);

  useEffect(() => {
    if (globeRef.current) globeRef.current.controls().autoRotate = spin;
  }, [spin]);

  const alerts = data?.alerts ?? [];
  const points = useMemo(
    () =>
      alerts.map((a: Alert) => ({
        lat: a.coordinates.lat,
        lng: a.coordinates.lon,
        sev: a.severity,
        color: SEVERITY_COLORS[a.severity] || "#5a6e7c",
        city: a.city.replace("_", " "),
        type: a.alert_type.replace(/_/g, " "),
        conf: a.confidence,
        zone: a.in_protected_zone,
        zoneName: a.zone_name,
        id: a.id,
      })),
    [alerts]
  );

  const cities = useMemo(() => {
    if (!data) return [];
    const g: Record<string, { lat: number; lng: number; n: number }> = {};
    data.alerts.forEach((a) => {
      const c = (g[a.city] ||= { lat: 0, lng: 0, n: 0 });
      c.lat += a.coordinates.lat; c.lng += a.coordinates.lon; c.n++;
    });
    return Object.entries(g).map(([city, v]) => ({ city: city.replace("_", " "), lat: v.lat / v.n, lng: v.lng / v.n, n: v.n }));
  }, [data]);

  function flyTo(lat: number, lng: number, altitude: number) {
    setSpin(false);
    globeRef.current?.pointOfView({ lat, lng, altitude }, 900);
  }

  const sevCounts = data?.report?.by_severity || {};

  return (
    <div>
      <SectionHead eyebrow="Stage 05+ · Geospatial Scope" title="Encroachment alerts on the globe">
        The Pillar V alerts on an interactive 3D Earth across 7 metros. Drag to rotate, scroll to zoom, hover any
        point for details, or click it to fly in. Runs fully offline — country outlines are bundled, no map tiles.
      </SectionHead>

      <ReadThis>
        Each glowing dot is one flagged encroachment event at its real coordinates; <b>bigger, redder, taller dots
        are more severe</b>, and violet-ringed cities have protected-zone hits. Use the <b>city buttons</b> to zoom
        straight into a metro — the alerts spread out so you can inspect each one. Hover a dot for its type and
        confidence; click it to fly in and pin its details.
      </ReadThis>

      <div className="panel" ref={wrapRef} style={{ overflow: "hidden", position: "relative", minHeight: 560 }}>
        {countries && data ? (
          <GlobeAny
            ref={globeRef}
            width={size.w}
            height={size.h}
            backgroundColor="rgba(0,0,0,0)"
            rendererConfig={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
            showGlobe={true}
            showAtmosphere
            atmosphereColor="#3fd4c4"
            atmosphereAltitude={0.17}
            hexPolygonsData={countries.features}
            hexPolygonResolution={3}
            hexPolygonMargin={0.3}
            hexPolygonAltitude={0.006}
            hexPolygonColor={() => "rgba(63,212,196,0.45)"}
            pointsData={points}
            pointLat="lat"
            pointLng="lng"
            pointColor="color"
            pointAltitude={(d: any) => (d.sev === "CRITICAL" ? 0.16 : d.sev === "HIGH" ? 0.11 : d.sev === "MEDIUM" ? 0.07 : 0.045)}
            pointRadius={(d: any) => (d.sev === "CRITICAL" ? 0.22 : d.sev === "HIGH" ? 0.17 : d.sev === "MEDIUM" ? 0.14 : 0.12)}
            pointResolution={12}
            pointsMerge={false}
            pointsTransitionDuration={0}
            pointLabel={(d: any) =>
              `<div style="font-family:'IBM Plex Mono',monospace;background:#14202c;border:1px solid ${d.color};border-radius:9px;padding:9px 11px;color:#e9f0f4;font-size:11px;line-height:1.5;box-shadow:0 10px 30px rgba(0,0,0,.7)">` +
              `<b style="color:${d.color}">${d.sev}</b> &middot; ${d.type}<br/>` +
              `<span style="color:#92a6b4">${d.city} &middot; ${(parseFloat(d.conf) * 100).toFixed(0)}% conf</span>` +
              (d.zone ? `<br/><span style="color:#9b8cf0">&#9888; ${d.zoneName || "protected zone"}</span>` : "") +
              `</div>`
            }
            onPointClick={(d: any) => { setSelected(d); flyTo(d.lat, d.lng, 0.16); }}
            ringsData={cities}
            ringLat="lat"
            ringLng="lng"
            ringColor={() => (t: number) => `rgba(63,212,196,${0.7 * (1 - t)})`}
            ringMaxRadius={1.3}
            ringPropagationSpeed={0.9}
            ringRepeatPeriod={1900}
            labelsData={cities}
            labelLat="lat"
            labelLng="lng"
            labelText="city"
            labelSize={1.15}
            labelDotRadius={0.28}
            labelColor={() => "rgba(233,240,244,0.92)"}
            labelResolution={2}
            onLabelClick={(d: any) => flyTo(d.lat, d.lng, 0.28)}
          />
        ) : (
          <Loading label="rendering globe" />
        )}

        {/* legend */}
        <div className="globe-legend">
          <div className="gl-h mono">SEVERITY</div>
          {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((s) => (
            <div className="gl-row" key={s}>
              <i style={{ background: SEVERITY_COLORS[s] }} />
              <span>{s}</span>
              <b>{(sevCounts as any)[s] || 0}</b>
            </div>
          ))}
          <button className="btn" style={{ marginTop: 12, width: "100%" }} onClick={() => setSpin((s) => !s)}>
            {spin ? "pause spin" : "resume spin"}
          </button>
        </div>

        {/* city focus */}
        <div className="globe-focus">
          <div className="gl-h mono" style={{ marginBottom: 8 }}>FLY TO</div>
          <div className="focus-btns">
            {cities.map((c) => (
              <button key={c.city} className="btn" onClick={() => flyTo(c.lat, c.lng, 0.28)}>{c.city}</button>
            ))}
            <button className="btn" onClick={() => { setSelected(null); setSpin(true); flyTo(INDIA_POV.lat, INDIA_POV.lng, INDIA_POV.altitude); }}>↺ India</button>
          </div>
        </div>

        {/* selected detail */}
        {selected && (
          <div className="globe-sel" style={{ borderColor: selected.color }}>
            <button className="sel-x" onClick={() => setSelected(null)}>✕</button>
            <div className="sel-sev" style={{ color: selected.color }}>{selected.sev}</div>
            <div className="sel-type">{selected.type}</div>
            <div className="sel-meta mono">
              {selected.city} · {(parseFloat(selected.conf) * 100).toFixed(0)}% conf<br />
              {selected.lat.toFixed(3)}, {selected.lng.toFixed(3)}
              {selected.zone && <><br /><span style={{ color: "var(--violet)" }}>⚠ {selected.zoneName || "protected zone"}</span></>}
            </div>
          </div>
        )}
      </div>

      <p className="faint mono" style={{ fontSize: 11, marginTop: 12 }}>
        {cities.length} cities · {alerts.length} alerts · drag to rotate · scroll to zoom · click a dot to fly in
      </p>

      <style>{`
        .globe-legend { position: absolute; top: 18px; left: 18px; background: rgba(11,17,24,0.74); border: 1px solid var(--line-strong); border-radius: 11px; padding: 14px; backdrop-filter: blur(6px); min-width: 150px; }
        .gl-h { font-size: 10px; letter-spacing: 0.16em; color: var(--text-faint); margin-bottom: 10px; }
        .gl-row { display: flex; align-items: center; gap: 9px; font-family: var(--font-mono); font-size: 12px; color: var(--text-dim); padding: 3px 0; }
        .gl-row i { width: 10px; height: 10px; border-radius: 50%; }
        .gl-row b { margin-left: auto; color: var(--text); }
        .globe-focus { position: absolute; top: 18px; right: 18px; background: rgba(11,17,24,0.74); border: 1px solid var(--line-strong); border-radius: 11px; padding: 14px; backdrop-filter: blur(6px); max-width: 180px; }
        .focus-btns { display: flex; flex-wrap: wrap; gap: 6px; }
        .focus-btns .btn { padding: 6px 10px; font-size: 11px; }
        .globe-sel { position: absolute; bottom: 18px; left: 18px; background: rgba(11,17,24,0.85); border: 1px solid; border-radius: 11px; padding: 14px 16px; backdrop-filter: blur(6px); min-width: 200px; }
        .sel-x { position: absolute; top: 8px; right: 10px; background: transparent; border: 0; color: var(--text-faint); font-size: 13px; }
        .sel-x:hover { color: var(--text); }
        .sel-sev { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.12em; }
        .sel-type { font-family: var(--font-display); font-size: 18px; text-transform: capitalize; margin: 3px 0 8px; }
        .sel-meta { font-size: 11px; color: var(--text-dim); line-height: 1.55; }
      `}</style>
    </div>
  );
}
