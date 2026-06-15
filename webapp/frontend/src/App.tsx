import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./api";
import Overview from "./pages/Overview";
import Classify from "./pages/Classify";
import Growth from "./pages/Growth";
import Forecast from "./pages/Forecast";
import Alerts from "./pages/Alerts";
import Benchmarks from "./pages/Benchmarks";
import Explain from "./pages/Explain";
import Reviewer from "./pages/Reviewer";
import Novelty from "./pages/Novelty";
import Limitations from "./pages/Limitations";
import Presenter from "./pages/Presenter";
import GlobeView from "./pages/GlobeView";
import Figures from "./pages/Figures";
import Welcome from "./components/Welcome";

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, viewBox: "0 0 24 24" };
const ICONS: Record<string, React.ReactNode> = {
  "/overview": <svg {...S}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>,
  "/classify": <svg {...S}><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><circle cx="12" cy="12" r="2.5" /></svg>,
  "/growth": <svg {...S}><path d="M3 20h18" /><path d="M5 16l4-5 3 3 5-7" /></svg>,
  "/forecast": <svg {...S}><path d="M3 17l5-5 4 3 7-8" /><path d="M16 7h5v5" /></svg>,
  "/alerts": <svg {...S}><path d="M12 3l9 16H3z" /><path d="M12 10v4" /><circle cx="12" cy="17" r="0.6" fill="currentColor" /></svg>,
  "/globe": <svg {...S}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18" /></svg>,
  "/benchmarks": <svg {...S}><path d="M4 20V10M10 20V4M16 20v-8M22 20H2" /></svg>,
  "/figures": <svg {...S}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.5" cy="9.5" r="1.5" /><path d="M21 16l-5-5L5 20" /></svg>,
  "/novelty": <svg {...S}><path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2z" /></svg>,
  "/limitations": <svg {...S}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" /><path d="M12 8v4M12 15.5v.5" /></svg>,
  "/explain": <svg {...S}><path d="M4 5a2 2 0 0 1 2-2h6v18H6a2 2 0 0 1-2-2z" /><path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" /><path d="M8 8h2M8 12h2M15 8h2M15 12h2" /></svg>,
  "/reviewer": <svg {...S}><path d="M21 12a8 8 0 0 1-8 8H7l-4 3v-7a8 8 0 1 1 18-4z" /><path d="M9.5 9.5a2.5 2.5 0 0 1 4 1.8c0 1.7-2.5 1.7-2.5 3.2M12 17v.5" /></svg>,
  "/presenter": <svg {...S}><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M12 17v3M8 20h8" /><path d="M10 8.5l4 2.5-4 2.5z" fill="currentColor" stroke="none" /></svg>,
};

const NAV = [
  { to: "/overview", idx: "01", label: "Overview" },
  { to: "/classify", idx: "02", label: "Live Classification" },
  { to: "/growth", idx: "03", label: "Urban Growth" },
  { to: "/forecast", idx: "04", label: "Sprawl Forecast" },
  { to: "/alerts", idx: "05", label: "Encroachment Alerts" },
  { to: "/globe", idx: "06", label: "Globe" },
  { to: "/benchmarks", idx: "07", label: "Benchmarks" },
  { to: "/figures", idx: "08", label: "Figures" },
  { to: "/novelty", idx: "09", label: "Novelty" },
  { to: "/limitations", idx: "10", label: "Limitations" },
  { to: "/explain", idx: "11", label: "Paper, Explained" },
  { to: "/reviewer", idx: "12", label: "Reviewer Q&A" },
  { to: "/presenter", idx: "13", label: "Presenter Mode" },
];

export default function App() {
  const [online, setOnline] = useState<boolean | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [tourOpen, setTourOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((h) => {
          if (!alive) return;
          setOnline(h.status === "ok");
          setModels(h.models_available);
        })
        .catch(() => alive && setOnline(false));
    check();
    // Keep checking so the dot turns green on its own once the backend finishes
    // loading (and goes red again if it stops).
    const id = setInterval(check, 4000);
    if (!localStorage.getItem("lm_seen_tour")) setTourOpen(true);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const closeTour = () => {
    setTourOpen(false);
    localStorage.setItem("lm_seen_tour", "1");
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="glyph" />
          <div>
            <div className="name">The Living Map</div>
            <div className="sub">URBAN&nbsp;EXPANSION&nbsp;OBS.</div>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} className={({ isActive }) => "nav-item" + (isActive ? " active" : "")}>
              <span className="nav-ic">{ICONS[n.to]}</span>
              <span className="nav-label">{n.label}</span>
              <span className="idx">{n.idx}</span>
            </NavLink>
          ))}
        </nav>

        <div className="foot">
          <div>
            <span className={"status-dot " + (online ? "ok" : online === false ? "down" : "")} />
            {online === null ? "connecting…" : online ? "backend online" : "backend offline"}
          </div>
          <div style={{ marginTop: 6, opacity: 0.7 }}>{models.length} models loaded</div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="eyebrow">Transfer Learning · Indian Metropolitan Regions</div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <button className="btn start-here" onClick={() => setTourOpen(true)} title="New here? Start the guided tour">
              <span className="sh-q">?</span> Start Here
            </button>
            <div className="venue">
              ACCEPTED · <b>IEEE CHANDICON 2026</b>
            </div>
          </div>
        </header>

        <div className="page">
          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<Overview />} />
            <Route path="/classify" element={<Classify />} />
            <Route path="/growth" element={<Growth />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/globe" element={<GlobeView />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/figures" element={<Figures />} />
            <Route path="/novelty" element={<Novelty />} />
            <Route path="/limitations" element={<Limitations />} />
            <Route path="/explain" element={<Explain />} />
            <Route path="/reviewer" element={<Reviewer />} />
            <Route path="/presenter" element={<Presenter />} />
          </Routes>
        </div>
      </div>

      <Welcome open={tourOpen} onClose={closeTour} />
    </div>
  );
}
