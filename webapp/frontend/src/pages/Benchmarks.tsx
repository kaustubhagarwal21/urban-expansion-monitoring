import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type MetricRow, type ModelRow } from "../api";
import { InfoTip, Loading, ReadThis, SectionHead, f3, pct, useApi } from "../ui";

const FAMILY_COLOR: Record<string, string> = {
  CNN: "#3fd4c4",
  Transformer: "#9b8cf0",
  Classical: "#5a6e7c",
};

function MetricTable({ rows, highlightKey, label }: { rows: MetricRow[] | ModelRow[]; highlightKey?: string; label: string }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>{label}</th>
          <th><InfoTip term="OA" /></th>
          <th><InfoTip term="F1" /></th>
          <th><InfoTip term="mIoU" /></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((m) => (
          <tr key={m.key} className={m.key === highlightKey ? "hi" : ""}>
            <td>{m.label}</td>
            <td><span className="num">{pct(m.oa_mean)}</span> <span className="std">± {((m.oa_std || 0) * 100).toFixed(1)}</span></td>
            <td><span className="num">{f3(m.f1_mean)}</span> <span className="std">± {(m.f1_std || 0).toFixed(3)}</span></td>
            <td><span className="num">{f3(m.miou_mean)}</span> <span className="std">± {(m.miou_std || 0).toFixed(3)}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Benchmarks() {
  const { data: models } = useApi<ModelRow[]>(api.models, []);
  const { data: loco } = useApi<MetricRow[]>(api.loco, []);
  const { data: ablation } = useApi<MetricRow[]>(api.ablation, []);

  if (!models) return <Loading label="loading benchmark results" />;

  const chartData = models
    .filter((m) => m.oa_mean != null)
    .map((m) => ({ name: m.label, oa: +(m.oa_mean! * 100).toFixed(2), err: (m.oa_std || 0) * 100, family: m.family }));

  return (
    <div>
      <SectionHead eyebrow="Evidence · 3-seed statistical validation" title="Benchmark results">
        Every number is <InfoTip term="mean ± std" /> over 3 random <InfoTip term="seeds" /> on real Indian
        satellite patches. ResNet50 leads in-distribution; Swin-Tiny generalises best across cities (see LOCO).
      </SectionHead>

      <ReadThis>
        <b>The bar chart</b> shows overall accuracy (higher = better) with the little whiskers = variation across seeds —
        shorter whiskers mean a more reliable model. <b>Main benchmark</b> tests on the same cities the model trained on;
        <b> Cross-city LOCO</b> tests on a city it has never seen, which is why those numbers are ~18 points lower.
        Hover the <span style={{ color: "var(--cyan)" }}>ⓘ</span> on any column to see what the metric means.
      </ReadThis>

      <div className="panel pad reveal" style={{ marginBottom: 18 }}>
        <div className="eyebrow" style={{ marginBottom: 12 }}>Overall accuracy · mean ± std</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 8, right: 20, left: 0, bottom: 6 }}>
            <CartesianGrid stroke="rgba(120,165,190,0.1)" vertical={false} />
            <XAxis dataKey="name" stroke="#5a6e7c" tick={{ fontFamily: "var(--font-mono)", fontSize: 11 }} interval={0} angle={-12} textAnchor="end" height={60} />
            <YAxis stroke="#5a6e7c" domain={[80, 100]} tick={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
            <Tooltip cursor={{ fill: "rgba(120,165,190,0.06)" }} contentStyle={{ background: "var(--panel-2)", border: "1px solid var(--line-strong)", borderRadius: 10, fontFamily: "var(--font-mono)", fontSize: 12 }} formatter={(v: any) => [`${v}%`, "OA"]} />
            <Bar dataKey="oa" radius={[5, 5, 0, 0]}>
              {chartData.map((d, i) => (<Cell key={i} fill={FAMILY_COLOR[d.family] || "#3fd4c4"} />))}
              <ErrorBar dataKey="err" width={5} strokeWidth={1.5} stroke="#64788a" direction="y" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="legend" style={{ marginTop: 10 }}>
          {Object.entries(FAMILY_COLOR).map(([k, c]) => (<span key={k}><i style={{ background: c }} />{k}</span>))}
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 18 }}>
        <div className="panel pad reveal">
          <div className="eyebrow" style={{ marginBottom: 12 }}>Main benchmark · in-distribution</div>
          <MetricTable rows={models} highlightKey="resnet50" label="Model" />
        </div>
        <div className="panel pad reveal" style={{ animationDelay: "0.05s" }}>
          <div className="eyebrow" style={{ marginBottom: 12 }}>Cross-city LOCO · leave-one-city-out</div>
          {loco ? <MetricTable rows={loco} highlightKey="swin_tiny" label="Model" /> : <Loading />}
          <p className="muted" style={{ fontSize: 13, marginTop: 14, lineHeight: 1.6 }}>
            The ~18% drop from in-distribution to held-out city is the real domain gap — Swin-Tiny's attention
            transfers best.
          </p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel pad reveal">
          <div className="eyebrow" style={{ marginBottom: 12 }}>Ablation · EfficientNet-B0</div>
          {ablation ? <MetricTable rows={ablation} label="Configuration" /> : <Loading />}
        </div>
        <div className="panel pad reveal" style={{ animationDelay: "0.05s" }}>
          <div className="eyebrow" style={{ marginBottom: 12 }}>Efficiency · deployment profile</div>
          <table className="table">
            <thead><tr><th>Model</th><th>Params</th><th>Latency</th><th>Throughput</th></tr></thead>
            <tbody>
              {models.filter((m) => m.params_m != null).map((m) => (
                <tr key={m.key} className={m.key === "mobilenet_v3_small" ? "hi" : ""}>
                  <td>{m.label}</td>
                  <td className="num">{m.params_m}M</td>
                  <td className="num">{m.latency_ms} ms</td>
                  <td className="num">{m.throughput_ps}/s</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ fontSize: 13, marginTop: 14, lineHeight: 1.6 }}>
            MobileNetV3 is the edge/real-time model — smallest and fastest, for the Pillar V alert engine.
          </p>
        </div>
      </div>
    </div>
  );
}
