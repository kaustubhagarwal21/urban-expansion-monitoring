// API client for the Urban Expansion Monitoring backend.
// CORS is open on the backend, so we hit it directly (works in dev, preview, build).
export const API = (import.meta.env.VITE_API_BASE as string) || "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json() as Promise<T>;
}

export const CLASS_COLORS: Record<string, string> = {
  Urban: "#f6a93b",
  "Non-Urban": "#5fcf8c",
  Transition: "#57b6ef",
};

export const SEVERITY_COLORS: Record<string, string> = {
  NONE: "#5a6e7c",
  LOW: "#e8c34a",
  MEDIUM: "#f6a93b",
  HIGH: "#ef7d3b",
  CRITICAL: "#ef5d5d",
};

// ---- types ----
export interface PipelineStage { id: string; name: string; desc: string; tech: string[]; }
export interface Overview {
  title: string; subtitle: string; venue: string;
  cities: string[]; classes: string[];
  headline: {
    best_model: string; best_oa: number; best_oa_std: number;
    n_models: number; n_cities: number; n_alerts: number; seeds: number[];
  };
  pipeline: PipelineStage[];
}

export interface ModelRow {
  key: string; label: string; family: string; role: string;
  oa_mean: number | null; oa_std: number | null;
  f1_mean: number | null; f1_std: number | null;
  miou_mean: number | null; miou_std: number | null;
  raw_oa: number[] | null;
  params_m: number | null; latency_ms: number | null; throughput_ps: number | null;
}

export interface MetricRow {
  key: string; label: string;
  oa_mean: number | null; oa_std: number | null;
  f1_mean: number | null; f1_std: number | null;
  miou_mean: number | null; miou_std: number | null;
  raw_oa: number[] | null;
}

export interface TsPoint { year: number; urban_area_km2: number; urban_fraction: number; total_area_km2: number; source: string; }
export type TimeSeries = Record<string, TsPoint[]>;

export interface ForecastPoint { year: number; mean: number; ci_lower: number; ci_upper: number; }
export type Forecasts = Record<string, ForecastPoint[]>;

export interface Alert {
  id: string; city: string; alert_type: string; severity: string;
  confidence: string; coordinates: { lat: number; lon: number };
  location: string; in_protected_zone: boolean; zone_name: string | null;
  requires_escalation: boolean; status: string; timestamp: string;
}
export interface AlertReport {
  total_alerts: number; escalated_count: number;
  by_severity: Record<string, number>;
  by_city: Record<string, { total: number; critical: number; high: number; protected_zone: number }>;
  by_alert_type: Record<string, number>;
  protected_zone_violations: number;
}
export interface AlertsPayload { alerts: Alert[]; report: AlertReport; dashboard: any; }

export interface Sample { id: string; city: string; true_class: number; true_label: string; preview: string; }
export interface ClassifyResult {
  backbone: string; predicted_class: number; predicted_label: string;
  confidence: number; probabilities: Record<string, number>;
  true_label?: string; city?: string;
}

export interface ExplainBlock { q: string; p: string; }
export interface ExplainSection { id: string; number: string; title: string; summary: string; blocks: ExplainBlock[]; }
export interface GlossaryItem { t: string; p: string; }
export interface ExplainPayload { title: string; venue: string; authors: string; sections: ExplainSection[]; glossary: GlossaryItem[]; }

export interface ReviewerQA {
  id: number; severity: "CRITICAL" | "MODERATE" | "MINOR"; asker: string;
  question: string; plain: string; points: string[]; one_liner: string;
}
export interface VenueTip { venue: string; want: string; tip: string; }
export interface ReviewerPayload { qa: ReviewerQA[]; venue_tips: VenueTip[]; }

export interface NoveltyPillar { tag: string; title: string; what: string; why_novel: string; evidence: string; }
export interface PositioningItem { area: string; text: string; }
export interface NoveltyPayload { pitch: string; pillars: NoveltyPillar[]; positioning: PositioningItem[]; }

export interface Limitation { tag: string; title: string; what: string; mitigation: string; future: string; }
export interface LimitationsPayload { limitations: Limitation[]; }

export interface Beat { n: number; route: string; page: string; min: number; click: string; say: string; }
export interface PresenterPayload { total_minutes: number; beats: Beat[]; tips: string[]; }

export interface ClassifyUploadResult extends ClassifyResult {}

export interface FigureItem { file: string; group: string; title: string; caption: string; read: string; takeaway: string; in_paper?: boolean; }
export interface FiguresPayload { figures: FigureItem[]; }

export interface TourSlide { kicker: string; title: string; body?: string; points?: string[]; goto?: { route: string; label: string }; }
export interface TourPayload { slides: TourSlide[]; }

export const SEVERITY_QA_COLORS: Record<string, string> = {
  CRITICAL: "#ef5d5d",
  MODERATE: "#f6a93b",
  MINOR: "#57b6ef",
};

// ---- endpoints ----
export const api = {
  health: () => get<{ status: string; models_available: string[] }>("/api/health"),
  overview: () => get<Overview>("/api/overview"),
  models: () => get<ModelRow[]>("/api/models"),
  loco: () => get<MetricRow[]>("/api/loco"),
  ablation: () => get<MetricRow[]>("/api/ablation"),
  pillars: () => get<any>("/api/pillars"),
  paper: () => get<any>("/api/paper"),
  timeseries: () => get<TimeSeries>("/api/timeseries"),
  forecasts: () => get<Forecasts>("/api/forecasts"),
  alerts: () => get<AlertsPayload>("/api/alerts"),
  explain: () => get<ExplainPayload>("/api/explain"),
  reviewer: () => get<ReviewerPayload>("/api/reviewer"),
  novelty: () => get<NoveltyPayload>("/api/novelty"),
  limitations: () => get<LimitationsPayload>("/api/limitations"),
  presenter: () => get<PresenterPayload>("/api/presenter"),
  figures: () => get<FiguresPayload>("/api/figures"),
  tour: () => get<TourPayload>("/api/tour"),
  samples: () => get<Sample[]>("/api/samples"),
  classify: (id: string, backbone: string) =>
    get<ClassifyResult>(`/api/classify/${id}?backbone=${backbone}`),
  sampleImg: (id: string) => `${API}/samples/${id}.png`,
  gradcamImg: (id: string, backbone: string) => `${API}/api/gradcam/${id}?backbone=${backbone}`,
  figure: (name: string) => `${API}/figures/${name}`,
  classifyUpload: async (file: File, backbone: string): Promise<ClassifyResult> => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/classify-upload?backbone=${backbone}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `upload -> ${r.status}`);
    return r.json();
  },
};
