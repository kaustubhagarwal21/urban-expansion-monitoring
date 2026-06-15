import { useEffect, useState } from "react";

/** tiny data-fetch hook */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    let attempts = 0;
    setData(null);
    setError(null);
    // Retry while the backend is still warming up (torch import ~20s on cold start).
    const attempt = () => {
      fn()
        .then((d) => { if (live) setData(d); })
        .catch((e) => {
          if (!live) return;
          attempts += 1;
          if (attempts < 12) setTimeout(attempt, 2000); // ~24s of retries
          else setError(String(e));
        });
    };
    attempt();
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--text-dim)", padding: 28 }}>
      <span className="spinner" /> <span className="mono" style={{ fontSize: 13 }}>{label}</span>
    </div>
  );
}

export function SectionHead({ eyebrow, title, children }: { eyebrow: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="section-head reveal">
      <div className="eyebrow">{eyebrow}</div>
      <h2 style={{ marginTop: 8 }}>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  );
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "—" : (x * 100).toFixed(d) + "%");
export const f3 = (x: number | null | undefined) => (x == null ? "—" : x.toFixed(3));

/** Shared plain-English definitions for jargon used across pages. */
export const TERMS: Record<string, string> = {
  OA: "Overall Accuracy — the percentage of image patches the model labels correctly.",
  F1: "F1 score — a balanced blend of precision and recall; fairer than accuracy when classes are imbalanced (0–1, higher is better).",
  mIoU: "Mean Intersection-over-Union — a stricter overlap score between predicted and true classes (0–1, higher is better).",
  LOCO: "Leave-One-City-Out — train on two cities, test on the third (unseen) one. The real test of cross-city generalisation.",
  seeds: "We repeat each experiment with 3 random starting points (seeds 42/123/7) and report mean ± standard deviation, so results aren't a fluke.",
  "mean ± std": "Average across 3 random seeds, plus how much it varied. Smaller std = more stable model.",
  Transition: "Our novel third class — a 100 m band around urban edges capturing the mixed, actively-growing fringe where encroachment happens.",
  "MC Dropout": "Monte-Carlo Dropout — run the forecaster 50 times with random neurons switched off; the spread of answers becomes the 95% confidence band.",
  params: "Number of learnable weights in the model (millions). Smaller = lighter to deploy.",
  latency: "Time to classify one patch (milliseconds). Lower = faster.",
  throughput: "Patches processed per second. Higher = faster.",
  ablation: "Removing one component at a time to measure whether it actually helps.",
  "Grad-CAM": "A heatmap that highlights which pixels most influenced the model's decision — warmer = more important.",
  confidence: "How sure the model is about its top prediction (the highest class probability, 0–100%).",
  "RGB composite": "A natural-colour image built from the satellite's red, green and blue bands so the patch looks like a normal photo.",
  precision: "Of the patches the model called class X, how many really were X.",
  recall: "Of the patches that truly are class X, how many the model found.",
  "t-SNE": "A way to squash high-dimensional model features into 2D so you can see which groups cluster together.",
  "confusion matrix": "A grid showing, for each true class, what the model predicted — the diagonal is correct, off-diagonal are mistakes.",
  Siamese: "Two identical networks comparing a before/after image pair to detect what changed.",
  "domain gap": "The accuracy a model loses when it meets data from a new place or time it wasn't trained on.",
  "cross-city": "Testing the model on a different city than the one(s) it was trained on.",
  backbone: "The main feature-extraction body of a neural network (e.g. ResNet50, Swin-Tiny).",
  FPN: "Feature Pyramid Network — lets the model see fine details and big patterns at the same time.",
  "progressive fine-tuning": "Adapting a pretrained model in stages — first the last layer, then more, then all of it — so its prior knowledge isn't erased.",
  "Sentinel-2": "A free European satellite, 10 m per pixel, optical (visible + infrared) — our main image source.",
  "SAR fusion": "Combining cloud-piercing radar imagery with optical imagery to (try to) improve classification.",
  ImageNet: "A huge dataset of everyday photos models are pre-trained on before being adapted to satellite images.",
  "self-attention": "A mechanism where every part of the image can look at every other part — good at general, transferable patterns.",
  SimCLR: "A self-supervised method that learns from unlabelled images by matching two augmented copies of the same image.",
};

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Auto-wrap the first occurrence of any known glossary term in a string with a hover definition. */
export function Glossarize({ text }: { text: string }) {
  const terms = Object.keys(TERMS).sort((a, b) => b.length - a.length);
  const re = new RegExp(`(?<![A-Za-z0-9])(${terms.map(escapeRe).join("|")})(?![A-Za-z0-9])`, "g");
  const used = new Set<string>();
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const matched = m[1];
    const canonical = Object.keys(TERMS).find((k) => k.toLowerCase() === matched.toLowerCase());
    if (!canonical || used.has(canonical)) continue;
    used.add(canonical);
    out.push(text.slice(last, m.index));
    out.push(<InfoTip key={`${canonical}-${m.index}`} term={canonical}>{matched}</InfoTip>);
    last = m.index + matched.length;
  }
  out.push(text.slice(last));
  return <>{out}</>;
}

/** Inline term with a hover/tap tooltip definition. */
export function InfoTip({ term, def, children }: { term?: string; def?: string; children?: React.ReactNode }) {
  const text = def ?? (term ? TERMS[term] : "") ?? "";
  return (
    <span className="itip" tabIndex={0}>
      {children ?? term}
      <sup className="itip-i">i</sup>
      <span className="itip-pop">{text}</span>
    </span>
  );
}

/** Collapsible "how to read this" helper note for data pages. */
export function ReadThis({ children, title = "How to read this" }: { children: React.ReactNode; title?: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className={"readthis reveal" + (open ? " open" : "")}>
      <button className="rt-head" onClick={() => setOpen((o) => !o)}>
        <span className="rt-ic">?</span>
        <span>{title}</span>
        <span className="rt-tog">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="rt-body">{children}</div>}
    </div>
  );
}

/** count-up number animation on mount */
export function CountUp({ value, duration = 1000, decimals = 0, suffix = "" }: { value: number; duration?: number; decimals?: number; suffix?: string }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setN(value * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
      else setN(value);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return <>{n.toFixed(decimals)}{suffix}</>;
}
