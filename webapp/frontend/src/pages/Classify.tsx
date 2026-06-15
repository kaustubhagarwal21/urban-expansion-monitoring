import { useEffect, useMemo, useState } from "react";
import { api, CLASS_COLORS, type ClassifyResult, type Sample } from "../api";
import { InfoTip, Loading, ReadThis, SectionHead, useApi } from "../ui";

const BACKBONES = [
  { key: "resnet50", label: "ResNet50", gradcam: true },
  { key: "efficientnet_b0", label: "EfficientNet-B0", gradcam: true },
  { key: "mobilenet_v3_small", label: "MobileNetV3", gradcam: true },
  { key: "swin_tiny", label: "Swin-Tiny", gradcam: false },
];

export default function Classify() {
  const { data: samples } = useApi(api.samples, []);
  const [selected, setSelected] = useState<string | null>(null);
  const [backbone, setBackbone] = useState("resnet50");
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<ClassifyResult | null>(null);
  const [upBusy, setUpBusy] = useState(false);
  const [upErr, setUpErr] = useState<string | null>(null);

  async function handleFile(file?: File | null) {
    if (!file) return;
    setUpBusy(true);
    setUpErr(null);
    setUpload(null);
    try {
      setUpload(await api.classifyUpload(file, backbone));
    } catch (e) {
      setUpErr(String((e as Error).message || e));
    } finally {
      setUpBusy(false);
    }
  }

  // default to first sample once loaded
  useEffect(() => {
    if (samples && samples.length && !selected) setSelected(samples[0].id);
  }, [samples, selected]);

  // classify whenever selection / backbone changes
  useEffect(() => {
    if (!selected) return;
    setBusy(true);
    setResult(null);
    api
      .classify(selected, backbone)
      .then(setResult)
      .finally(() => setBusy(false));
  }, [selected, backbone]);

  const meta = useMemo(() => samples?.find((s) => s.id === selected), [samples, selected]);
  const supportsGradcam = BACKBONES.find((b) => b.key === backbone)?.gradcam;

  if (!samples) return <Loading label="loading sample patches" />;

  return (
    <div>
      <SectionHead eyebrow="Stage 02 · Patch Classification" title="Live land-cover inference">
        Pick a real 256×256 Sentinel-2 patch and watch the trained network classify it in real time on CPU.
        Urban / Non-Urban / <InfoTip term="Transition" /> — the same model that powers the whole pipeline.{" "}
        <InfoTip term="Grad-CAM" /> reveals which pixels drove the decision.
      </SectionHead>

      <ReadThis>
        Pick a patch on the left (the coloured dot = its true class). The middle shows the{" "}
        <InfoTip term="RGB composite" /> (what the patch looks like) and the <InfoTip term="Grad-CAM" /> heatmap
        (what the model looked at). On the right, the bars are the model's probability for each class and the big
        label is its pick — a green ✓ means it matched the ground truth. Switch the <b>backbone</b> to compare
        models, or drop your own <b>.npy</b> patch at the bottom-left to classify it live.
      </ReadThis>

      <div className="grid" style={{ gridTemplateColumns: "320px 1fr", alignItems: "start" }}>
        {/* patch picker */}
        <div className="panel pad reveal">
          <div className="eyebrow" style={{ marginBottom: 12 }}>Sample patches</div>
          <div className="patch-grid">
            {samples.map((s: Sample) => (
              <button
                key={s.id}
                className={"patch" + (s.id === selected ? " on" : "")}
                onClick={() => setSelected(s.id)}
                title={`${s.city} · ${s.true_label}`}
              >
                <img src={api.sampleImg(s.id)} alt={s.true_label} />
                <span className="dot" style={{ background: CLASS_COLORS[s.true_label] }} />
              </button>
            ))}
          </div>
          <div className="legend" style={{ marginTop: 16 }}>
            {Object.entries(CLASS_COLORS).map(([k, c]) => (
              <span key={k}><i style={{ background: c }} />{k}</span>
            ))}
          </div>

          {/* Q&A upload */}
          <div
            className="upload"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files?.[0]); }}
          >
            <input id="npy" type="file" accept=".npy" hidden onChange={(e) => handleFile(e.target.files?.[0])} />
            <label htmlFor="npy" className="upload-zone">
              {upBusy ? "classifying…" : <>Drop a <b>.npy</b> patch to classify<br /><span className="faint">or click to browse · shape (6,256,256)</span></>}
            </label>
            {upErr && <div className="up-msg err">{upErr}</div>}
            {upload && (
              <div className="up-msg ok">
                → <b style={{ color: CLASS_COLORS[upload.predicted_label] }}>{upload.predicted_label}</b> · {(upload.confidence * 100).toFixed(1)}%
              </div>
            )}
          </div>
        </div>

        {/* inference panel */}
        <div className="grid" style={{ gap: 18 }}>
          <div className="panel pad reveal" style={{ animationDelay: "0.05s" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
              <div>
                <div className="eyebrow">Backbone</div>
                <div className="muted mono" style={{ fontSize: 12, marginTop: 4 }}>
                  {meta ? `${meta.city} · ground truth: ${meta.true_label}` : ""}
                </div>
              </div>
              <div className="seg">
                {BACKBONES.map((b) => (
                  <button key={b.key} className={backbone === b.key ? "on" : ""} onClick={() => setBackbone(b.key)}>
                    {b.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="infer">
              {/* RGB */}
              <figure>
                <img src={selected ? api.sampleImg(selected) : ""} alt="patch" />
                <figcaption>RGB composite</figcaption>
              </figure>
              {/* GradCAM */}
              <figure>
                {supportsGradcam && selected ? (
                  <img src={api.gradcamImg(selected, backbone)} alt="grad-cam" />
                ) : (
                  <div className="nogc">Grad-CAM<br />n/a for transformer</div>
                )}
                <figcaption>Grad-CAM attention</figcaption>
              </figure>

              {/* prediction */}
              <div className="pred">
                {busy && <Loading label="running inference" />}
                {!busy && result && (
                  <>
                    <div className="eyebrow">Prediction</div>
                    <div className="pred-label" style={{ color: CLASS_COLORS[result.predicted_label] }}>
                      {result.predicted_label}
                    </div>
                    <div className="mono" style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 16 }}>
                      {(result.confidence * 100).toFixed(1)}% confidence
                      {result.true_label &&
                        (result.predicted_label === result.true_label ? (
                          <span style={{ color: "var(--green)", marginLeft: 10 }}>✓ correct</span>
                        ) : (
                          <span style={{ color: "var(--red)", marginLeft: 10 }}>✗ vs {result.true_label}</span>
                        ))}
                    </div>
                    {Object.entries(result.probabilities).map(([cls, p]) => (
                      <div key={cls} style={{ marginBottom: 11 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }} className="mono">
                          <span>{cls}</span>
                          <span>{(p * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track" style={{ marginTop: 5 }}>
                          <div className="bar-fill" style={{ width: `${p * 100}%`, background: CLASS_COLORS[cls], transition: "width 0.5s ease" }} />
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .patch-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        .patch { position: relative; padding: 0; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; aspect-ratio: 1; background: var(--ink); transition: all 0.15s; }
        .patch img { width: 100%; height: 100%; object-fit: cover; display: block; image-rendering: auto; }
        .patch:hover { border-color: var(--line-strong); }
        .patch.on { border-color: var(--cyan); box-shadow: 0 0 0 1px var(--cyan), 0 0 18px -6px var(--cyan); }
        .patch .dot { position: absolute; top: 6px; right: 6px; width: 9px; height: 9px; border-radius: 50%; box-shadow: 0 0 0 2px rgba(0,0,0,0.4); }
        .infer { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 20px; margin-top: 22px; align-items: start; }
        @media (max-width: 1100px) { .infer { grid-template-columns: 1fr 1fr; } }
        .infer figure { margin: 0; }
        .infer figure img, .infer .nogc { width: 100%; aspect-ratio: 1; border-radius: 11px; border: 1px solid var(--line); display: block; object-fit: cover; }
        .infer .nogc { display: flex; align-items: center; justify-content: center; text-align: center; color: var(--text-faint); font-family: var(--font-mono); font-size: 12px; background: var(--ink); }
        .infer figcaption { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); margin-top: 8px; letter-spacing: 0.06em; }
        .pred-label { font-family: var(--font-display); font-size: 32px; margin: 8px 0 2px; }
        .upload { margin-top: 16px; }
        .upload-zone { display: block; text-align: center; padding: 18px 12px; border-radius: 10px; border: 1px dashed var(--line-strong); color: var(--text-dim); font-size: 13px; line-height: 1.6; cursor: pointer; transition: all 0.15s; }
        .upload-zone:hover { border-color: var(--cyan); color: var(--text); }
        .up-msg { margin-top: 10px; font-family: var(--font-mono); font-size: 12px; padding: 8px 10px; border-radius: 8px; }
        .up-msg.ok { background: rgba(63,212,196,0.08); border: 1px solid var(--line-strong); }
        .up-msg.err { color: var(--red); }
      `}</style>
    </div>
  );
}
