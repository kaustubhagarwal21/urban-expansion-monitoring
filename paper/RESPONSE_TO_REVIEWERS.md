# Response to Reviewers — Paper ID 2356

**Title:** Urban Expansion Monitoring Using Transfer Learning on Historical Satellite Imagery
**Venue:** IEEE CHANDICON 2026

We thank both reviewers for their constructive feedback. We have revised the manuscript
accordingly; the main changes are summarised here, followed by point-by-point responses.
All section/figure/table references below refer to the revised manuscript.

**Summary of revisions**
- Added an explicit **step-by-step methodology flowchart** (new Fig. 2, vector graphics).
- Rewrote the **Abstract** and **Conclusion** for clarity and structure.
- Added a bold **"Key results"** lead-in and bolded the best result in every table so outcomes are clearly highlighted.
- Regenerated **all figures at 300 DPI** (vector where possible) with clearer captions.
- Fixed the **author block to the standard IEEEtran format** (`\IEEEauthorblockN/A`); the paper now compiles cleanly in IEEE conference style (7 pages).

---

## Reviewer #1

**1. Include a flowchart to make the methodology more understandable.**
Added — new **Fig. 2** is a step-by-step flowchart of the pipeline: satellite imagery → patch
classifier → per-city urban-area time series → Bi-LSTM forecast (95% CI) → regulatory alert
engine. It complements the framework diagram (Fig. 1) and is referenced at the start of Section IV.

**2. The result section needs correction; results are not clearly highlighted.**
We added a bold **"Key results"** sentence at the top of Section V summarising the three headline
outcomes (97.5% in-distribution OA, 79.1% LOCO, calibrated forecasts + 55 routed alerts), and the
best value in every table is now bold (ResNet50 in Table II, Swin-Tiny in Table III). Each results
subsection now opens by pointing to its table/figure.

**3. The figure quality is poor.**
All figures are now regenerated at **300 DPI** (`bbox_inches='tight'`), and the new methodology
flowchart and architecture diagram are vector/high-resolution. Captions were expanded to be
self-contained.

**4. The abstract and conclusion are not written properly.**
Both were rewritten. The **Abstract** now follows a clear problem → approach → results → significance
flow; the **Conclusion** is restructured into contributions, key findings, explicit limitations, an
ethics note, and future work.

**5. The paper is not in proper IEEE format.**
The author block was converted from custom minipages to the standard `\IEEEauthorblockN` /
`\IEEEauthorblockA` macros; the document uses `IEEEtran` (conference) and now compiles to a
clean two-column IEEE layout.

---

## Reviewer #2

**1. Primary novelty vs. existing SVM/RF urban studies?**
The contribution is a benchmark + systems contribution rather than a new architecture: (i) the
**first Leave-One-City-Out (LOCO) cross-city benchmark** for Indian metros; (ii) a **connected
pipeline** (classify → time series → forecast → regulatory alert) that no prior Indian study links;
(iii) a **novel three-class Transition taxonomy**; (iv) **uncertainty-aware forecasting** (first for
Indian urban growth); and (v) the empirical **CNN–Transformer reversal**. Notably, even an *optimised*
SVM (92.6%) beats the best published Indian SVM (91.01%), and deep learning still improves on it by
~6 points — so the gain is real, not a weak-baseline artefact.

**2. Improve clarity of figures/tables (higher-resolution, legends, captions, more comparisons).**
Addressed as in Reviewer #1.2/1.3: figures regenerated at 300 DPI with clearer legends and
self-contained captions; tables bold the best result; Table I (literature) and the SOTA discussion
provide quantitative comparison to published numbers.

**3. How were the 2,730 patches selected, and do they represent diversity?**
256×256 patches were extracted by sliding window over cloud-masked **pre-monsoon Sentinel-2**
composites and label-matched to ESA WorldCover 2021, retaining only patches above a valid-pixel
threshold. The three cities are deliberately distinct morphological archetypes — **coastal (Mumbai),
radial-sprawl (Delhi NCR), IT-corridor (Bangalore)** — covering ~15% of India's urban population.
They represent major-metro diversity; generalisation to Tier-2/3 cities is stated as a limitation
and future work (Section VI).

**4. Contribution of SAR–optical fusion and SSL; was an ablation done?**
Yes (Table IV / Section V-C). We report **honest negative results**: SAR–optical fusion reaches
87.9% vs. 96.7% optical-only (only 33% of patches have paired SAR, plus a pre-/post-monsoon
mismatch), and SimCLR reaches 93.2% vs. 96.9% for ImageNet initialisation (ImageNet wins given
sufficient labels). Both are reported transparently with their causes rather than omitted.

**5. LOCO shows an 18% cross-city drop — primary factors?**
City-specific spectral/morphological signatures (the t-SNE analysis shows clear per-city clustering),
divergent growth morphologies, and CNNs over-fitting city-specific textures. Delhi NCR is the hardest
held-out target (gradual radial sprawl) and Bangalore the easiest. The 18% gap sits squarely within
the **15–25% benchmark-to-deployment gap** reported by a 2025 systematic review of 89 Sentinel-2
studies, independently validating our number.

**6. Why does Swin-Tiny outperform ResNet50 cross-city despite lower in-distribution accuracy?**
Self-attention learns abstract, **transferable** urban structure, whereas CNN filters memorise
**local, city-specific** textures. ResNet50 drops 20.4% under LOCO versus Swin-Tiny's 14.5%. This is
consistent with Bai *et al.* (2021) on the superior OOD robustness of transformers; the small
in-distribution dataset favours the CNN, but generalisation favours attention.

**7. How was uncertainty quantified and validated in the Bi-LSTM forecaster?**
Via **Monte-Carlo Dropout** — 50 stochastic forward passes at inference yield the mean and a 95%
confidence interval from the spread. We validate on a **temporal split** (train 1990–2015, val
2016–2019, test 2020–2023) and benchmark against a Ridge baseline (standalone R²=0.956; integrated
real-anchor R²=0.559). We note honestly that the 2024–2035 intervals are calibration/epistemic
estimates — future ground truth is inherently unavailable — and are validated indirectly on the test
period; this is stated as a limitation.

**8. What assumptions underlie the encroachment alerts, and how realistic are they?**
Predicted-expansion zones are intersected with a database of **10 regulatory zone types and 30+ named
protected areas**, with severity thresholds and rule-based routing. It is a **simulation-backed
proof-of-concept**: it encodes *real* law (CRZ 2019, Forest Conservation Act 1980, Wetland Rules 2017)
and the correct authority mappings, but is **not yet validated against municipal enforcement records
or live feeds**, and is explicitly **advisory / human-in-the-loop** rather than autonomous. This is
clearly scoped as a limitation and future-work item.

**9. Validated on independent datasets / imagery beyond ESA WorldCover 2021?**
Partially: labels are **cross-validated against Google Dynamic World**; we perform a **temporal
validation** by applying the 2021-trained model to 2019 and 2023 Sentinel-2 without retraining
(consistent extents — Mumbai +25%); and the change-detection component is evaluated on the independent
public **LEVIR-CD** benchmark (F1 = 0.949). We do not yet validate against a hand-labelled independent
Indian dataset (none currently exists) — acknowledged as the key future-work direction.

---

We believe these revisions address all comments and have strengthened the manuscript. We thank the
reviewers again for their time and feedback.
