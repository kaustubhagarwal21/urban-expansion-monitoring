"""Presentation-ready novelty content.

Cleaned and de-duplicated from outputs/paper/novelty_statement.json (which has
double-encoding artifacts) + findings_analysis.md. Drives the "Novelty" page so
the contribution story is front-and-centre for the CHANDICON talk.
"""

PITCH = (
    "This is not a new neural architecture. The novelty is a complete, statistically-"
    "rigorous, India-specific system: the first Leave-One-City-Out cross-city benchmark "
    "for Indian metros, a connected classify -> time-series -> forecast -> regulatory-alert "
    "pipeline, and a surprising empirical finding -- the CNN-Transformer ranking reversal -- "
    "all validated with 3-seed statistics on real Sentinel-2 data from Mumbai, Delhi NCR, "
    "and Bangalore."
)

# Each pillar: tag, title, what (plain), why_novel (the no-prior-work angle), evidence.
PILLARS = [
    {
        "tag": "First-of-its-kind",
        "title": "First Indian multi-city LOCO benchmark",
        "what": "We train on two cities and test on a third the model has never seen, across all three metros.",
        "why_novel": "No published work evaluates cross-city transfer learning across Indian metros with a Leave-One-City-Out protocol.",
        "evidence": "3 cities x 3 seeds; quantifies an 18% in-distribution -> cross-city drop.",
    },
    {
        "tag": "Surprising finding",
        "title": "CNN-Transformer ranking reversal",
        "what": "The best model flips depending on the test: ResNet50 wins on familiar cities, Swin-Tiny wins on unseen ones.",
        "why_novel": "First demonstration of this reversal on satellite urban classification -- and the first for Indian metros.",
        "evidence": "In-dist: ResNet50 97.5% > Swin 93.6%. LOCO: Swin 79.1% > ResNet50 77.1% (ResNet drops 20.4%, Swin only 14.5%).",
    },
    {
        "tag": "Systems",
        "title": "One connected pipeline, not five experiments",
        "what": "Classification feeds a city growth timeline, which feeds an LSTM forecast, which feeds the alert engine -- end to end.",
        "why_novel": "No published work connects satellite classification -> LSTM forecasting -> forecast-conditioned regulatory alerts in a single pipeline.",
        "evidence": "Executed run: 21 real GeoTIFFs (9 Sentinel-2 + 12 Landsat) -> time series -> 2024-2035 forecast -> 55 alerts, in 26 min.",
    },
    {
        "tag": "Method",
        "title": "Novel 3-class Transition taxonomy",
        "what": "A third class drawn as a 100m band around urban edges -- exactly the mixed, actively-growing pixels.",
        "why_novel": "No published Indian urban study uses a Transition buffer; all use binary urban/non-urban that mislabels boundary pixels.",
        "evidence": "Directly targets the mixed-pixel problem at 10m; the full loss reaches Transition F1 = 0.98.",
    },
    {
        "tag": "First-of-its-kind",
        "title": "Uncertainty-quantified forecasting",
        "what": "Forecasts to 2035 come with 95% confidence bands via Monte-Carlo Dropout, for risk-aware planning.",
        "why_novel": "No prior Indian urban-expansion study provides forecast uncertainty -- existing CA-Markov/ANN work gives point estimates only.",
        "evidence": "50 MC forward passes -> 95% CI per city per year (2024-2035).",
    },
    {
        "tag": "First-of-its-kind",
        "title": "India-specific regulatory alert engine",
        "what": "Alerts are checked against Indian environmental law and routed to the correct authority automatically.",
        "why_novel": "No published system encodes Indian environmental law for automated satellite-based encroachment detection.",
        "evidence": "10 zone types, 30+ named protected areas, 3 laws (CRZ 2019, Forest Act 1980, Wetland Rules 2017); routes to MoEFCC/CZMA/Forest Dept.",
    },
    {
        "tag": "Rigor",
        "title": "Statistical rigor + honest negatives",
        "what": "Every headline number is mean +/- std over 3 seeds with significance tests, and we report what didn't work.",
        "why_novel": "No published Indian urban classification study reports 3-seed mean +/- std with paired significance tests.",
        "evidence": "Optimized SVM (92.55%) beats published 91.01% yet still loses to DL by 5 pts; SAR fusion & SimCLR honestly reported as negatives.",
    },
    {
        "tag": "Insight",
        "title": "Local difficulty != transfer difficulty",
        "what": "The city hardest to learn locally (Bangalore) is the easiest to transfer to; Delhi NCR is the hardest to transfer to.",
        "why_novel": "First Indian cross-city analysis to separate and explain these two difficulty regimes by urban morphology.",
        "evidence": "Bangalore in-dist 80.5% (hardest) but LOCO 80.4-83.9% (easiest); Delhi LOCO 71.8-76.7% (hardest target).",
    },
]

POSITIONING = [
    {"area": "vs EuroSAT papers", "text": "Our 97.5% on real, uncurated Indian data with a 3-class taxonomy is competitive with EuroSAT SOTA (96.8-99.3%) -- on a harder dataset."},
    {"area": "vs Indian studies", "text": "Our optimized SVM (92.55%) already beats the best published Indian SVM (91.01%); DL then adds 5 more points. The higher Indian result (IRUNet 98.21%) is easier binary segmentation on one region."},
    {"area": "vs cross-city work", "text": "HighDAN / C2Seg target European and Chinese cities. We provide the first Indian-metro cross-city benchmark."},
    {"area": "vs change detection", "text": "Siamese methods reach F1 87-91% on LEVIR-CD but stop at detection. We extend to forecasting and alerting."},
    {"area": "vs urban forecasting", "text": "Indian CA-Markov / CA-ANN forecasts give point predictions disconnected from imagery. Ours is the first with 95% CI bands and connected to a real satellite pipeline."},
]
