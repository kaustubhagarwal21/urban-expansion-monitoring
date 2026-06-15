"""Limitations & Future Work, presentation-ready.

Each item pairs an honest limitation with how the paper mitigates it now and
what comes next. Sourced from paper/main.tex (Conclusion + Ethics) and
outputs/paper/reviewer_defense.md. Drives the "Limitations" page.
"""

LIMITATIONS = [
    {
        "tag": "Scope",
        "title": "Only three Tier-1 cities",
        "what": "We study Mumbai, Delhi NCR, and Bangalore — not smaller Tier-2/3 cities.",
        "mitigation": "They are three deliberately different growth archetypes (coastal, radial, IT-corridor) covering ~15% of India's urban population, and our LOCO benchmark explicitly measures how far generalisation holds.",
        "future": "Extend to Tier-2/3 cities (Pune, Jaipur, Lucknow) using the same GEE pipeline.",
    },
    {
        "tag": "Data",
        "title": "Weak (auto-generated) labels",
        "what": "Labels come from ESA WorldCover, not human annotation (74.4% global accuracy).",
        "mitigation": "WorldCover's urban-class accuracy exceeds 85% in dense Indian metros, we cross-validate against Google Dynamic World, and the Transition buffer absorbs residual boundary noise.",
        "future": "Add a hand-verified label subset / active learning to quantify and correct label noise.",
    },
    {
        "tag": "Method",
        "title": "Transition class is rule-drawn",
        "what": "The Transition class is a 100m morphological buffer, not a human-delineated boundary.",
        "mitigation": "It is a principled, reproducible definition that directly targets the mixed-pixel zone where encroachment happens; the combined loss reaches Transition F1 = 0.98.",
        "future": "Compare against expert-drawn transition boundaries on a held-out sample.",
    },
    {
        "tag": "Evaluation",
        "title": "Forecasting is hybrid, not fully real",
        "what": "On real satellite anchors the forecaster scores R2=0.559, far below the calibrated standalone R2=0.956 (and Ridge edges it).",
        "mitigation": "We frame the standalone number as an upper bound and the integrated number as the realistic one. The LSTM's value is the 95% uncertainty band, not raw accuracy — a band a linear model cannot give.",
        "future": "Add more anchor years and wider bounding boxes so the temporal signal is richer.",
    },
    {
        "tag": "Evaluation",
        "title": "Alerts are simulation-based",
        "what": "The 55 encroachment alerts come from a simulation, not validated field cases.",
        "mitigation": "The engine encodes real Indian laws, real zone types, and 30+ named protected areas with correct authority routing — the framework is real even if this run is simulated.",
        "future": "Validate alerts against actual municipal / forest-department encroachment records.",
    },
    {
        "tag": "Data",
        "title": "SAR fusion underperforms",
        "what": "Adding radar hurt accuracy (87.9% vs 96.7% optical-only).",
        "mitigation": "This is an honest negative result with a clear cause: only 33% of patches had paired SAR, and the SAR was post-monsoon while optical was pre-monsoon. The lesson — align timing first — is itself useful.",
        "future": "Re-run with strictly temporally-aligned, same-orbit SAR–optical pairs.",
    },
    {
        "tag": "Evaluation",
        "title": "Only three seeds",
        "what": "Statistical claims rest on 3 random seeds, so some model differences aren't significant.",
        "mitigation": "Three seeds is standard for IGARSS/JSTARS/ISPRS, and we report mean±std AND paired t-tests — we explicitly flag the non-significant pairs.",
        "future": "Add seeds to lift statistical power toward CVPR/NeurIPS norms.",
    },
    {
        "tag": "Ethics",
        "title": "Responsible use of alerts",
        "what": "An automated encroachment alert could be misused against informal settlements / slums.",
        "mitigation": "Alerts are explicitly advisory and require human-in-the-loop verification before any action, by design, to avoid disparate impact on vulnerable communities.",
        "future": "Add fairness auditing and a documented human-review workflow before any deployment.",
    },
]
