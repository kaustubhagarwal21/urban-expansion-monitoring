"""Anticipated reviewer / audience questions with layman-friendly answers.

Drives the "Reviewer Q&A" page so every likely question at CHANDICON can be
answered live. Source: outputs/paper/reviewer_defense.md.
Each item: severity, question, plain (one-line layman answer), points (defense
bullets), one_liner (the if-pressed punchline).
"""

QA = [
    # ---------------- CRITICAL ----------------
    {
        "id": 1, "severity": "CRITICAL", "asker": "Any venue",
        "question": "Only 3 cities — how can this claim to generalise to all of Indian urban expansion?",
        "plain": "We never claim pan-India coverage. The 3 cities are deliberately different archetypes, and our LOCO test is exactly how we measure (and report) the limits of generalisation.",
        "points": [
            "Mumbai (coastal), Delhi NCR (radial sprawl), Bangalore (IT-corridor) are 3 fundamentally different growth patterns.",
            "Together they hold ~15% of India's urban population.",
            "The LOCO benchmark explicitly tests generalisation — the 18% drop quantifies the real gap.",
            "Our contribution is the evaluation framework; adding Tier-2 cities is straightforward future work with the same GEE pipeline.",
        ],
        "one_liner": "Three metros with distinct morphologies covering 15% of India's urban population — and LOCO is how we honestly measure transfer.",
    },
    {
        "id": 2, "severity": "CRITICAL", "asker": "CVPR / NeurIPS reviewers",
        "question": "2,730 patches is small for deep learning — are the results reliable?",
        "plain": "Small but real, with transfer learning and 3-seed validation. Our super-tuned SVM (92.6%) even beats the published one — so the data is clean and DL's 5-point lead is real.",
        "points": [
            "Comparable in size to published Indian studies; each patch is 6.5 km² of REAL imagery, not synthetic.",
            "Risk mitigated by ImageNet transfer, 3-stage fine-tuning, and low variance (ResNet50 std=0.2%).",
            "Tight bounding boxes maximise discriminative signal per patch.",
            "Improved SVM (92.55%) > published SVM (91.01%) confirms data quality.",
        ],
        "one_liner": "Small-but-real with proper validation beats large synthetic benchmarks; our optimised SVM beats the published one, yet DL still wins by 5 points.",
    },
    {
        "id": 3, "severity": "CRITICAL", "asker": "Any ML-savvy reviewer",
        "question": "ResNet50 beats Swin-Tiny — doesn't the literature say Transformers are better?",
        "plain": "On familiar cities yes, ResNet50 wins — but on unseen cities the ranking REVERSES and Swin wins. That's a known small-data vs OOD effect, and it's one of our findings, not a bug.",
        "points": [
            "In-distribution: ResNet50 97.5% > Swin 93.6% (small data favours fewer parameters).",
            "Cross-city LOCO: Swin 79.1% > ResNet50 77.1% (self-attention generalises better).",
            "Swin's higher variance (2.6% vs 0.2%) shows it overfits on small data — matches literature.",
            "Confirms Bai et al. (2021) on satellite data for the first time.",
        ],
        "one_liner": "Small data favours CNNs; cross-city favours Transformers. The reversal is the finding.",
    },
    {
        "id": 4, "severity": "CRITICAL", "asker": "CVPR / NeurIPS reviewers",
        "question": "No novel architecture — you just apply existing models to a new dataset.",
        "plain": "Correct, and we say so. Our novelty is the benchmark, the LOCO evaluation, the Transition class, and the connected pipeline — a benchmark/systems paper, not an architecture paper.",
        "points": [
            "First LOCO cross-city benchmark for Indian urban expansion.",
            "The Swin-beats-ResNet-cross-city finding is a new empirical insight.",
            "The classify→forecast→alert pipeline is a systems contribution.",
            "Progressive fine-tuning + combined loss is a methodological contribution for this task.",
        ],
        "one_liner": "The models are existing; the benchmark, evaluation framework, and pipeline integration are the novelty.",
    },
    # ---------------- MODERATE ----------------
    {
        "id": 5, "severity": "MODERATE", "asker": "Any venue",
        "question": "The ablation shows FPN and the combined loss give marginal benefit — why include them?",
        "plain": "Their value is in the rare Transition class, not overall accuracy. The full method nails Transition (F1 0.98), which is the class that matters most for encroachment.",
        "points": [
            "FPN's +0.3% is concentrated on multi-scale urban boundaries.",
            "Combined loss matches CE-only on aggregate OA but balances the minority Transition class far better.",
            "Reporting marginal differences honestly is a strength, not a weakness.",
        ],
        "one_liner": "We keep them for class-balanced performance on the encroachment-relevant Transition class; the ablation shows the trade-off transparently.",
    },
    {
        "id": 6, "severity": "MODERATE", "asker": "Any venue",
        "question": "SAR fusion underperforms optical-only (87.9% vs 96.7%). Why include Pillar I?",
        "plain": "It's a realistic negative result with a clear cause (only 33% paired, wrong season). Reporting it is more useful than cherry-picking a win.",
        "points": [
            "Only 910/2,730 patches had SAR matches.",
            "Post-monsoon SAR paired with pre-monsoon optical — backscatter differs by season.",
            "Published work shows SAR helps mainly under cloud / strict temporal alignment.",
        ],
        "one_liner": "With temporally aligned pairs SAR would help; the current result honestly establishes the baseline and the lesson: align timing first.",
    },
    {
        "id": 7, "severity": "MODERATE", "asker": "Any venue",
        "question": "SimCLR underperforms ImageNet init (93.2% vs 96.9%). So self-supervision doesn't work?",
        "plain": "With enough labels, ImageNet transfer is expected to win. SimCLR's advantage only shows up when labels are very scarce (<500).",
        "points": [
            "Literature shows SSL wins mainly in low-label regimes.",
            "Our SimCLR had only 20 pretrain epochs vs industrial 100–800 on millions of patches.",
            "Meaningful guidance: with adequate labels and a good backbone, skip SSL.",
        ],
        "one_liner": "With 2,730 labels, ImageNet transfer remains the practical choice; SSL's edge appears only under scarce labels.",
    },
    {
        "id": 8, "severity": "MODERATE", "asker": "JSTARS / journal reviewers",
        "question": "Pillar IV's LSTM loses to a simple Ridge regression. Why use it?",
        "plain": "On raw accuracy Ridge wins, and we say so. The LSTM's value is the uncertainty bands and non-linear policy-event modelling that Ridge can't provide.",
        "points": [
            "With only 7 cities and ~30 time points, linear models are naturally competitive.",
            "LSTM provides MC-Dropout 95% confidence intervals; Ridge cannot.",
            "LSTM captures policy shocks (Smart City Mission, COVID) that Ridge treats as flat trends.",
        ],
        "one_liner": "We honestly report Ridge wins on aggregate error; the LSTM's value is uncertainty quantification and policy-event modelling.",
    },
    {
        "id": 9, "severity": "MODERATE", "asker": "RS-domain reviewers",
        "question": "Delhi NCR and Bangalore show ~100% urban fraction — your time series is flat.",
        "plain": "The boxes are tight around already-saturated urban cores, so near-100% is factually correct. Mumbai (with sea + national park inside) varies more.",
        "points": [
            "Tight ~30–50 km boxes maximise patch quality for the classifier.",
            "These cores are genuinely among the densest urban areas on earth.",
            "Mumbai varies 73–94% due to coastal water and Sanjay Gandhi NP.",
        ],
        "one_liner": "Tight boxes were optimised for classifier training; expanding to peri-urban fringes is the fix for richer temporal dynamics.",
    },
    {
        "id": 10, "severity": "MODERATE", "asker": "Statistically-minded reviewers",
        "question": "Only 3 seeds — can you really claim statistical significance?",
        "plain": "3 seeds is standard for IGARSS/JSTARS/ISPRS, and we report both mean±std AND paired t-tests — more rigorous than most RS papers.",
        "points": [
            "ResNet50 vs MobileNetV3 is significant (p=0.038); other pairs aren't, and we say so.",
            "ResNet50's 0.2% std shows stability regardless of initialisation.",
            "5+ seeds is expected only at NeurIPS/ICML.",
        ],
        "one_liner": "We acknowledge limited power with 3 seeds and report non-significant pairs honestly; more seeds would only strengthen the claims.",
    },
    # ---------------- MINOR ----------------
    {
        "id": 11, "severity": "MINOR", "asker": "RS reviewers",
        "question": "IRUNet (98.21%) beats your ResNet50 (97.5%). Why not use segmentation?",
        "plain": "Different, easier task: IRUNet does binary segmentation on one region; we do 3-class classification across 3 cities WITH cross-city testing.",
        "points": [
            "Binary segmentation inflates accuracy (most pixels are easy background).",
            "Our Transition class (15%) explicitly handles boundary ambiguity — harder.",
            "IRUNet has no cross-city evaluation; our LOCO reveals real domain gaps.",
        ],
        "one_liner": "97.5% on a harder 3-class multi-city task with LOCO is within 0.7 points of their easier binary single-region 98.21%.",
    },
    {
        "id": 12, "severity": "MINOR", "asker": "Any venue",
        "question": "Why a 3-class taxonomy? Binary urban/non-urban is standard.",
        "plain": "Binary forces ambiguous edge pixels into a wrong class. The Transition band gives those mixed pixels an honest home — and it's the class encroachment actually happens in.",
        "points": [
            "Mixed-pixel problem at 10m is well known.",
            "Transition = 100m morphological buffer absorbs that ambiguity.",
            "Combined loss specifically improves Transition balance.",
        ],
        "one_liner": "The Transition class is our solution to the mixed-pixel problem — it models boundary ambiguity instead of hiding it.",
    },
    {
        "id": 13, "severity": "MINOR", "asker": "RS reviewers",
        "question": "Why ESA WorldCover for labels? It's not human-annotated ground truth.",
        "plain": "It's the best free 10m global label product, and its urban-class accuracy in dense Indian metros far exceeds its 74.4% global figure. We also cross-check with Dynamic World.",
        "points": [
            "Manual annotation of 2,730 patches across 3 cities is impractical.",
            "Cross-validated against Google Dynamic World.",
            "Label noise is absorbed by the Transition buffer class.",
        ],
        "one_liner": "WorldCover's urban accuracy exceeds 85% in Indian metros; the Transition buffer absorbs residual boundary noise.",
    },
    {
        "id": 14, "severity": "MINOR", "asker": "RS reviewers",
        "question": "Why not use BigEarthNet or fMoW alongside EuroSAT?",
        "plain": "Our contribution is an Indian-specific benchmark. Those datasets are different tasks and wouldn't strengthen the core story.",
        "points": [
            "Final results use only Indian data (EuroSAT is just a debugging fallback).",
            "BigEarthNet / fMoW address different tasks.",
            "We chose depth on Indian data over breadth across benchmarks.",
        ],
        "one_liner": "We prioritise depth on Indian urban expansion over breadth across general RS benchmarks.",
    },
    {
        "id": 15, "severity": "MINOR", "asker": "RS reviewers",
        "question": "How do you handle cloud cover in Sentinel-2?",
        "plain": "We mask clouds using the scene-classification band, pick the clear pre-monsoon season, and median-composite over it.",
        "points": [
            "SCL band cloud masking before compositing.",
            "Pre-monsoon (Jan–Mar) = minimum cloud in India.",
            "Median compositing removes residual cloud artifacts.",
        ],
        "one_liner": "SCL masking + seasonal median compositing; pre-monsoon India has <10% cloud in most metros.",
    },
    {
        "id": 16, "severity": "MINOR", "asker": "RS reviewers",
        "question": "What about mixed pixels and the MAUP at 10m?",
        "plain": "That's exactly what the Transition class is for — it models the mixed urban-rural boundary zone instead of forcing it into Urban or Non-Urban.",
        "points": [
            "Mixed pixels mainly occur at urban-rural boundaries.",
            "The 100m Transition buffer explicitly covers those zones.",
            "3-class taxonomy handles ambiguity better than binary.",
        ],
        "one_liner": "The Transition class is our answer to mixed pixels — model the ambiguity, don't force a wrong label.",
    },
    {
        "id": 17, "severity": "MINOR", "asker": "Application reviewers",
        "question": "Can this actually be deployed operationally?",
        "plain": "Yes — MobileNetV3 hits SVM-level accuracy at 184 patches/sec in 30.5 MB, and the alert engine already routes to specific Indian authorities. (This very demo is the front end.)",
        "points": [
            "MobileNetV3-Small: 91.5% OA, 3.4M params, 30.5 MB, 184.4 patches/sec.",
            "The 6-point gap vs ResNet50 is the explicit accuracy-for-speed cost.",
            "Alerts route to MoEFCC / CZMA / State Forest Dept; full pipeline latency 77.5 ms.",
        ],
        "one_liner": "MobileNetV3 gives edge-ready speed; ResNet50 gives peak accuracy — operators pick per constraint. Full deployment needs gov-data integration.",
    },
]

VENUE_TIPS = [
    {"venue": "IGARSS / IEEE GRSL", "want": "Real data, practical methodology, clear results", "tip": "Lead with the LOCO benchmark as the main contribution; keep it concise (4 pages)."},
    {"venue": "IEEE JSTARS", "want": "Depth, comprehensive evaluation, application relevance", "tip": "Emphasise the systems integration and per-city analysis; explain why the LSTM is still valuable despite Ridge."},
    {"venue": "ISPRS", "want": "Methodological rigour, reproducibility", "tip": "Emphasise the GEE pipeline, WorldCover labels, and reproducibility appendix."},
    {"venue": "ACM SIGSPATIAL", "want": "Spatial-computing novelty, system design", "tip": "Lead with the pipeline architecture and the alert engine."},
    {"venue": "CVPR EarthVision", "want": "Novel method, strong baselines, visuals", "tip": "Frame as a benchmark/dataset contribution; lean on GradCAM, t-SNE, and the figures."},
]
