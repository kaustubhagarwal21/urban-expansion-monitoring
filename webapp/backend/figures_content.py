"""Curated, captioned list of the real paper figures for the gallery page.

Only figures that actually exist in outputs/figures/ are returned. Each figure
carries a short caption, a "read" (how to read the chart) and a "takeaway" (the
one-line point), so every figure is fully explained in plain language.
"""
import app_paths as P

FIGURES = [
    {"file": "fig1_architecture.png", "group": "Architecture", "title": "Five-pillar framework",
     "caption": "The end-to-end pipeline: imagery -> classifier -> urban-area time series -> Bi-LSTM forecast -> regulatory alerts.",
     "read": "Follow the arrows left-to-right: raw satellite imagery enters on the left, is classified patch-by-patch, aggregated into a per-city growth curve, forecast forward, then checked against protected zones to raise alerts.",
     "takeaway": "Everything is one connected system, not five separate experiments — that connection is the contribution."},
    {"file": "sample_patches.png", "group": "Data", "title": "Example input patches",
     "caption": "Sample 256x256 Sentinel-2 patches across the three Indian metros, with their Urban / Non-Urban / Transition labels.",
     "read": "Each tile is one 256x256 input the model sees (about 6.5 km^2 of ground). Colour is a natural-colour composite; the label says what class it belongs to.",
     "takeaway": "These are real satellite patches, not synthetic data — the basis of every result."},
    {"file": "fig2_model_comparison.png", "group": "Results", "title": "Model comparison (3-seed)",
     "caption": "Overall accuracy of all six models with error bars over three seeds.",
     "read": "Each bar is a model's mean accuracy; the whisker on top is the variation across 3 random seeds — shorter whisker = more reliable.",
     "takeaway": "ResNet50 is both the most accurate and the most stable (tiny whisker)."},
    {"file": "fig3_loco_heatmap.png", "group": "Results", "title": "LOCO cross-city heatmap",
     "caption": "Leave-One-City-Out transfer accuracy.",
     "read": "Rows/cells show how well a model trained on other cities does on a held-out city. Darker/cooler = lower accuracy on the unseen city.",
     "takeaway": "Accuracy drops ~18% on an unseen city — the real cross-city domain gap."},
    {"file": "fig4_urban_timeseries.png", "group": "Results", "title": "Urban expansion time series",
     "caption": "Built-up area 1990-2035 per city.",
     "read": "Solid line = satellite-measured built-up area; dashed = the model's forecast; the shaded band is the 95% confidence range. Wider band = more uncertainty.",
     "takeaway": "The shaded band is the novelty — uncertainty-aware forecasting, a first for Indian metros."},
    {"file": "fig5_ablation.png", "group": "Results", "title": "Ablation study",
     "caption": "Full method vs. no-FPN vs. CE-only on EfficientNet-B0.",
     "read": "Each bar removes one component. Compare its height to the full method to see how much that component contributed.",
     "takeaway": "FPN and the combined loss help mainly the rare Transition class, not raw accuracy — and we say so honestly."},
    {"file": "fig6_pillar_comparison.png", "group": "Results", "title": "Pillar I + II comparison",
     "caption": "Optical-only vs. SAR fusion, and ImageNet vs. SimCLR pre-training.",
     "read": "Compare the paired bars: in each pair the left is the proposed/baseline and the right is the alternative being tested.",
     "takeaway": "Both alternatives (SAR fusion, SimCLR) lose here — honest negative results with clear reasons."},
    {"file": "fig8_efficiency_accuracy.png", "group": "Results", "title": "Efficiency vs. accuracy",
     "caption": "Model size / speed plotted against accuracy.",
     "read": "Each dot is a model: x-axis is cost (params or latency), y-axis is accuracy. Top-left = accurate AND cheap (best).",
     "takeaway": "ResNet50 sits in the sweet spot; MobileNetV3 trades ~6 points of accuracy for the smallest footprint."},
    {"file": "gradcam/gradcam_resnet50.png", "group": "Explainability", "title": "Grad-CAM — ResNet50",
     "caption": "Where ResNet50 looks when it decides a patch is Urban.",
     "read": "The heatmap overlays the patch: warm/red areas are what drove the decision, cool/transparent areas were ignored.",
     "takeaway": "ResNet50 focuses on real built-up structures and roads — it learned meaningful features, not noise."},
    {"file": "gradcam/gradcam_efficientnet_b0.png", "group": "Explainability", "title": "Grad-CAM — EfficientNet-B0",
     "caption": "EfficientNet-B0's attention on the same kind of scene.",
     "read": "Same heatmap idea: warm = influential pixels. Notice the activation is broader/fuzzier than ResNet50's.",
     "takeaway": "It spreads attention over mixed urban-vegetation edges — slightly less sharp than ResNet50."},
    {"file": "gradcam/gradcam_mobilenet_v3_small.png", "group": "Explainability", "title": "Grad-CAM — MobileNetV3",
     "caption": "The lightweight model's attention.",
     "read": "Warm = influential pixels. The tiny model fixes on a few high-contrast fragments rather than the whole structure.",
     "takeaway": "Explains why the edge model is a little weaker on ambiguous Transition scenes."},
    {"file": "fig_domain_shift_tsne.png", "group": "Analysis", "title": "t-SNE domain shift",
     "caption": "Model features projected to 2D, coloured by city.",
     "read": "Each dot is a patch's learned feature, squashed to 2D. Dots of the same colour (city) clumping together means that city has a distinct 'fingerprint'.",
     "takeaway": "Cities form separate clusters — visual proof that the cross-city gap is real, not noise."},
    {"file": "fig_per_city_confusion.png", "group": "Analysis", "title": "Per-city confusion matrices",
     "caption": "What the model predicted vs. the truth, per city.",
     "read": "In each grid, rows = true class, columns = predicted. The diagonal is correct; bright off-diagonal cells are systematic mistakes.",
     "takeaway": "Bangalore's dispersed IT-corridor fabric drives most Non-Urban -> Transition confusion."},
    {"file": "fig_failure_cases.png", "group": "Analysis", "title": "Failure cases",
     "caption": "Representative misclassified patches.",
     "read": "Each tile is one patch the model got wrong, labelled with predicted vs. true class.",
     "takeaway": "Most errors are genuine boundary ambiguity (edge of a city), not random label noise."},
    {"file": "fig7_temporal_validation.png", "group": "Analysis", "title": "Temporal validation (2019 vs 2023)",
     "caption": "The 2021-trained model applied to 2019 and 2023 imagery without retraining.",
     "read": "Compare the two years' estimated urban extents per city; the difference is the detected growth.",
     "takeaway": "Mumbai shows +25% growth, consistent with known coastal construction — the model generalises across time too."},
]


def list_figures():
    return [f for f in FIGURES if (P.FIGURES / f["file"]).exists()]
