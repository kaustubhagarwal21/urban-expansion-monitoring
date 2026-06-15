"""Presenter Mode — an in-app talk script for the ~11-minute CHANDICON demo.

Each beat tells you which page to be on, what to click, and what to say. Drives
the "Presenter Mode" page so the app doubles as a teleprompter.
"""

TOTAL_MINUTES = 11

BEATS = [
    {
        "n": 1, "route": "/overview", "page": "Overview", "min": 1.5,
        "click": "Open on the Overview page; gesture at the headline stats and the 5-stage pipeline.",
        "say": "Indian metros will hold 600 million people by 2031, and they're growing into protected forests, wetlands and coasts. Existing studies stop at single-city SVMs. We built one connected system — classify, track, forecast, alert — and it's accepted at CHANDICON. Here's the whole pipeline in one view.",
    },
    {
        "n": 2, "route": "/classify", "page": "Live Classification", "min": 2.0,
        "click": "Pick an Urban patch → it classifies live; point at the Grad-CAM heatmap. Then switch the backbone from ResNet50 to Swin-Tiny.",
        "say": "This is real inference on a real Sentinel-2 patch, running on CPU. The model says Urban with high confidence, and Grad-CAM shows it's looking at the built-up structures — not random texture. Three classes: Urban, Non-Urban, and our novel Transition class for the messy growing edge.",
    },
    {
        "n": 3, "route": "/growth", "page": "Urban Growth", "min": 1.0,
        "click": "Show the 1990–2023 multi-city time series.",
        "say": "We classify every patch and sum it per city per year, which reconstructs the built-up footprint over three decades. This curve is the input to our forecaster.",
    },
    {
        "n": 4, "route": "/forecast", "page": "Sprawl Forecast", "min": 1.5,
        "click": "Switch between cities; point at the shaded 95% confidence band.",
        "say": "A Bi-LSTM forecasts each city to 2035. The key point isn't the line — it's the shaded band. We give planners a confidence interval via Monte-Carlo Dropout. No prior Indian urban-forecasting study provides uncertainty. That's a first.",
    },
    {
        "n": 5, "route": "/alerts", "page": "Encroachment Alerts", "min": 1.5,
        "click": "Filter to CRITICAL; hover a marker near Mumbai/Chennai.",
        "say": "Predicted growth is checked against Indian regulatory zones — CRZ, forests, wetlands. Each alert is sized by severity and routed to the correct authority: environment ministry, coastal authority, or forest department. 55 alerts, 11 near protected zones, in a 7-city simulation.",
    },
    {
        "n": 6, "route": "/benchmarks", "page": "Benchmarks", "min": 1.5,
        "click": "Show the leaderboard (ResNet50 97.5%), then point at the LOCO table.",
        "say": "Everything is 3-seed mean±std. ResNet50 hits 97.5%, six points over the best published Indian SVM. But here's the money result — on a held-out city the ranking REVERSES: the Transformer, Swin-Tiny, generalises better than the CNN. CNNs memorise city-specific textures; attention learns transferable structure.",
    },
    {
        "n": 7, "route": "/novelty", "page": "Novelty", "min": 1.0,
        "click": "Read the pitch line; point at the 'First-of-its-kind' and 'Surprising finding' cards.",
        "say": "So the novelty isn't a new architecture — it's the first Indian cross-city benchmark, the first connected classify-to-alert pipeline, the CNN-Transformer reversal, and uncertainty-aware forecasting, all with honest negative results.",
    },
    {
        "n": 8, "route": "/limitations", "page": "Limitations", "min": 0.5,
        "click": "Glance at the limitations cards — especially the Ethics one.",
        "say": "We're upfront: three cities, weak labels, simulated alerts, mismatched SAR. And the alerts are advisory only, human-in-the-loop, so they can't be misused against informal settlements.",
    },
    {
        "n": 9, "route": "/overview", "page": "Close", "min": 0.5,
        "click": "Return to Overview as the closing frame.",
        "say": "Everything you saw runs from real satellite data through one pipeline to policy-actionable alerts. Code and data are on GitHub. Thank you — happy to take questions.",
    },
]

TIPS = [
    "Rehearse to 10 minutes so you have buffer for questions and tech delays.",
    "Have the fallback screen-capture video open in a second tab in case the live demo or WiFi fails.",
    "The Reviewer Q&A page (nav 12) is your defense cheat-sheet — skim it right before the session.",
    "Speak to the SHADED BAND on the forecast and the LOCO REVERSAL on benchmarks — those two are your most memorable points.",
    "Run everything offline: the demo needs no internet (fonts, map, and sample patches are bundled).",
]
