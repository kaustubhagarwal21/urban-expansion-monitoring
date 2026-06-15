# CHANDICON 2026 — Demo & Talk Guide

Everything you need to present *The Living Map* live, plus a fallback if the venue
tech fails. The in-app **Presenter Mode** (nav 13) is your live teleprompter; this
file is the printable backup + recording plan.

---

## 1. Before you leave / before the session

- [ ] Laptop charged + charger in bag.
- [ ] `cd webapp; ./run.ps1` works offline (airplane mode test) — backend :8000, frontend :5173.
- [ ] Browser zoom set so text is readable from the back of the room (Ctrl + once or twice).
- [ ] **Fallback video recorded and saved locally** (see §3) and open in a second browser tab.
- [ ] Skim the **Reviewer Q&A** page (nav 12) once — that's your defense sheet.
- [ ] Close Slack/email/notifications. Full-screen the browser (F11).
- [ ] Have the paper PDF open in another tab.

---

## 2. The 11-minute talk (matches Presenter Mode, nav 13)

| # | Page | ~min | Say (one line) |
|---|------|------|----------------|
| 1 | Overview | 1.5 | The problem + "one connected system, accepted at CHANDICON." Point at the pipeline + Fig 1. |
| 2 | Live Classification | 2.0 | Real CPU inference + Grad-CAM; introduce the 3 classes. |
| 3 | Urban Growth | 1.0 | Classify → sum per city → 30-year growth curve. |
| 4 | Sprawl Forecast | 1.5 | "Speak to the **shaded band** — uncertainty is the novelty." |
| 5 | Encroachment Alerts | 1.5 | Map + severity + regulatory routing. |
| 6 | Benchmarks | 1.5 | 97.5%, then the **LOCO ranking reversal** (the money result). |
| 7 | Novelty | 1.0 | The pitch + top 2 pillars. |
| 8 | Limitations | 0.5 | Own the limits, including the ethics note. |
| 9 | Overview (close) | 0.5 | Repo on GitHub, thank you, questions. |

Your two most memorable beats: **the shaded forecast band** and **the LOCO reversal**. Land those.

---

## 3. Fallback screen-capture (record this!)

Conference projectors and WiFi fail. Record a **2.5–3 min** silent screen capture of the
working demo so you can play it if the live app won't cooperate.

**How to record (any one):**
- Windows: `Win + Alt + R` (Xbox Game Bar) records the active window.
- OBS Studio (free) → Display Capture → Start Recording.
- PowerPoint → Insert → Screen Recording.

**Shot list (rough seconds):**
1. (0:00–0:20) Overview — slow scroll through stat cards + pipeline + Fig 1.
2. (0:20–0:55) Live Classification — click an Urban patch, let it classify, hover Grad-CAM, switch ResNet50 → Swin-Tiny.
3. (0:55–1:15) Urban Growth — the multi-city time series.
4. (1:15–1:45) Forecast — switch Mumbai → Delhi → Bangalore; pause on the shaded CI band.
5. (1:45–2:15) Alerts — filter to CRITICAL, hover a couple of markers, scroll the feed.
6. (2:15–2:45) Benchmarks — leaderboard, then the LOCO table (linger on Swin > ResNet50).
7. (2:45–3:00) Novelty — the pitch line.

Save as `webapp/demo_fallback.mp4` (git-ignored) and keep a copy on a USB stick.

---

## 4. If something breaks mid-demo

- **App won't load** → play the fallback video, narrate over it.
- **Backend offline (sidebar dot red)** → the data pages still need it; restart with `./run.ps1`, or switch to the video.
- **A question you don't know** → "Great question — we cover that in our limitations; the short answer is …" then open the **Reviewer Q&A** page.
- **Time running short** → skip beats 3 and 8; never skip Forecast (4) or Benchmarks (6).

---

## 5. Q&A live trick

On the **Live Classification** page you can drag-and-drop any `.npy` patch (shape `(6,256,256)`)
and classify it on the spot — useful if an audience member is technical and wants to poke at it.
