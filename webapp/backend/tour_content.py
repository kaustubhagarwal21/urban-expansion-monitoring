"""A plain-language "Start Here" tour of the whole project.

Written so a complete non-expert (a family member, a non-CS audience member)
can understand what this is in two minutes. Shown automatically on first visit
and re-openable from the top bar. Drives the Welcome modal.
"""

TOUR = [
    {
        "kicker": "Start here",
        "title": "What is this, in one breath?",
        "body": "A system that watches Indian cities grow — from space. It uses satellite photos and AI to see where cities are spreading, predict where they'll grow next, and raise an alarm when construction creeps into protected forests, lakes, or coastlines.",
    },
    {
        "kicker": "Why it matters",
        "title": "The problem we're solving",
        "body": "India's big cities are exploding in size. Laws protect forests, wetlands and coasts — yet illegal construction keeps happening there. Checking a whole city on foot is impossible. Satellites already photograph everything; what was missing is software that turns those photos into something a city planner can actually act on.",
    },
    {
        "kicker": "How it works",
        "title": "The AI in four simple steps",
        "points": [
            "Look at each small square of land in a satellite photo and label it: City, Not-City, or 'Edge' (a place that's actively turning into city).",
            "Add up the 'City' squares each year — that draws a graph of how the city has grown.",
            "Predict the next 10+ years of growth, and honestly show how sure (or unsure) it is.",
            "If predicted growth would hit a protected zone, send an alert to the right government authority.",
        ],
    },
    {
        "kicker": "What's new",
        "title": "Why it's special",
        "body": "It's the first system to do all four steps together for Indian cities; the first to test whether the AI still works on a city it has never seen before; and the first to tell you how confident its forecasts are. The research behind it was accepted at the IEEE CHANDICON 2026 conference.",
        "goto": {"route": "/novelty", "label": "See what's novel"},
    },
    {
        "kicker": "Get going",
        "title": "How to explore this app",
        "body": "Use the menu on the left. A good first path: Overview → Live Classification (classify a real photo yourself) → Urban Growth → Sprawl Forecast → Encroachment Alerts or the 3D Globe. Tip: hover any underlined word or the small ⓘ icons for a plain-English definition, and every page has a blue 'How to read this' box.",
        "goto": {"route": "/classify", "label": "Try Live Classification"},
    },
]
