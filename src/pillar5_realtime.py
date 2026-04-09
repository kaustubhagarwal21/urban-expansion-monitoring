"""
Pillar V: Real-Time Monitoring & Encroachment Alert System (India)
===================================================================
Near real-time monitoring pipelines for Indian metropolitan regions.

Research contributions:
  1. India-specific regulatory zone framework (CRZ, Forest Conservation Act,
     Western Ghats ESA, Wetland Rules, Green Belt notifications)
  2. Multi-severity alert classification with escalation logic
  3. Lightweight EfficientNet-based bi-temporal change detector optimized
     for low-latency inference (target: <24h from acquisition to alert)
  4. Encroachment detection on protected boundaries with buffer analysis
  5. Dashboard-ready structured alert output with geospatial metadata

Legal framework modelled:
  - Coastal Regulation Zone (CRZ) Notification 2019
  - Forest Conservation Act 1980 (amended 2023)
  - Wetland (Conservation and Management) Rules 2017
  - Western Ghats Ecologically Sensitive Area (Kasturirangan Report)
  - State-level Green Belt and Buffer Zone notifications
  - RERA (Real Estate Regulation) Act 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, time, json, copy
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier
from src.dataset import generate_patch
from src.metrics import evaluate
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ═══════════════════════════════════════════════════════
#  India Regulatory Zone Framework
# ═══════════════════════════════════════════════════════

REGULATORY_ZONES = {
    # CRZ - Coastal Regulation Zone (MoEFCC Notification 2019)
    "CRZ_I": {
        "description": "Ecologically sensitive: mangroves, corals, salt marshes, turtle nesting",
        "buffer_m": 0,  # No construction allowed
        "severity": "CRITICAL",
        "response_time_hours": 6,
        "authority": "MoEFCC / State Coastal Zone Management Authority",
        "applicable_cities": ["Mumbai", "Chennai"],
        "penalty": "Demolition + fine up to INR 1 Cr",
    },
    "CRZ_II": {
        "description": "Developed urban areas within CRZ limits",
        "buffer_m": 200,  # 200m from HTL
        "severity": "HIGH",
        "response_time_hours": 24,
        "authority": "State CZMA",
        "applicable_cities": ["Mumbai", "Chennai"],
        "penalty": "Regularization or demolition",
    },
    "CRZ_III": {
        "description": "Relatively undisturbed rural coastal areas",
        "buffer_m": 200,
        "severity": "HIGH",
        "response_time_hours": 24,
        "authority": "District CZMA",
        "applicable_cities": ["Mumbai", "Chennai"],
        "penalty": "Demolition + restoration",
    },
    # Forest Conservation Act
    "FOREST_RESERVE": {
        "description": "Reserved Forest under Forest Conservation Act 1980",
        "buffer_m": 100,
        "severity": "CRITICAL",
        "response_time_hours": 12,
        "authority": "MoEFCC / State Forest Department",
        "applicable_cities": ["Mumbai", "Bangalore", "Pune", "Chennai"],
        "penalty": "Criminal prosecution + restoration",
    },
    "PROTECTED_FOREST": {
        "description": "Protected Forest / National Park buffer",
        "buffer_m": 500,  # Eco-sensitive zone
        "severity": "CRITICAL",
        "response_time_hours": 6,
        "authority": "National Board for Wildlife",
        "applicable_cities": ["Mumbai"],  # Sanjay Gandhi NP
        "penalty": "Criminal prosecution",
    },
    # Wetland Rules 2017
    "WETLAND": {
        "description": "Notified wetland under Wetland Rules 2017",
        "buffer_m": 50,
        "severity": "CRITICAL",
        "response_time_hours": 12,
        "authority": "State Wetland Authority / National Wetland Authority",
        "applicable_cities": ["Bangalore", "Chennai", "Hyderabad"],
        "penalty": "Restoration + fine",
    },
    # Water body protection
    "LAKE_BUFFER": {
        "description": "Lake / tank buffer zone (75m from boundary)",
        "buffer_m": 75,
        "severity": "HIGH",
        "response_time_hours": 24,
        "authority": "Lake Development Authority / Municipal Corporation",
        "applicable_cities": ["Bangalore", "Hyderabad"],
        "penalty": "Demolition per court order",
    },
    # River buffers
    "RIVER_FLOODPLAIN": {
        "description": "River floodplain / riverbed encroachment",
        "buffer_m": 300,
        "severity": "CRITICAL",
        "response_time_hours": 12,
        "authority": "National Green Tribunal / State Irrigation Dept",
        "applicable_cities": ["Delhi_NCR", "Pune", "Ahmedabad", "Chennai"],
        "penalty": "NGT-ordered demolition",
    },
    # Green belt
    "GREEN_BELT": {
        "description": "Municipal green belt / agricultural zone",
        "buffer_m": 0,
        "severity": "HIGH",
        "response_time_hours": 48,
        "authority": "Municipal Corporation / Town Planning Dept",
        "applicable_cities": CITIES,
        "penalty": "Regularization or demolition",
    },
    # Western Ghats ESA
    "WESTERN_GHATS_ESA": {
        "description": "Western Ghats Ecologically Sensitive Area (Kasturirangan)",
        "buffer_m": 0,
        "severity": "CRITICAL",
        "response_time_hours": 12,
        "authority": "MoEFCC / Western Ghats Ecology Expert Panel",
        "applicable_cities": ["Pune", "Mumbai"],
        "penalty": "Ban on new construction + restoration",
    },
}

# City-specific protected zones with approximate boundaries
CITY_PROTECTED_ZONES = {
    "Mumbai": {
        "Sanjay Gandhi National Park": {"type": "PROTECTED_FOREST", "area_sqkm": 87},
        "Mangrove zones (CRZ-I)": {"type": "CRZ_I", "area_sqkm": 66},
        "CRZ coastal belt": {"type": "CRZ_II", "area_sqkm": 45},
        "Aarey Colony (green belt)": {"type": "GREEN_BELT", "area_sqkm": 13},
        "Mithi River floodplain": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 8},
        "Powai Lake buffer": {"type": "LAKE_BUFFER", "area_sqkm": 3},
    },
    "Delhi_NCR": {
        "Yamuna floodplain": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 97},
        "Aravalli Ridge (reserved forest)": {"type": "FOREST_RESERVE", "area_sqkm": 77},
        "NCR green belt": {"type": "GREEN_BELT", "area_sqkm": 250},
        "Najafgarh jheel (wetland)": {"type": "WETLAND", "area_sqkm": 10},
        "Asola Bhatti Wildlife Sanctuary": {"type": "PROTECTED_FOREST", "area_sqkm": 32},
    },
    "Bangalore": {
        "Lake boundaries (Bellandur, Varthur, etc.)": {"type": "LAKE_BUFFER", "area_sqkm": 15},
        "Bannerghatta NP buffer": {"type": "PROTECTED_FOREST", "area_sqkm": 25},
        "Hesaraghatta Lake wetland": {"type": "WETLAND", "area_sqkm": 5},
        "Agricultural green belt (north)": {"type": "GREEN_BELT", "area_sqkm": 120},
        "Turahalli forest reserve": {"type": "FOREST_RESERVE", "area_sqkm": 3},
    },
    "Hyderabad": {
        "Hussain Sagar buffer": {"type": "LAKE_BUFFER", "area_sqkm": 6},
        "Osman Sagar / Himayat Sagar": {"type": "LAKE_BUFFER", "area_sqkm": 35},
        "Musi riverbed": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 12},
        "ORR green belt": {"type": "GREEN_BELT", "area_sqkm": 80},
        "KBR National Park": {"type": "PROTECTED_FOREST", "area_sqkm": 1.5},
    },
    "Chennai": {
        "Pallikaranai marsh (wetland)": {"type": "WETLAND", "area_sqkm": 5},
        "Adyar estuary (CRZ-I)": {"type": "CRZ_I", "area_sqkm": 2},
        "CRZ coastal belt": {"type": "CRZ_II", "area_sqkm": 30},
        "Guindy NP buffer": {"type": "PROTECTED_FOREST", "area_sqkm": 2.7},
        "Adyar/Cooum river floodplain": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 15},
    },
    "Pune": {
        "Western Ghats ESA zones": {"type": "WESTERN_GHATS_ESA", "area_sqkm": 200},
        "Sinhagad Fort protected area": {"type": "FOREST_RESERVE", "area_sqkm": 15},
        "Mula-Mutha river buffer": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 10},
        "Pashan Lake buffer": {"type": "LAKE_BUFFER", "area_sqkm": 2},
        "Pune cantonment green belt": {"type": "GREEN_BELT", "area_sqkm": 25},
    },
    "Ahmedabad": {
        "Sabarmati riverfront buffer": {"type": "RIVER_FLOODPLAIN", "area_sqkm": 8},
        "Thol Bird Sanctuary buffer": {"type": "WETLAND", "area_sqkm": 7},
        "SIT area green belt": {"type": "GREEN_BELT", "area_sqkm": 40},
        "Nalsarovar wetland buffer": {"type": "WETLAND", "area_sqkm": 120},
    },
}

# Alert severity levels with color codes and escalation
SEVERITY_LEVELS = {
    0: {"name": "NONE", "color": "#808080", "escalation": False},
    1: {"name": "LOW", "color": "#4CAF50", "escalation": False},
    2: {"name": "MEDIUM", "color": "#FF9800", "escalation": False},
    3: {"name": "HIGH", "color": "#F44336", "escalation": True},
    4: {"name": "CRITICAL", "color": "#9C27B0", "escalation": True},
}

ALERT_TYPES = {
    "UNAUTHORIZED_CONSTRUCTION": {
        "severity": "HIGH",
        "description": "New construction detected in non-designated zone",
        "response_time_hours": 24,
        "authority": "Municipal Corporation / Town Planning",
    },
    "AGRICULTURAL_ENCROACHMENT": {
        "severity": "CRITICAL",
        "description": "Urban expansion into agricultural/protected land",
        "response_time_hours": 12,
        "authority": "District Collector / Revenue Department",
    },
    "INFORMAL_SETTLEMENT": {
        "severity": "MEDIUM",
        "description": "Informal settlement expansion detected",
        "response_time_hours": 48,
        "authority": "Slum Rehabilitation Authority",
    },
    "WETLAND_ENCROACHMENT": {
        "severity": "CRITICAL",
        "description": "Construction activity near notified wetlands",
        "response_time_hours": 12,
        "authority": "State Wetland Authority / NGT",
    },
    "FOREST_ENCROACHMENT": {
        "severity": "CRITICAL",
        "description": "Urban expansion into reserved/protected forest",
        "response_time_hours": 6,
        "authority": "State Forest Department / MoEFCC",
    },
    "CRZ_VIOLATION": {
        "severity": "CRITICAL",
        "description": "Construction in Coastal Regulation Zone",
        "response_time_hours": 12,
        "authority": "State CZMA / MoEFCC",
    },
    "LAKE_ENCROACHMENT": {
        "severity": "HIGH",
        "description": "Construction within lake/tank buffer zone",
        "response_time_hours": 24,
        "authority": "Lake Development Authority",
    },
    "FLOODPLAIN_ENCROACHMENT": {
        "severity": "CRITICAL",
        "description": "Construction on river floodplain",
        "response_time_hours": 12,
        "authority": "National Green Tribunal",
    },
    "RAPID_DENSIFICATION": {
        "severity": "LOW",
        "description": "Unusual rate of construction in urban zone",
        "response_time_hours": 72,
        "authority": "Town Planning Department",
    },
}


# ═══════════════════════════════════════════════════════
#  Change Detection Stream Dataset
# ═══════════════════════════════════════════════════════

class RealTimeStreamDataset(Dataset):
    """
    Simulates a continuous stream of bi-temporal satellite observations
    over Indian metropolitan areas. Each sample represents a location
    monitored over consecutive Sentinel-2 passes.
    """

    def __init__(self, num_observations, change_probability=0.15, seed=SEED):
        super().__init__()
        self.num_observations = num_observations
        rng = np.random.RandomState(seed)

        self.observations = []
        for i in range(num_observations):
            city_idx = rng.randint(0, len(CITIES))
            city = CITIES[city_idx]

            # Sentinel-2 revisit: 5-day repeat cycle
            gap_days = rng.choice([5, 10, 15])

            # Determine if change occurred
            has_change = rng.random() < change_probability

            # Determine zone type for this observation
            city_zones = CITY_PROTECTED_ZONES.get(city, {})
            zone_names = list(city_zones.keys())
            in_protected = rng.random() < 0.25 and len(zone_names) > 0

            if in_protected:
                zone_name = rng.choice(zone_names)
                zone_info = city_zones[zone_name]
                zone_type = zone_info["type"]
                reg_zone = REGULATORY_ZONES[zone_type]
            else:
                zone_name = None
                zone_type = None
                reg_zone = None

            if has_change:
                # Determine alert type based on zone
                if in_protected and zone_type:
                    # Map zone types to alert types
                    zone_to_alert = {
                        "CRZ_I": "CRZ_VIOLATION",
                        "CRZ_II": "CRZ_VIOLATION",
                        "CRZ_III": "CRZ_VIOLATION",
                        "FOREST_RESERVE": "FOREST_ENCROACHMENT",
                        "PROTECTED_FOREST": "FOREST_ENCROACHMENT",
                        "WETLAND": "WETLAND_ENCROACHMENT",
                        "LAKE_BUFFER": "LAKE_ENCROACHMENT",
                        "RIVER_FLOODPLAIN": "FLOODPLAIN_ENCROACHMENT",
                        "GREEN_BELT": "AGRICULTURAL_ENCROACHMENT",
                        "WESTERN_GHATS_ESA": "FOREST_ENCROACHMENT",
                    }
                    change_type = zone_to_alert.get(zone_type, "UNAUTHORIZED_CONSTRUCTION")
                else:
                    change_type = rng.choice([
                        "UNAUTHORIZED_CONSTRUCTION",
                        "INFORMAL_SETTLEMENT",
                        "RAPID_DENSIFICATION",
                    ])

                t1_label = rng.choice([1, 2])  # non-urban or transition
                t2_label = rng.choice([0, 2])   # urban or transition
            else:
                change_type = None
                t1_label = rng.choice([0, 1])
                t2_label = t1_label

            self.observations.append({
                "city": city,
                "city_idx": city_idx,
                "gap_days": gap_days,
                "has_change": has_change,
                "change_type": change_type,
                "t1_label": t1_label,
                "t2_label": t2_label,
                "in_protected_zone": in_protected,
                "zone_name": zone_name,
                "zone_type": zone_type,
                "timestamp": datetime(2024, 1, 1) + timedelta(days=int(i * 0.5)),
                "lat": CITY_BOUNDS[city][1] + rng.uniform(0, CITY_BOUNDS[city][3] - CITY_BOUNDS[city][1]),
                "lon": CITY_BOUNDS[city][0] + rng.uniform(0, CITY_BOUNDS[city][2] - CITY_BOUNDS[city][0]),
            })

    def __len__(self):
        return self.num_observations

    def __getitem__(self, idx):
        obs = self.observations[idx]
        t1 = generate_patch(obs["t1_label"], patch_size=128, num_channels=NUM_CHANNELS)
        t2 = generate_patch(obs["t2_label"], patch_size=128, num_channels=NUM_CHANNELS)

        # Add temporal noise
        t2 += np.random.normal(0, 0.01, t2.shape).astype(np.float32)
        t2 = np.clip(t2, 0, 1)

        change_label = 1 if obs["has_change"] else 0

        # Severity based on regulatory zone
        severity = 0
        if obs["has_change"] and obs["change_type"]:
            sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            base_severity = sev_map.get(ALERT_TYPES[obs["change_type"]]["severity"], 1)
            # Escalate if in protected zone
            if obs["in_protected_zone"]:
                severity = min(base_severity + 1, 4)
            else:
                severity = base_severity

        return (torch.from_numpy(t1), torch.from_numpy(t2),
                torch.tensor(change_label, dtype=torch.long),
                torch.tensor(severity, dtype=torch.long))


# ═══════════════════════════════════════════════════════
#  Real-Time Change Detector
# ═══════════════════════════════════════════════════════

class RealTimeChangeDetector(nn.Module):
    """
    Lightweight change detection model optimized for low-latency inference.
    Dual output: binary change detection + severity classification.

    Architecture:
    - Shared EfficientNet-B0 encoder (FPN-based)
    - Feature difference + concatenation
    - Change head: binary change/no-change
    - Severity head: 5-level severity (NONE to CRITICAL)
    - Alert type head: classifies the type of encroachment
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True,
                 num_alert_types=len(ALERT_TYPES)):
        super().__init__()
        self.encoder = UrbanClassifier(backbone_name, pretrained)
        feat_dim = FPN_CHANNELS * 3

        # Difference-based change detection
        self.change_head = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),  # f1, f2, |f2-f1|
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # change / no-change
        )

        # Severity classifier
        self.severity_head = nn.Sequential(
            nn.Linear(feat_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 5),  # 0=none, 1=low, 2=medium, 3=high, 4=critical
        )

        # Alert type classifier
        self.alert_type_head = nn.Sequential(
            nn.Linear(feat_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_alert_types),
        )

    def forward(self, t1, t2):
        f1 = self.encoder.get_feature_vector(t1)
        f2 = self.encoder.get_feature_vector(t2)
        diff = torch.abs(f2 - f1)

        combined = torch.cat([f1, f2, diff], dim=1)
        change_logits = self.change_head(combined)
        severity_logits = self.severity_head(combined)
        alert_type_logits = self.alert_type_head(combined)

        return change_logits, severity_logits, alert_type_logits

    def detect(self, t1, t2):
        """Single-pass detection for real-time use."""
        self.eval()
        with torch.no_grad():
            change_logits, severity_logits, alert_type_logits = self.forward(t1, t2)
            change_pred = change_logits.argmax(1)
            severity_pred = severity_logits.argmax(1)
            change_confidence = F.softmax(change_logits, dim=1)[:, 1]
            alert_type_pred = alert_type_logits.argmax(1)
        return change_pred, severity_pred, change_confidence, alert_type_pred


# ═══════════════════════════════════════════════════════
#  Alert Engine with Regulatory Compliance
# ═══════════════════════════════════════════════════════

class AlertEngine:
    """
    Processes change detections and generates structured alerts
    with India-specific regulatory context and escalation logic.
    """

    SEVERITY_NAMES = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ALERT_TYPE_NAMES = list(ALERT_TYPES.keys())

    def __init__(self, output_dir=None):
        self.alerts = []
        self.escalated = []
        self.output_dir = output_dir or os.path.join(OUTPUT_DIR, "alerts")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_detection(self, city, location, change_detected, severity,
                         timestamp, confidence, zone_name=None, zone_type=None,
                         alert_type_idx=None, lat=None, lon=None):
        """Process a change detection and generate alert with regulatory context."""
        if not change_detected:
            return None

        severity_name = self.SEVERITY_NAMES[min(severity, 4)]

        # Get regulatory context
        reg_context = {}
        if zone_type and zone_type in REGULATORY_ZONES:
            reg = REGULATORY_ZONES[zone_type]
            reg_context = {
                "zone_type": zone_type,
                "zone_description": reg["description"],
                "buffer_m": reg["buffer_m"],
                "authority": reg["authority"],
                "penalty": reg.get("penalty", "As per applicable law"),
                "response_time_hours": reg["response_time_hours"],
            }

        # Determine alert type
        alert_type_name = None
        if alert_type_idx is not None and 0 <= alert_type_idx < len(self.ALERT_TYPE_NAMES):
            alert_type_name = self.ALERT_TYPE_NAMES[alert_type_idx]

        alert = {
            "id": f"ALERT-{len(self.alerts)+1:04d}",
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            "city": city,
            "location": location,
            "coordinates": {"lat": lat, "lon": lon} if lat and lon else None,
            "severity": severity_name,
            "confidence": f"{confidence:.3f}",
            "alert_type": alert_type_name,
            "in_protected_zone": zone_name is not None,
            "zone_name": zone_name,
            "regulatory_context": reg_context,
            "status": "ACTIVE",
            "requires_escalation": severity >= 3,
            "requires_site_inspection": severity >= 2,
        }

        self.alerts.append(alert)

        # Auto-escalate critical alerts
        if severity >= 4:
            self.escalated.append(alert)

        return alert

    def generate_report(self):
        """Generate comprehensive monitoring report."""
        if not self.alerts:
            return {"total_alerts": 0, "message": "No alerts generated."}

        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "escalated_count": len(self.escalated),
            "by_severity": {},
            "by_city": {},
            "by_alert_type": {},
            "by_zone_type": {},
            "protected_zone_violations": 0,
            "regulatory_actions_required": [],
        }

        # Severity breakdown
        for sev in self.SEVERITY_NAMES[1:]:
            count = sum(1 for a in self.alerts if a["severity"] == sev)
            report["by_severity"][sev] = count

        # City breakdown
        for city in CITIES:
            city_alerts = [a for a in self.alerts if a["city"] == city]
            if city_alerts:
                report["by_city"][city] = {
                    "total": len(city_alerts),
                    "critical": sum(1 for a in city_alerts if a["severity"] == "CRITICAL"),
                    "high": sum(1 for a in city_alerts if a["severity"] == "HIGH"),
                    "protected_zone": sum(1 for a in city_alerts if a["in_protected_zone"]),
                }

        # Alert type breakdown
        for alert_type in ALERT_TYPES:
            count = sum(1 for a in self.alerts if a.get("alert_type") == alert_type)
            if count > 0:
                report["by_alert_type"][alert_type] = count

        # Zone violations
        zone_violations = defaultdict(int)
        for a in self.alerts:
            if a.get("regulatory_context", {}).get("zone_type"):
                zone_violations[a["regulatory_context"]["zone_type"]] += 1
        report["by_zone_type"] = dict(zone_violations)

        # Protected zone violations
        report["protected_zone_violations"] = sum(
            1 for a in self.alerts if a["in_protected_zone"]
        )

        # Actions required
        for a in self.escalated:
            reg = a.get("regulatory_context", {})
            if reg:
                report["regulatory_actions_required"].append({
                    "alert_id": a["id"],
                    "city": a["city"],
                    "zone": a.get("zone_name", "Unknown"),
                    "authority": reg.get("authority", "Municipal Corporation"),
                    "response_deadline_hours": reg.get("response_time_hours", 24),
                    "penalty": reg.get("penalty", "As per applicable law"),
                })

        return report

    def save_alerts(self):
        """Save alerts and report to JSON files."""
        # Alerts
        alerts_path = os.path.join(self.output_dir, "alerts.json")
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2, default=str)

        # Report
        report = self.generate_report()
        report_path = os.path.join(self.output_dir, "alert_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Dashboard summary (for visualization)
        dashboard = self._generate_dashboard_data()
        dashboard_path = os.path.join(self.output_dir, "dashboard.json")
        with open(dashboard_path, "w") as f:
            json.dump(dashboard, f, indent=2, default=str)

        return alerts_path, report_path

    def _generate_dashboard_data(self):
        """Generate dashboard-ready data for visualization."""
        dashboard = {
            "summary": {
                "total": len(self.alerts),
                "critical": sum(1 for a in self.alerts if a["severity"] == "CRITICAL"),
                "high": sum(1 for a in self.alerts if a["severity"] == "HIGH"),
                "protected_violations": sum(1 for a in self.alerts if a["in_protected_zone"]),
            },
            "city_heatmap": {},
            "timeline": [],
            "top_violations": [],
        }

        # City heatmap data
        for city in CITIES:
            city_alerts = [a for a in self.alerts if a["city"] == city]
            dashboard["city_heatmap"][city] = {
                "count": len(city_alerts),
                "severity_score": sum(
                    {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(a["severity"], 0)
                    for a in city_alerts
                ),
            }

        # Timeline (alerts per day)
        by_date = defaultdict(int)
        for a in self.alerts:
            date = a["timestamp"][:10] if isinstance(a["timestamp"], str) else a["timestamp"]
            by_date[date] += 1
        dashboard["timeline"] = [{"date": d, "count": c} for d, c in sorted(by_date.items())]

        # Top violations
        dashboard["top_violations"] = [
            {"id": a["id"], "city": a["city"], "severity": a["severity"],
             "zone": a.get("zone_name", "N/A"), "type": a.get("alert_type", "Unknown")}
            for a in sorted(self.alerts,
                          key=lambda x: {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(x["severity"], 0),
                          reverse=True)[:10]
        ]

        return dashboard


# ═══════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════

def train_realtime_detector(device="cuda", epochs=10):
    """Train the real-time change detection model."""
    print(f"\n{'='*60}")
    print("  PILLAR V: Real-Time Monitoring & Encroachment Alert System")
    print(f"  Cities: {', '.join(CITIES)}")
    print(f"  Regulatory zones: {len(REGULATORY_ZONES)} types")
    print(f"  Alert types: {len(ALERT_TYPES)} categories")
    print(f"{'='*60}")

    n_train = int(2000 * TRAIN_RATIO)
    n_val = int(2000 * VAL_RATIO)
    n_test = 2000 - n_train - n_val

    train_ds = RealTimeStreamDataset(n_train, seed=SEED + 50)
    val_ds = RealTimeStreamDataset(n_val, change_probability=0.15, seed=SEED + 51)
    test_ds = RealTimeStreamDataset(n_test, change_probability=0.15, seed=SEED + 52)

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    model = RealTimeChangeDetector("efficientnet_b0", pretrained=True).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {num_params:.1f}M")

    change_criterion = nn.CrossEntropyLoss()
    # Upweight minority severity classes (NONE is dominant, MEDIUM/HIGH/CRITICAL are rare)
    severity_weights = torch.tensor([0.5, 2.0, 4.0, 3.0, 3.0], dtype=torch.float32).to(device)
    severity_criterion = nn.CrossEntropyLoss(weight=severity_weights)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_state = None
    history = {"train_loss": [], "val_acc": [], "val_severity_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for t1, t2, change_label, severity_label in train_loader:
            t1, t2 = t1.to(device), t2.to(device)
            change_label = change_label.to(device)
            severity_label = severity_label.to(device)

            change_logits, severity_logits, _ = model(t1, t2)
            loss = (change_criterion(change_logits, change_label) +
                    0.5 * severity_criterion(severity_logits, severity_label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * t1.size(0)
            correct += (change_logits.argmax(1) == change_label).sum().item()
            total += t1.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_sev_correct, val_sev_total = 0, 0
        with torch.no_grad():
            for t1, t2, change_label, severity_label in val_loader:
                t1, t2 = t1.to(device), t2.to(device)
                change_label = change_label.to(device)
                severity_label = severity_label.to(device)
                change_logits, severity_logits, _ = model(t1, t2)
                preds = change_logits.argmax(1)
                val_correct += (preds == change_label).sum().item()
                val_total += t1.size(0)
                # Severity accuracy (only for changed samples)
                changed_mask = change_label == 1
                if changed_mask.any():
                    sev_preds = severity_logits[changed_mask].argmax(1)
                    val_sev_correct += (sev_preds == severity_label[changed_mask]).sum().item()
                    val_sev_total += changed_mask.sum().item()

        val_acc = val_correct / val_total
        val_sev_acc = val_sev_correct / max(val_sev_total, 1)
        history["train_loss"].append(total_loss / total)
        history["val_acc"].append(val_acc)
        history["val_severity_acc"].append(val_sev_acc)

        print(f"  Epoch {epoch:2d}/{epochs} | Loss {total_loss/total:.4f} | "
              f"TrAcc {correct/total:.4f} | VaAcc {val_acc:.4f} | SevAcc {val_sev_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ── Test ──
    model.load_state_dict(best_state)
    model.eval()
    test_change_preds, test_change_labels = [], []
    test_sev_preds, test_sev_labels = [], []

    with torch.no_grad():
        for t1, t2, change_label, severity_label in test_loader:
            t1, t2 = t1.to(device), t2.to(device)
            change_logits, severity_logits, _ = model(t1, t2)
            test_change_preds.extend(change_logits.argmax(1).cpu().numpy())
            test_change_labels.extend(change_label.numpy())
            test_sev_preds.extend(severity_logits.argmax(1).cpu().numpy())
            test_sev_labels.extend(severity_label.numpy())

    change_acc = np.mean(np.array(test_change_preds) == np.array(test_change_labels))
    print(f"\n  {'='*50}")
    print(f"  TEST RESULTS")
    print(f"  {'='*50}")
    print(f"    Change Detection Accuracy: {change_acc:.4f}")

    # Per-severity accuracy
    for sev in range(5):
        mask = np.array(test_sev_labels) == sev
        if mask.any():
            sev_acc = np.mean(np.array(test_sev_preds)[mask] == sev)
            print(f"    Severity {SEVERITY_LEVELS[sev]['name']:8s} Accuracy: {sev_acc:.4f} (n={mask.sum()})")

    # ── Monitoring Simulation ──
    print(f"\n  Simulating Real-Time Monitoring Pipeline...")
    alert_engine = simulate_monitoring(model, device)

    # ── Save ──
    save_path = os.path.join(MODEL_DIR, "pillar5_realtime.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"  Model saved to {save_path}")

    return model, history, alert_engine, change_acc


def simulate_monitoring(model, device, num_observations=300):
    """
    Simulate a real-time monitoring session over Indian cities.
    Generates alerts with full regulatory context.
    """
    model.eval()
    alert_engine = AlertEngine()
    stream = RealTimeStreamDataset(num_observations, change_probability=0.20, seed=SEED + 100)

    detections = 0
    for i in range(len(stream)):
        t1, t2, true_change, true_severity = stream[i]
        obs = stream.observations[i]

        t1_batch = t1.unsqueeze(0).to(device)
        t2_batch = t2.unsqueeze(0).to(device)

        with torch.no_grad():
            change_logits, severity_logits, alert_type_logits = model(t1_batch, t2_batch)
            change_pred = change_logits.argmax(1).item()
            severity_pred = severity_logits.argmax(1).item()
            confidence = F.softmax(change_logits, dim=1)[0, 1].item()
            alert_type_pred = alert_type_logits.argmax(1).item()

        if change_pred == 1:
            detections += 1
            alert = alert_engine.process_detection(
                city=obs["city"],
                location=f"Grid {i:04d}",
                change_detected=True,
                severity=severity_pred,
                timestamp=obs["timestamp"],
                confidence=confidence,
                zone_name=obs.get("zone_name"),
                zone_type=obs.get("zone_type"),
                alert_type_idx=alert_type_pred,
                lat=obs.get("lat"),
                lon=obs.get("lon"),
            )

    # Save and report
    alerts_path, report_path = alert_engine.save_alerts()
    report = alert_engine.generate_report()

    print(f"\n  Monitoring Simulation Results:")
    print(f"    Observations processed: {num_observations}")
    print(f"    Changes detected: {detections}")
    print(f"    Total alerts: {report['total_alerts']}")
    print(f"    Escalated: {report.get('escalated_count', 0)}")
    print(f"    Protected zone violations: {report.get('protected_zone_violations', 0)}")

    print(f"\n    Alerts by severity:")
    for sev, count in report.get("by_severity", {}).items():
        print(f"      {sev:10s}: {count}")

    print(f"\n    Alerts by city:")
    for city, data in report.get("by_city", {}).items():
        print(f"      {city:15s}: {data['total']} total, {data['critical']} critical, "
              f"{data['protected_zone']} in protected zones")

    if report.get("by_zone_type"):
        print(f"\n    Violations by regulatory zone:")
        for zone, count in report["by_zone_type"].items():
            print(f"      {zone:25s}: {count}")

    if report.get("regulatory_actions_required"):
        print(f"\n    Regulatory Actions Required: {len(report['regulatory_actions_required'])}")
        for action in report["regulatory_actions_required"][:5]:
            print(f"      {action['alert_id']}: {action['city']} - {action['zone']} -> {action['authority']}")

    print(f"\n    Alerts saved to: {alerts_path}")
    print(f"    Report saved to: {report_path}")

    return alert_engine


# ═══════════════════════════════════════════════════════
#  Latency Benchmark
# ═══════════════════════════════════════════════════════

def benchmark_latency(model, device, num_runs=100):
    """Measure inference latency for real-time feasibility assessment."""
    model.eval()
    t1 = torch.randn(1, NUM_CHANNELS, 128, 128).to(device)
    t2 = torch.randn(1, NUM_CHANNELS, 128, 128).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(t1, t2)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(t1, t2)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    p50_ms = np.percentile(times, 50) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    throughput = 1000 / avg_ms

    # Compute time for full city coverage
    # Typical: ~10,000 patches per city at 128x128 from Sentinel-2
    patches_per_city = 10000
    city_coverage_sec = patches_per_city / throughput

    print(f"\n  Inference Latency Benchmark:")
    print(f"    Mean:       {avg_ms:.2f} ms")
    print(f"    P50:        {p50_ms:.2f} ms")
    print(f"    P95:        {p95_ms:.2f} ms")
    print(f"    P99:        {p99_ms:.2f} ms")
    print(f"    Throughput: {throughput:.0f} patches/sec")
    print(f"\n  Full City Coverage Estimate:")
    print(f"    Patches per city:  ~{patches_per_city:,}")
    print(f"    Time per city:     {city_coverage_sec:.1f} sec ({city_coverage_sec/60:.1f} min)")
    print(f"    All 7 cities:      {7*city_coverage_sec/60:.1f} min")
    print(f"    Within 24h target: {'YES' if 7*city_coverage_sec < 86400 else 'NO'}")

    return {"mean_ms": avg_ms, "p50_ms": p50_ms, "p95_ms": p95_ms,
            "p99_ms": p99_ms, "throughput": throughput,
            "city_coverage_sec": city_coverage_sec}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history, alert_engine, acc = train_realtime_detector(device, epochs=5)
    benchmark_latency(model, device)
