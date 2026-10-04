import os
import math
import hashlib
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium.plugins import Fullscreen, HeatMap, LocateControl, MiniMap
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore")

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="NER Smart Logistics • Live GIS Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size:2.05rem;font-weight:800;color:#17324d;margin-bottom:.15rem;}
    .sub-header {font-size:1rem;color:#52606d;margin-bottom:1rem;}
    .small-note {font-size:.78rem;color:#68737d;}
    .live-pill {display:inline-block;padding:.25rem .65rem;border-radius:999px;background:#e6ffed;color:#137333;font-weight:700;font-size:.78rem;}
    .demo-pill {display:inline-block;padding:.25rem .65rem;border-radius:999px;background:#fff4ce;color:#8a5a00;font-weight:700;font-size:.78rem;}
    .legend-box {padding:.7rem .9rem;border-radius:.5rem;background:#f8fafc;border:1px solid #e2e8f0;font-size:.85rem;}
    .alert-high {background:#fff0f0;border-left:5px solid #d32f2f;padding:.65rem .85rem;margin:.35rem 0;border-radius:5px;}
    .alert-medium {background:#fff8e8;border-left:5px solid #ed8b00;padding:.65rem .85rem;margin:.35rem 0;border-radius:5px;}
    .alert-low {background:#effaf2;border-left:5px solid #2e7d32;padding:.65rem .85rem;margin:.35rem 0;border-radius:5px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NER reference geography
# ============================================================
NER_STATES = {
    "Assam": {"lat": 26.2006, "lon": 92.9376, "capital": "Dispur"},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278, "capital": "Itanagar"},
    "Manipur": {"lat": 24.6637, "lon": 93.9063, "capital": "Imphal"},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662, "capital": "Shillong"},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376, "capital": "Aizawl"},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624, "capital": "Kohima"},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122, "capital": "Gangtok"},
    "Tripura": {"lat": 23.9408, "lon": 91.9882, "capital": "Agartala"},
}

CITY_COORDS = {
    "Guwahati": (26.1445, 91.7362),
    "Tezpur": (26.6528, 92.7926),
    "North Lakhimpur": (27.2352, 94.1010),
    "Itanagar": (27.0844, 93.6053),
    "Ziro": (27.5449, 93.8196),
    "Dimapur": (25.9090, 93.7279),
    "Kohima": (25.6751, 94.1086),
    "Imphal": (24.8170, 93.9368),
    "Moreh": (24.2500, 94.3000),
    "Shillong": (25.5788, 91.8933),
    "Aizawl": (23.7271, 92.7176),
    "Silchar": (24.8333, 92.7789),
    "Gangtok": (27.3389, 88.6065),
    "Siliguri": (26.7271, 88.3953),
    "Agartala": (23.8315, 91.2868),
    "Sabroom": (23.0010, 91.7200),
    "Jorhat": (26.7509, 94.2037),
    "Sivasagar": (26.9825, 94.6425),
    "Bomdila": (27.2646, 92.4143),
    "Tawang": (27.5861, 91.8594),
    "Dibrugarh": (27.4728, 94.9120),
    "Tinsukia": (27.4922, 95.3468),
}

# Corridor paths are reference geometry. OpenStreetMap is the surrounding map layer.
ROAD_SEGMENTS = [
    {"id": "NH27-AS-01", "name": "Guwahati – Tezpur", "state": "Assam", "type": "NH", "base_time": 3.5, "endpoints": ("Guwahati", "Tezpur"), "slope": 12, "elevation": 120, "river_km": 3.0, "road_condition": .82, "history": 2},
    {"id": "NH15-AS-02", "name": "Tezpur – North Lakhimpur", "state": "Assam", "type": "NH", "base_time": 4.0, "endpoints": ("Tezpur", "North Lakhimpur"), "slope": 10, "elevation": 105, "river_km": 1.8, "road_condition": .78, "history": 2},
    {"id": "NH13-AR-01", "name": "Itanagar – Ziro", "state": "Arunachal Pradesh", "type": "NH", "base_time": 5.5, "endpoints": ("Itanagar", "Ziro"), "slope": 34, "elevation": 1450, "river_km": 6.5, "road_condition": .66, "history": 4},
    {"id": "NH29-NL-01", "name": "Dimapur – Kohima", "state": "Nagaland", "type": "NH", "base_time": 3.0, "endpoints": ("Dimapur", "Kohima"), "slope": 29, "elevation": 780, "river_km": 10.0, "road_condition": .70, "history": 3},
    {"id": "NH2-MN-01", "name": "Imphal – Moreh", "state": "Manipur", "type": "NH", "base_time": 4.5, "endpoints": ("Imphal", "Moreh"), "slope": 23, "elevation": 980, "river_km": 11.0, "road_condition": .74, "history": 3},
    {"id": "NH6-ML-01", "name": "Shillong – Guwahati", "state": "Meghalaya", "type": "NH", "base_time": 3.2, "endpoints": ("Shillong", "Guwahati"), "slope": 27, "elevation": 1000, "river_km": 8.0, "road_condition": .72, "history": 3},
    {"id": "NH54-MZ-01", "name": "Aizawl – Silchar", "state": "Mizoram", "type": "NH", "base_time": 6.0, "endpoints": ("Aizawl", "Silchar"), "slope": 36, "elevation": 1050, "river_km": 12.0, "road_condition": .61, "history": 4},
    {"id": "NH10-SK-01", "name": "Gangtok – Siliguri", "state": "Sikkim", "type": "NH", "base_time": 4.8, "endpoints": ("Gangtok", "Siliguri"), "slope": 38, "elevation": 1800, "river_km": 7.0, "road_condition": .64, "history": 4},
    {"id": "NH8-TR-01", "name": "Agartala – Sabroom", "state": "Tripura", "type": "NH", "base_time": 3.8, "endpoints": ("Agartala", "Sabroom"), "slope": 14, "elevation": 70, "river_km": 4.5, "road_condition": .79, "history": 2},
    {"id": "SH-AS-03", "name": "Jorhat – Sivasagar", "state": "Assam", "type": "SH", "base_time": 1.5, "endpoints": ("Jorhat", "Sivasagar"), "slope": 7, "elevation": 95, "river_km": 4.0, "road_condition": .70, "history": 2},
    {"id": "MDR-AR-02", "name": "Bomdila – Tawang", "state": "Arunachal Pradesh", "type": "MDR", "base_time": 7.0, "endpoints": ("Bomdila", "Tawang"), "slope": 42, "elevation": 2900, "river_km": 15.0, "road_condition": .57, "history": 5},
    {"id": "NH37-AS-04", "name": "Dibrugarh – Tinsukia", "state": "Assam", "type": "NH", "base_time": 2.2, "endpoints": ("Dibrugarh", "Tinsukia"), "slope": 6, "elevation": 100, "river_km": 2.5, "road_condition": .80, "history": 2},
]

# Reference river geometry for an immediately understandable NER map.
RIVERS = [
    {"name": "Brahmaputra", "coords": [(27.95, 93.90), (27.65, 94.70), (27.50, 95.40), (27.25, 94.85), (26.95, 94.10), (26.55, 93.40), (26.15, 92.40), (25.85, 91.20), (25.55, 90.20)]},
    {"name": "Barak", "coords": [(24.85, 92.80), (24.55, 92.70), (24.20, 92.95), (23.95, 93.15), (23.70, 93.35)]},
    {"name": "Teesta", "coords": [(27.90, 88.75), (27.60, 88.55), (27.35, 88.60), (27.10, 88.45), (26.85, 88.50)]},
    {"name": "Subansiri", "coords": [(29.00, 94.10), (28.55, 94.15), (28.05, 94.05), (27.70, 94.00), (27.35, 93.95)]},
]

FACILITIES = [
    {"name": "Guwahati Logistics Hub", "lat": 26.1445, "lon": 91.7362, "kind": "Logistics Hub"},
    {"name": "Tezpur Airport / Supply Node", "lat": 26.7097, "lon": 92.7847, "kind": "Airport"},
    {"name": "Dibrugarh Airport / Supply Node", "lat": 27.4839, "lon": 95.0169, "kind": "Airport"},
    {"name": "Imphal Airport / Supply Node", "lat": 24.7599, "lon": 93.8967, "kind": "Airport"},
    {"name": "Lengpui Airport / Supply Node", "lat": 23.8400, "lon": 92.6197, "kind": "Airport"},
    {"name": "Agartala Supply Node", "lat": 23.8869, "lon": 91.2409, "kind": "Logistics Hub"},
    {"name": "Gangtok Supply Node", "lat": 27.3389, "lon": 88.6065, "kind": "Logistics Hub"},
]

# Demo GPS positions. In LIVE mode these can be replaced through an external REST endpoint.
DEMO_VEHICLES = [
    {"id": "MED-001", "type": "Medicine Van", "origin": "Guwahati", "dest": "Itanagar", "lat": 26.80, "lon": 93.50, "status": "In Transit", "eta_hrs": 4.2, "commodity": "Essential Medicines"},
    {"id": "FOOD-012", "type": "Supply Truck", "origin": "Siliguri", "dest": "Gangtok", "lat": 27.00, "lon": 88.40, "status": "In Transit", "eta_hrs": 2.8, "commodity": "Food Supplies"},
    {"id": "AGR-007", "type": "Agri Truck", "origin": "Jorhat", "dest": "Dibrugarh", "lat": 27.20, "lon": 94.90, "status": "Delayed", "eta_hrs": 3.5, "commodity": "Agricultural Produce"},
    {"id": "CONST-003", "type": "Construction", "origin": "Dimapur", "dest": "Kohima", "lat": 25.85, "lon": 94.05, "status": "In Transit", "eta_hrs": 1.5, "commodity": "Construction Materials"},
    {"id": "MED-015", "type": "Medicine Van", "origin": "Agartala", "dest": "Aizawl", "lat": 23.80, "lon": 92.40, "status": "High Risk Zone", "eta_hrs": 5.0, "commodity": "Vaccines & Medicines"},
]

FIELD_REPORTS_DEFAULT = [
    {"time": "10:15", "location": "NH13 near Ziro", "type": "Landslide debris", "severity": "High", "reporter": "Field Officer - AR", "lat": 27.55, "lon": 93.85},
    {"time": "08:40", "location": "NH29 Dimapur-Kohima", "type": "Heavy rain / slippery", "severity": "Medium", "reporter": "PWD Nagaland", "lat": 25.90, "lon": 94.10},
]

# ============================================================
# Utility functions
# ============================================================
def stable_number(key: str, low: float = 0.0, high: float = 1.0) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    n = int(digest[:12], 16) / float(16 ** 12 - 1)
    return low + (high - low) * n


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    r = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def corridor_points(seg: Dict) -> List[Tuple[float, float]]:
    start = CITY_COORDS[seg["endpoints"][0]]
    end = CITY_COORDS[seg["endpoints"][1]]
    # Add a deterministic mid-point bend so corridors read as routes rather than dots.
    mid_lat = (start[0] + end[0]) / 2 + stable_number(seg["id"] + "lat", -0.18, 0.18)
    mid_lon = (start[1] + end[1]) / 2 + stable_number(seg["id"] + "lon", -0.18, 0.18)
    return [start, (mid_lat, mid_lon), end]


def risk_color(level: str) -> str:
    return {"High": "#d32f2f", "Medium": "#f59e0b", "Low": "#2e7d32"}.get(level, "#64748b")


def traffic_color(level: str) -> str:
    return {"Severe": "#7f1d1d", "Heavy": "#dc2626", "Moderate": "#f59e0b", "Free": "#16a34a"}.get(level, "#64748b")


def classify_risk(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def weather_for_point(lat: float, lon: float) -> Dict:
    """Fetch current weather without an API key. Falls back gracefully to demo values."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,rain,wind_speed_10m",
        "timezone": "auto",
    }
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=7)
        response.raise_for_status()
        current = response.json().get("current", {})
        return {
            "temp": float(current.get("temperature_2m", 0)),
            "rain_mm": float(current.get("rain", current.get("precipitation", 0))),
            "wind_kmh": float(current.get("wind_speed_10m", 0)),
            "source": "Open-Meteo current weather",
            "live": True,
        }
    except Exception:
        return {
            "temp": 24 + stable_number(f"{lat:.3f},{lon:.3f}temp", -4, 6),
            "rain_mm": stable_number(f"{lat:.3f},{lon:.3f}rain", 0, 25),
            "wind_kmh": stable_number(f"{lat:.3f},{lon:.3f}wind", 3, 22),
            "source": "Demo fallback",
            "live": False,
        }


@st.cache_data(ttl=300, show_spinner=False)
def get_weather_snapshot(selected_states: Tuple[str, ...]) -> Dict[str, Dict]:
    out = {}
    for state in selected_states:
        ref = NER_STATES[state]
        out[state] = weather_for_point(ref["lat"], ref["lon"])
    return out


def traffic_for_segment(seg: Dict, live_mode: bool = False) -> Dict:
    """Optional TomTom traffic flow; otherwise transparent estimated traffic state."""
    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if live_mode and api_key:
        lat = (CITY_COORDS[seg["endpoints"][0]][0] + CITY_COORDS[seg["endpoints"][1]][0]) / 2
        lon = (CITY_COORDS[seg["endpoints"][0]][1] + CITY_COORDS[seg["endpoints"][1]][1]) / 2
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        try:
            r = requests.get(url, params={"point": f"{lat},{lon}", "key": api_key}, timeout=7)
            r.raise_for_status()
            data = r.json().get("flowSegmentData", {})
            current = float(data.get("currentSpeed", 0) or 0)
            free = float(data.get("freeFlowSpeed", current or 1) or 1)
            ratio = current / free if free else 1
            level = "Free" if ratio >= .85 else ("Moderate" if ratio >= .65 else ("Heavy" if ratio >= .45 else "Severe"))
            return {"level": level, "ratio": ratio, "speed": current, "free": free, "delay": max(0, 1 - ratio), "source": "TomTom live traffic", "live": True}
        except Exception:
            pass

    # Demo/estimated traffic: deterministic with time-of-day and corridor context.
    hour = datetime.now().hour
    rush = 0.18 if hour in (8, 9, 10, 17, 18, 19) else 0.0
    base = stable_number(seg["id"] + datetime.now().strftime("%Y-%m-%d"), 0.10, 0.78)
    ratio = max(.20, min(1.0, 1 - (base + rush)))
    level = "Free" if ratio >= .85 else ("Moderate" if ratio >= .65 else ("Heavy" if ratio >= .45 else "Severe"))
    return {"level": level, "ratio": ratio, "speed": None, "free": None, "delay": 1 - ratio, "source": "Estimated traffic", "live": False}


class GeoRiskPredictor:
    """Geospatial Risk Predictor that uses GradientBoostingClassifier if available,
    or a calibrated pure-NumPy hazard model when C-extension DLLs / OpenMP are blocked
    by Windows Application Control / AppLocker policies."""
    def __init__(self, backend_model=None):
        self.backend_model = backend_model

    def predict_proba(self, X):
        if self.backend_model is not None:
            try:
                return self.backend_model.predict_proba(X)
            except Exception:
                pass
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Features: [rain24, slope, road_condition, traffic_delay, soil_proxy, elevation, river_km, history]
        logit = (
            0.075 * X[:, 0]
            + 0.065 * X[:, 1]
            - 2.2 * X[:, 2]
            + 0.12 * X[:, 3]
            + 1.15 * X[:, 4]
            - 0.16 * X[:, 6]
            + 0.30 * X[:, 7]
            - 0.00023 * X[:, 5]
            - 3.5
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        prob = np.clip(prob, 0.02, 0.98)
        return np.column_stack([1.0 - prob, prob])


def train_geo_risk_model():
    if not HAS_SKLEARN:
        return GeoRiskPredictor()
    try:
        np.random.seed(42)
        n = 900
        X = np.column_stack([
            np.random.uniform(0, 70, n),     # rainfall mm
            np.random.uniform(3, 48, n),     # slope
            np.random.uniform(.45, 1.0, n),  # road condition
            np.random.uniform(0, 15, n),     # traffic delay index
            np.random.uniform(.2, 1.0, n),   # soil saturation proxy
            np.random.uniform(20, 3200, n),  # elevation
            np.random.uniform(.2, 20, n),    # river distance km
            np.random.poisson(2.0, n),       # historical incidents
        ])
        logit = (
            0.075 * X[:, 0]
            + 0.065 * X[:, 1]
            - 2.2 * X[:, 2]
            + 0.12 * X[:, 3]
            + 1.15 * X[:, 4]
            + 0.16 * X[:, 6] * -1
            + 0.30 * X[:, 7]
            - 0.00023 * X[:, 5]
            - 3.5
        )
        prob = 1 / (1 + np.exp(-logit))
        y = (np.random.rand(n) < prob).astype(int)
        model = GradientBoostingClassifier(n_estimators=110, max_depth=4, random_state=42)
        model.fit(X, y)
        return GeoRiskPredictor(backend_model=model)
    except Exception:
        return GeoRiskPredictor()


risk_model = train_geo_risk_model()


def compute_segment_status(seg: Dict, weather: Dict, traffic: Dict, rain_boost: float) -> Dict:
    rain24 = max(0.0, weather["rain_mm"] * 6 + rain_boost)
    traffic_delay = traffic["delay"] * 10
    soil_proxy = min(.98, .30 + rain24 / 90)
    features = np.array([[rain24, seg["slope"], seg["road_condition"], traffic_delay, soil_proxy, seg["elevation"], seg["river_km"], seg["history"]]])
    model_prob = float(risk_model.predict_proba(features)[0][1])

    landslide = min(0.98, 0.38 * model_prob + 0.32 * min(seg["slope"] / 45, 1) + 0.30 * min(rain24 / 70, 1))
    flood = min(0.98, 0.48 * min(rain24 / 60, 1) + 0.27 * (1 - min(seg["river_km"] / 15, 1)) + 0.25 * (1 - min(seg["elevation"] / 2200, 1)))
    composite = min(0.98, 0.72 * model_prob + 0.18 * flood + 0.10 * traffic["delay"])
    level = classify_risk(composite)
    disaster_type = "Landslide" if landslide >= flood and landslide >= .55 else ("Flood / Waterlogging" if flood > landslide and flood >= .55 else "Transport disruption")

    return {
        "rain24": rain24,
        "model_prob": model_prob,
        "landslide": landslide,
        "flood": flood,
        "score": composite,
        "level": level,
        "disaster_type": disaster_type,
        "traffic": traffic,
        "factors": {
            "Rainfall estimate (24h)": f"{rain24:.1f} mm",
            "Terrain slope": f"{seg['slope']}°",
            "Road condition": f"{seg['road_condition']:.0%}",
            "River proximity": f"{seg['river_km']:.1f} km",
            "Elevation": f"{seg['elevation']} m",
            "History index": str(seg["history"]),
        },
    }


def get_live_vehicles(live_mode: bool) -> Tuple[List[Dict], bool]:
    url = os.getenv("GPS_API_URL", "").strip()
    if live_mode and url:
        try:
            response = requests.get(url, timeout=7)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = payload.get("vehicles", [])
            if isinstance(payload, list) and payload:
                return payload, True
        except Exception:
            pass
    return DEMO_VEHICLES, False


def make_risk_grid(weather_by_state: Dict[str, Dict], selected_states: List[str], rain_boost: float) -> List[Tuple[float, float, float, str]]:
    points = []
    offsets = [(-.50, -.65), (-.40, .15), (-.15, -.35), (.00, .00), (.25, .25), (.40, -.20), (.55, .55)]
    for state in selected_states:
        ref = NER_STATES[state]
        weather = weather_by_state[state]
        for idx, (dlat, dlon) in enumerate(offsets):
            lat = ref["lat"] + dlat
            lon = ref["lon"] + dlon
            slope = stable_number(f"grid-slope-{state}-{idx}", 5, 46)
            elev = stable_number(f"grid-elev-{state}-{idx}", 60, 2800)
            river_near = stable_number(f"grid-river-{state}-{idx}", 1.5, 18)
            rain24 = max(0, weather["rain_mm"] * 6 + rain_boost)
            score = min(0.98, 0.48 * min(rain24 / 70, 1) + 0.26 * (slope / 46) + 0.16 * (1 - min(river_near / 18, 1)) + 0.10 * (1 - min(elev / 3000, 1)))
            level = classify_risk(score)
            points.append((lat, lon, score, level))
    return points


def build_map(selected_states, risk_status, vehicles, field_reports, toggles, route_coords=None):
    m = folium.Map(
        location=[26.0, 93.0],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Light map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite context",
        overlay=False,
        control=True,
    ).add_to(m)

    Fullscreen(position="topleft", title="Full screen", title_cancel="Exit full screen").add_to(m)
    LocateControl(auto_start=False).add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)

    fg_states = folium.FeatureGroup(name="State hubs", show=True)
    fg_roads = folium.FeatureGroup(name="Roads & critical corridors", show=True)
    fg_rivers = folium.FeatureGroup(name="Rivers & water systems", show=toggles["rivers"])
    fg_facilities = folium.FeatureGroup(name="Logistics / service nodes", show=toggles["facilities"])
    fg_traffic = folium.FeatureGroup(name="Traffic", show=toggles["traffic"])
    fg_vehicles = folium.FeatureGroup(name="Live vehicle tracking", show=toggles["vehicles"])
    fg_risk = folium.FeatureGroup(name="Natural-disaster risk zones", show=toggles["risk_zones"])
    fg_reports = folium.FeatureGroup(name="Field incident reports", show=toggles["reports"])
    fg_route = folium.FeatureGroup(name="Selected route", show=route_coords is not None)

    for state in selected_states:
        info = NER_STATES[state]
        folium.Marker(
            [info["lat"], info["lon"]],
            tooltip=f"{state} • {info['capital']}",
            popup=f"<b>{state}</b><br>Capital / hub: {info['capital']}",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(fg_states)

    for seg in ROAD_SEGMENTS:
        if seg["state"] not in selected_states:
            continue
        stt = risk_status[seg["id"]]
        color = risk_color(stt["level"])
        points = corridor_points(seg)
        popup = (
            f"<b>{seg['name']}</b><br>"
            f"Corridor: {seg['id']}<br>"
            f"Risk: <b>{stt['level']}</b> ({stt['score']:.0%})<br>"
            f"Hazard: {stt['disaster_type']}<br>"
            f"Traffic: {stt['traffic']['level']}<br>"
            f"24h rain estimate: {stt['rain24']:.1f} mm<br>"
            f"Road condition index: {seg['road_condition']:.0%}"
        )
        folium.PolyLine(
            points,
            color=color,
            weight=9 if seg["type"] == "NH" else 6,
            opacity=.90,
            tooltip=f"{seg['name']} • {stt['level']} risk",
            popup=popup,
        ).add_to(fg_roads)
        # Bright center line makes the risk corridor easy to read over any basemap.
        folium.PolyLine(points, color="#ffffff", weight=2, opacity=.75).add_to(fg_roads)

    if toggles["rivers"]:
        for river in RIVERS:
            folium.PolyLine(
                river["coords"],
                color="#1976d2",
                weight=5,
                opacity=.75,
                tooltip=f"River: {river['name']}",
            ).add_to(fg_rivers)

    for facility in FACILITIES:
        folium.Marker(
            [facility["lat"], facility["lon"]],
            tooltip=f"{facility['name']} • {facility['kind']}",
            popup=f"<b>{facility['name']}</b><br>{facility['kind']}",
            icon=folium.Icon(color="cadetblue", icon="plus", prefix="fa"),
        ).add_to(fg_facilities)

    for seg in ROAD_SEGMENTS:
        if seg["state"] not in selected_states:
            continue
        t = risk_status[seg["id"]]["traffic"]
        if t["level"] in ("Heavy", "Severe", "Moderate"):
            points = corridor_points(seg)
            p = points[1]
            folium.CircleMarker(
                p,
                radius=9 if t["level"] == "Severe" else 7,
                color=traffic_color(t["level"]),
                fill=True,
                fill_color=traffic_color(t["level"]),
                fill_opacity=.90,
                tooltip=f"Traffic: {t['level']} • {seg['name']}",
                popup=(f"<b>{seg['name']}</b><br>Traffic: {t['level']}<br>"
                       f"Source: {t['source']}")
            ).add_to(fg_traffic)

    for v in vehicles:
        color = "purple" if v.get("status") == "In Transit" else "darkred"
        folium.Marker(
            [v["lat"], v["lon"]],
            tooltip=f"{v['id']} • {v.get('commodity', 'Cargo')}",
            popup=(f"<b>{v['id']}</b><br>{v.get('commodity', '')}<br>"
                   f"Status: <b>{v.get('status', 'Unknown')}</b><br>"
                   f"Route: {v.get('origin', '?')} → {v.get('dest', '?')}<br>"
                   f"ETA: {v.get('eta_hrs', '?')} hrs"),
            icon=folium.Icon(color=color, icon="truck", prefix="fa"),
        ).add_to(fg_vehicles)

    for lat, lon, score, level in make_risk_grid(st.session_state.weather_snapshot, selected_states, st.session_state.rain_boost):
        if level == "Low":
            continue
        c = risk_color(level)
        folium.Circle(
            [lat, lon],
            radius=18000 if level == "High" else 12000,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=.15 if level == "High" else .10,
            weight=2,
            tooltip=f"Predicted {level} natural-hazard zone • {score:.0%}",
            popup=(f"<b>Geospatial hazard zone</b><br>Predicted level: {level}<br>"
                   f"Composite hazard score: {score:.0%}<br>"
                   f"Uses rainfall, terrain slope, elevation and river proximity proxies."),
        ).add_to(fg_risk)

    for rep in field_reports:
        if rep.get("lat") is None or rep.get("lon") is None:
            continue
        color = "red" if rep["severity"] == "High" else "orange" if rep["severity"] == "Medium" else "green"
        folium.Marker(
            [rep["lat"], rep["lon"]],
            tooltip=f"Field report • {rep['severity']}",
            popup=(f"<b>{rep['type']}</b><br>{rep['location']}<br>Severity: {rep['severity']}<br>"
                   f"Reporter: {rep['reporter']}<br>Time: {rep['time']}"),
            icon=folium.Icon(color=color, icon="exclamation-triangle", prefix="fa"),
        ).add_to(fg_reports)

    if route_coords:
        folium.PolyLine(route_coords, color="#6a1b9a", weight=10, opacity=.8, tooltip="Selected route").add_to(fg_route)
        folium.PolyLine(route_coords, color="#ffffff", weight=3, opacity=.9).add_to(fg_route)

    for fg in [fg_states, fg_roads, fg_rivers, fg_facilities, fg_traffic, fg_vehicles, fg_risk, fg_reports, fg_route]:
        fg.add_to(m)

    # Risk heat layer uses the same geospatial risk grid but is optional for readability.
    if toggles["heatmap"]:
        heat_data = [[lat, lon, score] for lat, lon, score, level in make_risk_grid(st.session_state.weather_snapshot, selected_states, st.session_state.rain_boost)]
        HeatMap(heat_data, radius=30, blur=22, min_opacity=.18, name="Risk heat surface").add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend_html = """
    <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #bbb; border-radius: 6px; font-size: 12px; box-shadow: 0 1px 5px rgba(0,0,0,.2);">
    <b>Map Legend</b><br>
    <span style="color:#d32f2f">━━</span> High risk corridor<br>
    <span style="color:#f59e0b">━━</span> Medium risk corridor<br>
    <span style="color:#2e7d32">━━</span> Low risk corridor<br>
    <span style="color:#1976d2">━━</span> River / water system<br>
    🚛 Vehicle &nbsp; ⚠ Incident &nbsp; ＋ Service node
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def try_osrm_route(origin: str, destination: str) -> List[Tuple[float, float]]:
    start = CITY_COORDS[origin]
    end = CITY_COORDS[destination]
    url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
    try:
        response = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=10)
        response.raise_for_status()
        geometry = response.json()["routes"][0]["geometry"]["coordinates"]
        return [(lat, lon) for lon, lat in geometry]
    except Exception:
        return [start, end]

# ============================================================
# Session state
# ============================================================
if "field_reports" not in st.session_state:
    st.session_state.field_reports = FIELD_REPORTS_DEFAULT.copy()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "weather_snapshot" not in st.session_state:
    st.session_state.weather_snapshot = get_weather_snapshot(tuple(NER_STATES.keys()))
if "rain_boost" not in st.session_state:
    st.session_state.rain_boost = 0
if "route_coords" not in st.session_state:
    st.session_state.route_coords = None
if "route_summary" not in st.session_state:
    st.session_state.route_summary = None

# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.title("🛰️ NER Command Center")
    st.caption("Live GIS • accessibility • logistics • hazard intelligence")
    st.divider()

    mode = st.radio("Data mode", ["LIVE", "DEMO"], index=0, horizontal=True)
    selected_states = st.multiselect("States / UTs", list(NER_STATES.keys()), default=list(NER_STATES.keys()))
    risk_filter = st.selectbox("Corridor risk", ["All", "High", "Medium", "Low"])

    st.markdown("**Map layers**")
    show_rivers = st.checkbox("Rivers / water systems", True)
    show_traffic = st.checkbox("Traffic hotspots", True)
    show_vehicles = st.checkbox("Vehicle GPS", True)
    show_risk_zones = st.checkbox("Natural-disaster risk zones", True)
    show_heatmap = st.checkbox("Risk heat surface", False)
    show_facilities = st.checkbox("Service / logistics nodes", True)
    show_reports = st.checkbox("Field reports", True)

    st.markdown("**Scenario controls**")
    st.session_state.rain_boost = st.slider("Extra rainfall stress (mm)", 0, 80, st.session_state.rain_boost, 5)

    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=True)
    refresh = st.button("🔄 Refresh live weather / GPS / risk", use_container_width=True)
    if refresh:
        get_weather_snapshot.clear()
        st.session_state.weather_snapshot = get_weather_snapshot(tuple(selected_states or NER_STATES.keys()))
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.divider()
    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%d %b %Y, %H:%M:%S')}")
    st.caption("LIVE weather uses Open-Meteo. GPS/traffic use configured APIs when available; otherwise the app clearly labels estimated/demo data.")

# Auto-refresh turns the dashboard into a live-monitoring view; GPS/traffic endpoints are polled on each rerun.
if auto_refresh:
    st_autorefresh(interval=30_000, key="ner_live_refresh")

# Always have at least one state for calculations.
if not selected_states:
    selected_states = list(NER_STATES.keys())

# Refresh weather cache for the current selection.
st.session_state.weather_snapshot = get_weather_snapshot(tuple(selected_states))

weather_source_live = all(v["live"] for v in st.session_state.weather_snapshot.values())
vehicles, gps_live = get_live_vehicles(mode == "LIVE")

risk_status = {}
for seg in ROAD_SEGMENTS:
    if seg["state"] not in selected_states:
        continue
    weather = st.session_state.weather_snapshot[seg["state"]]
    traffic = traffic_for_segment(seg, mode == "LIVE")
    risk_status[seg["id"]] = compute_segment_status(seg, weather, traffic, st.session_state.rain_boost)

# ============================================================
# Header and KPIs
# ============================================================
status_html = '<span class="live-pill">● LIVE DATA CONNECTED</span>' if (mode == "LIVE" and (weather_source_live or gps_live)) else '<span class="demo-pill">● DEMO / ESTIMATED DATA</span>'
st.markdown('<p class="main-header">🛰️ NER Smart Logistics Accessibility Intelligence Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">One map for roads, corridors, rivers, traffic, vehicles, natural-hazard risk, incidents and nearby logistics infrastructure.</p>', unsafe_allow_html=True)
st.markdown(status_html, unsafe_allow_html=True)

high_risk = sum(1 for r in risk_status.values() if r["level"] == "High")
medium_risk = sum(1 for r in risk_status.values() if r["level"] == "Medium")
traffic_hotspots = sum(1 for r in risk_status.values() if r["traffic"]["level"] in ("Heavy", "Severe"))
tracked = len(vehicles)
vehicle_risk = sum(1 for v in vehicles if v.get("status") in ("Delayed", "High Risk Zone"))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Critical corridors", high_risk)
c2.metric("Medium-risk corridors", medium_risk)
c3.metric("Traffic hotspots", traffic_hotspots)
c4.metric("Tracked vehicles", tracked)
c5.metric("Vehicles at risk", vehicle_risk)
c6.metric("Field incidents", len(st.session_state.field_reports))

st.divider()

# ============================================================
# Main interface
# ============================================================
tab_map, tab_risk, tab_route, tab_vehicle, tab_alerts, tab_weather = st.tabs([
    "🗺️ Command Map",
    "⚠️ Risk & Disaster Prediction",
    "🛣️ Route & Accessibility",
    "🚛 Live Vehicle Tracking",
    "🚨 Alerts & Field Reports",
    "🌧️ Weather & Geography",
])

with tab_map:
    st.subheader("Live-style GIS command map")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.caption("Use the layer control inside the map to switch roads, rivers, traffic, vehicle GPS, hazards, facilities and field reports on/off.")
        toggles = {
            "rivers": show_rivers,
            "traffic": show_traffic,
            "vehicles": show_vehicles,
            "risk_zones": show_risk_zones,
            "heatmap": show_heatmap,
            "facilities": show_facilities,
            "reports": show_reports,
        }
        m = build_map(selected_states, risk_status, vehicles, st.session_state.field_reports, toggles, st.session_state.route_coords)
        st_folium(m, width=None, height=680, returned_objects=[])
    with col_b:
        st.markdown("### How to read the map")
        st.markdown("""
        **🔴 High risk** — corridor or surrounding hazard score is elevated.  
        **🟠 Medium risk** — monitor conditions and field reports.  
        **🟢 Low risk** — lower modeled disruption level.  
        **🔵 Blue lines** — major river / water-system reference layer.  
        **🚛 Purple** — normal tracked vehicle.  
        **🚛 Dark red** — delayed / high-risk vehicle.  
        **⚠️ Incident marker** — geo-tagged field report.
        """)
        st.markdown("### Surrounding context")
        st.write("The base map exposes surrounding streets and places; optional satellite and light-map layers provide additional context. Colored overlays are the platform's logistics and hazard interpretation.")
        st.markdown("### Data status")
        st.write(f"Weather: {'LIVE' if weather_source_live else 'fallback/demo'}")
        st.write(f"GPS: {'LIVE API' if gps_live else 'demo positions'}")
        traffic_live = any(v["traffic"]["live"] for v in risk_status.values())
        st.write(f"Traffic: {'LIVE API' if traffic_live else 'estimated'}")

with tab_risk:
    st.subheader("Geospatial risk & natural-disaster prediction")
    st.caption("The model combines rainfall, terrain slope, elevation, road condition, river proximity, historical incident index and traffic disruption. This prototype is a decision-support model, not an official disaster warning system.")

    rows = []
    for seg in ROAD_SEGMENTS:
        if seg["id"] not in risk_status:
            continue
        r = risk_status[seg["id"]]
        if risk_filter != "All" and r["level"] != risk_filter:
            continue
        rows.append({
            "Corridor": seg["name"],
            "State": seg["state"],
            "Risk": r["level"],
            "Risk score": f"{r['score']:.0%}",
            "Primary hazard": r["disaster_type"],
            "Landslide": f"{r['landslide']:.0%}",
            "Flood": f"{r['flood']:.0%}",
            "Traffic": r["traffic"]["level"],
            "24h rain": f"{r['rain24']:.1f} mm",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if risk_status:
        selected_corridor = st.selectbox("Explain one corridor", list(risk_status.keys()), format_func=lambda x: next(s["name"] for s in ROAD_SEGMENTS if s["id"] == x))
        seg = next(s for s in ROAD_SEGMENTS if s["id"] == selected_corridor)
        r = risk_status[selected_corridor]
        st.markdown(f"### {seg['name']}")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Composite risk", f"{r['score']:.0%}")
        x2.metric("Landslide", f"{r['landslide']:.0%}")
        x3.metric("Flood / waterlogging", f"{r['flood']:.0%}")
        x4.metric("Traffic", r["traffic"]["level"])
        st.write("**Main geographic drivers**")
        st.json(r["factors"])

with tab_route:
    st.subheader("Route & accessibility analysis")
    o, d = st.columns(2)
    city_options = sorted(CITY_COORDS.keys())
    with o:
        origin = st.selectbox("Origin", city_options, index=city_options.index("Guwahati"))
    with d:
        destination = st.selectbox("Destination", city_options, index=city_options.index("Itanagar"))
    vehicle_type = st.radio("Cargo type", ["Medicine / emergency", "Food / general supply", "Heavy construction"], horizontal=True)

    if st.button("🧭 Calculate accessible route", type="primary"):
        route = try_osrm_route(origin, destination)
        st.session_state.route_coords = route
        distance = 0.0
        for a, b in zip(route[:-1], route[1:]):
            distance += haversine_km(a, b)
        base_hours = max(1.0, distance / 45)
        if vehicle_type == "Heavy construction":
            base_hours *= 1.25
        st.session_state.route_summary = {"distance": distance, "hours": base_hours}
        st.success(f"Route generated: {origin} → {destination}")

    if st.session_state.route_summary:
        route_sum = st.session_state.route_summary
        a, b, c = st.columns(3)
        a.metric("Route distance", f"{route_sum['distance']:.0f} km")
        b.metric("Base travel estimate", f"{route_sum['hours']:.1f} h")
        c.metric("Map status", "Route displayed")
        st.info("The purple line on the Command Map is a road-network route from OSRM. Traffic/hazard overlays are kept separate so users can understand why a route may be disrupted.")

with tab_vehicle:
    st.subheader("Live GPS / essential-cargo tracking")
    dfv = pd.DataFrame(vehicles)
    cols = ["id", "type", "commodity", "origin", "dest", "status", "eta_hrs", "lat", "lon"]
    cols = [c for c in cols if c in dfv.columns]
    st.dataframe(dfv[cols].rename(columns={"id": "Vehicle ID", "type": "Type", "commodity": "Cargo", "origin": "From", "dest": "To", "status": "Status", "eta_hrs": "ETA (h)", "lat": "Latitude", "lon": "Longitude"}), use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    a.success(f"Normal / in transit: {sum(1 for v in vehicles if v.get('status') == 'In Transit')}")
    b.warning(f"Delayed: {sum(1 for v in vehicles if v.get('status') == 'Delayed')}")
    c.error(f"High risk zone: {sum(1 for v in vehicles if v.get('status') == 'High Risk Zone')}")
    st.caption("For true live fleet tracking, set GPS_API_URL to a REST endpoint that returns a JSON array (or {'vehicles': [...]}) with id, lat, lon, status and cargo fields.")

with tab_alerts:
    st.subheader("Unified alert centre")
    alerts = []
    for seg in ROAD_SEGMENTS:
        if seg["id"] not in risk_status:
            continue
        r = risk_status[seg["id"]]
        if r["level"] == "High":
            alerts.append(("High", f"{seg['name']}: {r['disaster_type']} risk {r['score']:.0%}; traffic is {r['traffic']['level']}."))
        elif r["level"] == "Medium":
            alerts.append(("Medium", f"{seg['name']}: monitor conditions; modeled risk {r['score']:.0%}."))
    for v in vehicles:
        if v.get("status") != "In Transit":
            sev = "High" if v.get("status") == "High Risk Zone" else "Medium"
            alerts.append((sev, f"Vehicle {v.get('id')}: {v.get('status')} — {v.get('commodity', 'cargo')}."))
    for rep in st.session_state.field_reports:
        alerts.append((rep["severity"], f"Field report at {rep['location']}: {rep['type']} ({rep['reporter']})."))

    if not alerts:
        st.success("No active alerts from the current filters.")
    else:
        for level, message in sorted(alerts, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x[0]]):
            css = "alert-high" if level == "High" else "alert-medium" if level == "Medium" else "alert-low"
            st.markdown(f'<div class="{css}"><b>{level}</b> — {message}</div>', unsafe_allow_html=True)

    st.markdown("### Submit geo-tagged field report")
    a, b = st.columns(2)
    with a:
        loc = st.text_input("Location / road")
        rtype = st.selectbox("Incident", ["Landslide", "Flooding / waterlogging", "Road damage", "Heavy rain", "Bridge issue", "Traffic congestion", "Other"])
    with b:
        sev = st.select_slider("Severity", ["Low", "Medium", "High"])
        lat = st.number_input("Latitude", value=26.20, format="%.5f")
        lon = st.number_input("Longitude", value=92.94, format="%.5f")
    if st.button("📍 Add field report"):
        st.session_state.field_reports.insert(0, {"time": datetime.now().strftime("%H:%M"), "location": loc or "Unspecified", "type": rtype, "severity": sev, "reporter": "Demo Field Officer", "lat": lat, "lon": lon})
        st.success("Field report added to the map and alert centre.")
        st.rerun()

with tab_weather:
    st.subheader("Weather, terrain & surrounding geography")
    wx_rows = []
    for state in selected_states:
        w = st.session_state.weather_snapshot[state]
        wx_rows.append({"State": state, "Temperature °C": round(w["temp"], 1), "Current rain mm": round(w["rain_mm"], 1), "Wind km/h": round(w["wind_kmh"], 1), "Source": w["source"]})
    st.dataframe(pd.DataFrame(wx_rows), use_container_width=True, hide_index=True)

    st.markdown("### Why the map can predict risk")
    st.write("A geographic-risk engine can combine weather intensity with terrain slope, elevation, proximity to rivers, road condition, historical incident patterns and live/estimated traffic. The current prototype demonstrates that logic with reference geographic features; production deployment should replace proxies with authoritative GIS layers, hydrology, DEM, road-condition and historical-disaster datasets.")
    st.markdown("### Important surrounding layers for the production version")
    st.write("Road network • state/district boundaries • rivers/drainage • bridges • tunnels • hospitals • warehouses • airports • fuel stations • telecom coverage • landslide/flood susceptibility • rain radar • road works • closures • traffic speed • vehicle GPS • emergency shelters.")

st.divider()
st.caption("NER Smart Logistics Accessibility Intelligence Platform • Upgraded GIS command-center prototype • Weather can be live via Open-Meteo; GPS and traffic become live when their API integrations are configured. Hazard predictions are prototype decision-support outputs, not official warnings.")
