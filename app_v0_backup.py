import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import random
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="NER Smart Logistics Intelligence Platform",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-high { background-color: #fed7d7; border-left: 5px solid #e53e3e; padding: 0.8rem; margin: 0.4rem 0; border-radius: 4px; }
    .alert-medium { background-color: #feebc8; border-left: 5px solid #dd6b20; padding: 0.8rem; margin: 0.4rem 0; border-radius: 4px; }
    .alert-low { background-color: #c6f6d5; border-left: 5px solid #38a169; padding: 0.8rem; margin: 0.4rem 0; border-radius: 4px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Simulated Data - NER Focus
# -----------------------------
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

# Major corridors / sample road segments (simplified)
ROAD_SEGMENTS = [
    {"id": "NH27-AS-01", "name": "Guwahati – Tezpur", "state": "Assam", "lat": 26.35, "lon": 92.80, "base_time": 3.5, "type": "NH"},
    {"id": "NH15-AS-02", "name": "Tezpur – North Lakhimpur", "state": "Assam", "lat": 27.10, "lon": 94.10, "base_time": 4.0, "type": "NH"},
    {"id": "NH13-AR-01", "name": "Itanagar – Ziro", "state": "Arunachal Pradesh", "lat": 27.55, "lon": 93.85, "base_time": 5.5, "type": "NH"},
    {"id": "NH29-NL-01", "name": "Dimapur – Kohima", "state": "Nagaland", "lat": 25.90, "lon": 94.10, "base_time": 3.0, "type": "NH"},
    {"id": "NH2-MN-01", "name": "Imphal – Moreh", "state": "Manipur", "lat": 24.50, "lon": 94.20, "base_time": 4.5, "type": "NH"},
    {"id": "NH6-ML-01", "name": "Shillong – Guwahati", "state": "Meghalaya", "lat": 25.70, "lon": 91.90, "base_time": 3.2, "type": "NH"},
    {"id": "NH54-MZ-01", "name": "Aizawl – Silchar", "state": "Mizoram", "lat": 24.20, "lon": 92.80, "base_time": 6.0, "type": "NH"},
    {"id": "NH10-SK-01", "name": "Gangtok – Siliguri", "state": "Sikkim", "lat": 27.20, "lon": 88.50, "base_time": 4.8, "type": "NH"},
    {"id": "NH8-TR-01", "name": "Agartala – Sabroom", "state": "Tripura", "lat": 23.40, "lon": 91.70, "base_time": 3.8, "type": "NH"},
    {"id": "SH-AS-03", "name": "Jorhat – Sivasagar", "state": "Assam", "lat": 26.85, "lon": 94.40, "base_time": 1.5, "type": "SH"},
    {"id": "MDR-AR-02", "name": "Bomdila – Tawang", "state": "Arunachal Pradesh", "lat": 27.45, "lon": 92.10, "base_time": 7.0, "type": "MDR"},
    {"id": "NH37-AS-04", "name": "Dibrugarh – Tinsukia", "state": "Assam", "lat": 27.45, "lon": 95.20, "base_time": 2.2, "type": "NH"},
]

# Sample vehicles carrying essential goods
VEHICLES = [
    {"id": "MED-001", "type": "Medicine Van", "origin": "Guwahati", "dest": "Itanagar", "lat": 26.80, "lon": 93.50, "status": "In Transit", "eta_hrs": 4.2, "commodity": "Essential Medicines"},
    {"id": "FOOD-012", "type": "Supply Truck", "origin": "Siliguri", "dest": "Gangtok", "lat": 27.00, "lon": 88.40, "status": "In Transit", "eta_hrs": 2.8, "commodity": "Food Supplies"},
    {"id": "AGR-007", "type": "Agri Truck", "origin": "Jorhat", "dest": "Dibrugarh", "lat": 27.20, "lon": 94.90, "status": "Delayed", "eta_hrs": 3.5, "commodity": "Agricultural Produce"},
    {"id": "CONST-003", "type": "Construction", "origin": "Dimapur", "dest": "Kohima", "lat": 25.85, "lon": 94.05, "status": "In Transit", "eta_hrs": 1.5, "commodity": "Construction Materials"},
    {"id": "MED-015", "type": "Medicine Van", "origin": "Agartala", "dest": "Aizawl", "lat": 23.80, "lon": 92.40, "status": "High Risk Zone", "eta_hrs": 5.0, "commodity": "Vaccines & Medicines"},
]

# -----------------------------
# Mock AI Risk Model
# -----------------------------
@st.cache_resource
def train_mock_risk_model():
    """Train a simple GradientBoosting model on synthetic data that mimics NER conditions."""
    np.random.seed(42)
    n = 800
    # Features: rainfall_24h, slope, road_condition, historical_events, soil_moisture, elevation
    X = np.column_stack([
        np.random.exponential(15, n),          # rainfall
        np.random.uniform(5, 45, n),           # slope
        np.random.uniform(0.3, 1.0, n),        # road condition (1=good)
        np.random.poisson(2, n),               # historical events
        np.random.uniform(0.2, 0.95, n),       # soil moisture
        np.random.uniform(50, 3500, n)         # elevation
    ])
    # Higher chance of blockage with high rain + steep slope + poor road + history
    logit = (0.08 * X[:,0] + 0.06 * X[:,1] - 1.5 * X[:,2] + 0.4 * X[:,3] + 1.2 * X[:,4] - 0.0003 * X[:,5] - 3)
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n) < prob).astype(int)
    
    model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

risk_model = train_mock_risk_model()

def predict_segment_risk(segment, rainfall=None, hour=None):
    """Predict risk for a road segment using the mock AI model + heuristics."""
    if rainfall is None:
        # Simulate current weather influence
        rainfall = random.uniform(2, 45) if random.random() > 0.4 else random.uniform(0, 8)
    
    slope = 15 + (hash(segment["id"]) % 25)          # pseudo terrain
    road_cond = 0.75 if segment["type"] == "NH" else 0.55
    hist = 1 + (hash(segment["name"]) % 4)
    soil = 0.4 + (hash(segment["id"]) % 50) / 100
    elev = 200 + (hash(segment["state"]) % 2000)
    
    features = np.array([[rainfall, slope, road_cond, hist, soil, elev]])
    proba = risk_model.predict_proba(features)[0][1]
    
    # Add some temporal / random variation
    proba = min(0.95, max(0.05, proba + random.uniform(-0.08, 0.12)))
    
    if proba > 0.65:
        level = "High"
        color = "red"
    elif proba > 0.35:
        level = "Medium"
        color = "orange"
    else:
        level = "Low"
        color = "green"
    
    return {
        "probability": round(proba, 3),
        "level": level,
        "color": color,
        "rainfall_mm": round(rainfall, 1),
        "factors": {
            "Rainfall (24h)": f"{rainfall:.1f} mm",
            "Terrain Slope": f"{slope:.0f}°",
            "Road Condition": "Good" if road_cond > 0.7 else "Fair/Poor",
            "Historical Events": hist
        }
    }

# Pre-compute risks for all segments
def get_all_segment_risks():
    risks = {}
    for seg in ROAD_SEGMENTS:
        risks[seg["id"]] = predict_segment_risk(seg)
    return risks

# -----------------------------
# Session State
# -----------------------------
if "segment_risks" not in st.session_state:
    st.session_state.segment_risks = get_all_segment_risks()
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "field_reports" not in st.session_state:
    st.session_state.field_reports = [
        {"time": "10:15", "location": "NH13 near Ziro", "type": "Landslide debris", "severity": "High", "reporter": "Field Officer - AR"},
        {"time": "08:40", "location": "NH29 Dimapur-Kohima", "type": "Heavy rain / slippery", "severity": "Medium", "reporter": "PWD Nagaland"},
    ]
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/truck.png", width=70)
    st.title("NER Logistics AI")
    st.caption("Smart Accessibility Intelligence Platform")
    st.markdown("---")
    
    st.subheader("Filters")
    selected_states = st.multiselect(
        "States / UTs",
        options=list(NER_STATES.keys()),
        default=list(NER_STATES.keys())
    )
    
    risk_filter = st.selectbox("Risk Level Filter", ["All", "High", "Medium", "Low"])
    
    st.markdown("---")
    st.subheader("Simulation Controls")
    if st.button("🔄 Refresh Risk Predictions", use_container_width=True):
        st.session_state.segment_risks = get_all_segment_risks()
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    rain_boost = st.slider("Simulate Extra Rainfall (mm)", 0, 80, 0, 5)
    st.caption("Higher values increase predicted disruption risk")
    
    st.markdown("---")
    st.info(f"Last AI refresh:\n{st.session_state.last_refresh.strftime('%H:%M:%S')}")
    st.caption("Prototype • Simulated Data • NER Focus")

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="main-header">🚛 NER Smart Logistics Accessibility Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered real-time monitoring, disruption prediction & route optimization for North Eastern Region</p>', unsafe_allow_html=True)

# -----------------------------
# KPI Row
# -----------------------------
risks = st.session_state.segment_risks
high_risk_count = sum(1 for r in risks.values() if r["level"] == "High")
medium_risk_count = sum(1 for r in risks.values() if r["level"] == "Medium")
active_vehicles = len(VEHICLES)
delayed_vehicles = sum(1 for v in VEHICLES if v["status"] in ["Delayed", "High Risk Zone"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Monitored Segments", len(ROAD_SEGMENTS), help="Key NH/SH corridors")
col2.metric("High Risk Corridors", high_risk_count, delta=f"{high_risk_count} critical" if high_risk_count else "Stable", delta_color="inverse")
col3.metric("Medium Risk", medium_risk_count)
col4.metric("Tracked Vehicles", active_vehicles)
col5.metric("Delayed / At Risk", delayed_vehicles, delta_color="inverse")

st.markdown("---")

# -----------------------------
# Main Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺 Live Accessibility Map",
    "🤖 AI Risk Predictions",
    "🛣️ Route Optimizer",
    "🚚 Vehicle Tracking",
    "🚨 Alerts & Field Reports",
    "📊 District Dashboard"
])

# ========== TAB 1: Live Map ==========
with tab1:
    st.subheader("Real-time Road & Corridor Accessibility")
    
    # Create map centered on NER
    m = folium.Map(location=[26.0, 93.0], zoom_start=6, tiles="OpenStreetMap")
    
    # Add state markers
    for state, info in NER_STATES.items():
        if state in selected_states:
            folium.Marker(
                location=[info["lat"], info["lon"]],
                popup=f"<b>{state}</b><br>Capital: {info['capital']}",
                tooltip=state,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
    
    # Add road segments with risk coloring
    for seg in ROAD_SEGMENTS:
        if seg["state"] not in selected_states:
            continue
        risk = risks[seg["id"]]
        if risk_filter != "All" and risk["level"] != risk_filter:
            continue
        
        # Apply rain boost if set
        adj_prob = min(0.98, risk["probability"] + (rain_boost / 120))
        if adj_prob > 0.65:
            color = "red"
            level = "High"
        elif adj_prob > 0.35:
            color = "orange"
            level = "Medium"
        else:
            color = "green"
            level = "Low"
        
        folium.CircleMarker(
            location=[seg["lat"], seg["lon"]],
            radius=11,
            popup=folium.Popup(
                f"""<b>{seg['name']}</b><br>
                ID: {seg['id']}<br>
                Type: {seg['type']}<br>
                Risk: <b>{level}</b> ({adj_prob:.0%})<br>
                Base Travel: {seg['base_time']} hrs<br>
                Rainfall influence: {risk['rainfall_mm']} mm""",
                max_width=280
            ),
            tooltip=f"{seg['name']} • {level} Risk",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75
        ).add_to(m)
    
    # Add vehicles
    for v in VEHICLES:
        color = "purple" if v["status"] == "In Transit" else "darkred"
        folium.Marker(
            location=[v["lat"], v["lon"]],
            popup=f"<b>{v['id']}</b><br>{v['commodity']}<br>Status: {v['status']}<br>ETA: {v['eta_hrs']} hrs",
            tooltip=f"{v['id']} – {v['commodity']}",
            icon=folium.Icon(color=color, icon="truck", prefix="fa")
        ).add_to(m)
    
    st_folium(m, width=None, height=520, returned_objects=[])
    
    st.caption("🟢 Low Risk  |  🟠 Medium Risk  |  🔴 High Risk  |  🚛 Vehicles (purple = normal, dark red = delayed/at-risk)")

# ========== TAB 2: AI Risk Predictions ==========
with tab2:
    st.subheader("AI-Powered Disruption Risk Predictions")
    st.markdown("Model: Gradient Boosting Classifier trained on synthetic NER-like features (rainfall, slope, road condition, historical events, soil moisture, elevation)")
    
    # Risk table
    rows = []
    for seg in ROAD_SEGMENTS:
        if seg["state"] not in selected_states:
            continue
        r = risks[seg["id"]]
        adj_prob = min(0.98, r["probability"] + (rain_boost / 120))
        level = "High" if adj_prob > 0.65 else ("Medium" if adj_prob > 0.35 else "Low")
        rows.append({
            "Segment ID": seg["id"],
            "Corridor": seg["name"],
            "State": seg["state"],
            "Type": seg["type"],
            "Risk Probability": f"{adj_prob:.1%}",
            "Risk Level": level,
            "Key Factor": r["factors"]["Rainfall (24h)"],
            "Base Travel (hrs)": seg["base_time"]
        })
    
    df_risk = pd.DataFrame(rows)
    
    # Color the risk level column
    def color_risk(val):
        if val == "High":
            return "background-color: #fc8181; color: white; font-weight: bold"
        elif val == "Medium":
            return "background-color: #f6ad55; color: black; font-weight: bold"
        return "background-color: #68d391; color: black"
    
    if not df_risk.empty:
        st.dataframe(
            df_risk.style.map(color_risk, subset=["Risk Level"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No segments match current filters.")
    
    st.markdown("#### Explainability – Top Contributing Factors (sample)")
    sample_seg = st.selectbox("Select segment for detailed AI explanation", 
                              options=[s["name"] for s in ROAD_SEGMENTS if s["state"] in selected_states])
    if sample_seg:
        seg = next(s for s in ROAD_SEGMENTS if s["name"] == sample_seg)
        r = risks[seg["id"]]
        cols = st.columns(4)
        for i, (k, v) in enumerate(r["factors"].items()):
            cols[i % 4].metric(k, v)
        
        st.info(f"**AI Insight:** Current predicted blockage probability is **{min(0.98, r['probability'] + rain_boost/120):.1%}**. "
                f"Main drivers are recent rainfall and terrain characteristics typical of the {seg['state']} corridor.")

# ========== TAB 3: Route Optimizer ==========
with tab3:
    st.subheader("AI Route Optimization & Alternate Suggestions")
    
    col_a, col_b = st.columns(2)
    with col_a:
        origin = st.selectbox("Origin", ["Guwahati", "Siliguri", "Dimapur", "Imphal", "Agartala", "Jorhat", "Itanagar"])
    with col_b:
        dest = st.selectbox("Destination", ["Itanagar", "Gangtok", "Kohima", "Aizawl", "Dibrugarh", "Tawang", "Moreh"])
    
    vehicle_type = st.radio("Vehicle Type", ["Light Vehicle / Medicine Van", "Heavy Truck (Food/Construction)"], horizontal=True)
    
    if st.button("🔍 Generate Optimized Routes", type="primary"):
        with st.spinner("Running risk-aware multi-objective routing..."):
            # Simulated routing logic
            base_routes = [
                {
                    "name": "Primary Recommended",
                    "distance_km": random.randint(180, 420),
                    "base_hrs": round(random.uniform(4.5, 9.5), 1),
                    "risk_score": round(random.uniform(0.15, 0.55), 2),
                    "segments": random.sample([s["name"] for s in ROAD_SEGMENTS], k=3),
                    "notes": "Balances time and current risk scores"
                },
                {
                    "name": "Low-Risk Alternative",
                    "distance_km": random.randint(220, 480),
                    "base_hrs": round(random.uniform(5.5, 11.0), 1),
                    "risk_score": round(random.uniform(0.08, 0.30), 2),
                    "segments": random.sample([s["name"] for s in ROAD_SEGMENTS], k=3),
                    "notes": "Avoids current high-risk corridors; longer but safer"
                },
                {
                    "name": "Fastest (Higher Risk)",
                    "distance_km": random.randint(160, 380),
                    "base_hrs": round(random.uniform(3.8, 8.0), 1),
                    "risk_score": round(random.uniform(0.40, 0.75), 2),
                    "segments": random.sample([s["name"] for s in ROAD_SEGMENTS], k=2),
                    "notes": "Shortest time but passes near elevated risk zones"
                }
            ]
            
            # Adjust for vehicle type
            if "Heavy" in vehicle_type:
                for rt in base_routes:
                    rt["base_hrs"] = round(rt["base_hrs"] * 1.25, 1)
                    rt["notes"] += " | Heavy vehicle restrictions applied"
            
            st.success(f"Routes computed for **{origin} → {dest}**")
            
            for i, rt in enumerate(base_routes):
                risk_level = "High" if rt["risk_score"] > 0.55 else ("Medium" if rt["risk_score"] > 0.30 else "Low")
                with st.expander(f"{'⭐ ' if i==0 else ''}{rt['name']}  |  Risk: {risk_level} ({rt['risk_score']:.0%})  |  ETA: {rt['base_hrs']} hrs", expanded=(i==0)):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Distance", f"{rt['distance_km']} km")
                    c2.metric("Estimated Time", f"{rt['base_hrs']} hrs")
                    c3.metric("Composite Risk", f"{rt['risk_score']:.0%}")
                    st.write("**Via segments:**", " → ".join(rt["segments"]))
                    st.caption(rt["notes"])
                    if risk_level == "High":
                        st.warning("⚠️ This route currently carries elevated disruption probability. Consider the Low-Risk Alternative for critical cargo (medicines).")

# ========== TAB 4: Vehicle Tracking ==========
with tab4:
    st.subheader("GPS-based Tracking of Essential Commodity Vehicles")
    
    df_veh = pd.DataFrame(VEHICLES)
    st.dataframe(
        df_veh[["id", "type", "commodity", "origin", "dest", "status", "eta_hrs"]].rename(columns={
            "id": "Vehicle ID", "type": "Type", "commodity": "Commodity",
            "origin": "From", "dest": "To", "status": "Status", "eta_hrs": "ETA (hrs)"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("#### Live Status Summary")
    c1, c2, c3 = st.columns(3)
    c1.success(f"**In Transit (Normal):** {sum(1 for v in VEHICLES if v['status']=='In Transit')}")
    c2.warning(f"**Delayed:** {sum(1 for v in VEHICLES if v['status']=='Delayed')}")
    c3.error(f"**High Risk Zone:** {sum(1 for v in VEHICLES if v['status']=='High Risk Zone')}")
    
    st.info("In a full system this would stream live GPS coordinates from devices fitted on medicine vans, food trucks and construction vehicles, with geofencing alerts when entering high-risk corridors.")

# ========== TAB 5: Alerts & Field Reports ==========
with tab5:
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("🚨 Active Alerts")
        
        # Generate dynamic alerts from current risks
        dynamic_alerts = []
        for seg in ROAD_SEGMENTS:
            r = risks[seg["id"]]
            adj = min(0.98, r["probability"] + rain_boost/120)
            if adj > 0.65:
                dynamic_alerts.append({
                    "level": "High",
                    "msg": f"High disruption risk on **{seg['name']}** ({seg['state']}) – Probability {adj:.0%}. Consider re-routing essential cargo."
                })
            elif adj > 0.45:
                dynamic_alerts.append({
                    "level": "Medium",
                    "msg": f"Elevated risk on **{seg['name']}**. Monitor rainfall and field updates."
                })
        
        for v in VEHICLES:
            if v["status"] != "In Transit":
                dynamic_alerts.append({
                    "level": "High" if "Risk" in v["status"] else "Medium",
                    "msg": f"Vehicle **{v['id']}** ({v['commodity']}) is marked **{v['status']}**. Current ETA {v['eta_hrs']} hrs."
                })
        
        if not dynamic_alerts:
            st.success("No critical alerts at the moment. Network status stable.")
        else:
            for a in dynamic_alerts[:8]:
                css = "alert-high" if a["level"] == "High" else "alert-medium"
                st.markdown(f'<div class="{css}">{a["msg"]}</div>', unsafe_allow_html=True)
    
    with col_right:
        st.subheader("📝 Field Reports (Geo-tagged)")
        
        for fr in st.session_state.field_reports:
            st.markdown(f"""
            **{fr['time']}** • {fr['location']}  
            Type: `{fr['type']}` | Severity: **{fr['severity']}**  
            Reporter: {fr['reporter']}
            """)
            st.markdown("---")
        
        with st.expander("➕ Submit New Field Report (Simulation)"):
            loc = st.text_input("Location / Road Segment")
            rtype = st.selectbox("Incident Type", ["Landslide", "Flooding / Waterlogging", "Road Damage", "Heavy Rain", "Bridge Issue", "Traffic Congestion", "Other"])
            sev = st.select_slider("Severity", ["Low", "Medium", "High"])
            if st.button("Submit Report"):
                st.session_state.field_reports.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "location": loc or "Unspecified",
                    "type": rtype,
                    "severity": sev,
                    "reporter": "Demo Field Officer"
                })
                st.success("Report submitted (simulated). In production this would be geo-tagged + photo + offline-capable.")
                st.rerun()

# ========== TAB 6: District Dashboard ==========
with tab6:
    st.subheader("District / State Connectivity & Logistics Overview")
    
    # Aggregate by state
    state_stats = []
    for state in selected_states:
        segs = [s for s in ROAD_SEGMENTS if s["state"] == state]
        if not segs:
            continue
        high = sum(1 for s in segs if risks[s["id"]]["level"] == "High")
        med = sum(1 for s in segs if risks[s["id"]]["level"] == "Medium")
        avg_risk = np.mean([risks[s["id"]]["probability"] for s in segs])
        state_stats.append({
            "State": state,
            "Monitored Corridors": len(segs),
            "High Risk": high,
            "Medium Risk": med,
            "Avg Risk Score": round(avg_risk, 3),
            "Status": "Critical" if high >= 1 else ("Caution" if med >= 1 else "Stable")
        })
    
    df_state = pd.DataFrame(state_stats)
    if not df_state.empty:
        st.dataframe(df_state, use_container_width=True, hide_index=True)
        
        # Simple bar chart of risk
        st.bar_chart(df_state.set_index("State")[["High Risk", "Medium Risk"]])
    
    st.markdown("#### Logistics Bottleneck Indicators (Simulated)")
    st.write("""
    - **Assam (Guwahati hub)** remains the primary logistics gateway; any disruption on NH27/NH15 cascades region-wide.  
    - **Arunachal Pradesh** corridors (especially toward Tawang & Ziro) show higher terrain-driven risk.  
    - **Medicine supply chains** to remote districts benefit most from proactive re-routing when High risk is flagged.  
    - During monsoon, the platform prioritizes emergency accessibility routes for health & disaster response.
    """)
    
    st.success("This dashboard would integrate live district-level connectivity status, supply gap analysis, and emergency corridor recommendations in the full system.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("NER Smart Logistics Accessibility Intelligence Platform • Prototype v0.9 • AI models are simulated for demonstration • Data is synthetic and for illustration only")
st.caption("Built to demonstrate: Real-time monitoring • AI disruption prediction • Route optimization • Vehicle tracking • Alerts • Field reporting • Multilingual-ready architecture")
