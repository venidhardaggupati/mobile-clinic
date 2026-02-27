import math, os
import streamlit as st
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

# ── ML Integration (Hour 12) ──────────────────────────────────────────────────
try:
    from hour12_predictive_ml import load_model, predict_tomorrow
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RHLC · Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS  — dark tactical HMI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

/* ── CSS variables ────────────────────────────────────────────────────────── */
:root {
  --bg-primary:    #0a0f1e;
  --bg-secondary:  #0d1526;
  --bg-panel:      #111b30;
  --bg-card:       #162038;
  --accent-blue:   #00aaff;
  --accent-green:  #00ff88;
  --accent-red:    #ff3355;
  --accent-orange: #ff8c00;
  --accent-dim:    #1e3a5f;
  --text-primary:  #e8f4ff;
  --text-dim:      #607a99;
  --text-mono:     #00ccff;
  --border:        #1e3a5f;
  --radius:        12px;
  --glow-blue:     0 0 16px rgba(0,170,255,0.25);
  --glow-green:    0 0 16px rgba(0,255,136,0.2);
  --glow-red:      0 0 20px rgba(255,51,85,0.4);
}

/* ── Global reset ─────────────────────────────────────────────────────────── */
.stApp {
  background: var(--bg-primary) !important;
  font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text-primary) !important;
  letter-spacing: 0.04em;
}
h1 { font-size: 2.2rem !important; font-weight: 700 !important; }
h3 { font-size: 1.3rem !important; color: var(--accent-blue) !important; }

/* ── Paragraphs & captions ────────────────────────────────────────────────── */
p, .stMarkdown p, label, .stCaption { color: var(--text-dim) !important; }

/* ── Metric cards — digital gauge style ──────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--accent-dim) !important;
  border-radius: var(--radius) !important;
  padding: 16px 18px !important;
  box-shadow: var(--glow-blue) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 28px rgba(0,170,255,0.4) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--text-mono) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--accent-green) !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0066cc, #00aaff) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  padding: 0.6rem 1.2rem !important;
  box-shadow: 0 0 20px rgba(0,170,255,0.35) !important;
  transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 36px rgba(0,170,255,0.65) !important;
  transform: scale(1.02) !important;
}

/* ── Selectbox / Slider ───────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSlider > div { color: var(--text-primary) !important; }
[data-baseweb="select"] > div {
  background: var(--bg-card) !important;
  border-color: var(--accent-dim) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}

/* ── DataFrames ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel) !important;
  border-radius: var(--radius) var(--radius) 0 0 !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  color: var(--text-dim) !important;
  letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-blue) !important;
  border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
details {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 4px 8px !important;
}
summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  color: var(--accent-blue) !important;
  letter-spacing: 0.05em !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.stProgress > div > div { background: var(--bg-card) !important; border-radius: 8px !important; }
.stProgress > div > div > div {
  background: linear-gradient(90deg, #00aaff, #00ff88) !important;
  border-radius: 8px !important;
}

/* ── Alerts / Toasts ──────────────────────────────────────────────────────── */
.stSuccess {
  background: rgba(0,255,136,0.08) !important;
  border: 1px solid var(--accent-green) !important;
  border-radius: var(--radius) !important;
  color: var(--accent-green) !important;
}
.stWarning {
  background: rgba(255,140,0,0.08) !important;
  border: 1px solid var(--accent-orange) !important;
  border-radius: var(--radius) !important;
}
.stInfo {
  background: rgba(0,170,255,0.08) !important;
  border: 1px solid var(--accent-blue) !important;
  border-radius: var(--radius) !important;
}
.stError {
  background: rgba(255,51,85,0.08) !important;
  border: 1px solid var(--accent-red) !important;
  border-radius: var(--radius) !important;
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── scan-line texture overlay ───────────────────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.04) 2px,
    rgba(0,0,0,0.04) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Pulsing hotspot CSS (injected into Folium via DivIcon) ───────────────── */
</style>

<style>
@keyframes pulse-ring {
  0%   { transform: scale(0.6); opacity: 0.8; }
  70%  { transform: scale(2.2); opacity: 0; }
  100% { transform: scale(2.2); opacity: 0; }
}
@keyframes pulse-core {
  0%, 100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(255,51,85,0.7); }
  50%       { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(255,51,85,0); }
}
.pulse-wrapper {
  position: relative; width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
}
.pulse-ring {
  position: absolute; width: 22px; height: 22px;
  border: 2px solid #ff3355; border-radius: 50%;
  animation: pulse-ring 1.6s cubic-bezier(0.215,0.61,0.355,1) infinite;
}
.pulse-ring:nth-child(2) { animation-delay: 0.55s; }
.pulse-core {
  width: 14px; height: 14px; background: #ff3355; border-radius: 50%;
  border: 2px solid #fff;
  animation: pulse-core 1.6s ease-in-out infinite;
  z-index: 1;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PALETTE
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_FILES = [
    "villages.csv", "outbreak.csv", "outbreaks_critical.csv",
    "matrix_normal.csv", "matrix_monsoon.csv",
]

SCENARIOS = {
    "🌤  Normal Operations": {
        "matrix": "matrix_normal.csv", "outbreak": "outbreak.csv",
        "badge": "NORMAL", "color": "#00ff88",
        "desc": "Standard road conditions. All routes passable.",
    },
    "🌧  Monsoon (Flooded Roads)": {
        "matrix": "matrix_monsoon.csv", "outbreak": "outbreak.csv",
        "badge": "MONSOON", "color": "#00aaff",
        "desc": "Koheda & Rachakonda: 2.5× travel penalty due to flooding.",
    },
    "🚨  Critical Outbreak": {
        "matrix": "matrix_normal.csv", "outbreak": "outbreaks_critical.csv",
        "badge": "CRITICAL", "color": "#ff3355",
        "desc": "Ibrahimpatnam cases escalated ×5. Severity recalculated.",
    },
}

CLR = dict(
    critical="#ff3355", warning="#ff8c00", stable="#00ff88",
    visited="#00cc66", skipped="#3a4a5a", route="#00aaff",
    paper="rgba(0,0,0,0)", font="#e8f4ff",
)

CHART_BASE = dict(
    paper_bgcolor=CLR["paper"], plot_bgcolor="rgba(13,21,38,0.9)",
    font=dict(family="JetBrains Mono, monospace", size=11, color=CLR["font"]),
    margin=dict(l=12, r=12, t=40, b=12),
)


# ══════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION GATE
# ══════════════════════════════════════════════════════════════════════════════
def validate_files() -> bool:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if not missing:
        return True

    st.markdown("""
    <div style='text-align:center;padding:60px 40px;'>
      <div style='font-size:3rem;margin-bottom:16px;'>⚠️</div>
      <h2 style='font-family:Rajdhani,sans-serif;color:#00aaff;margin-bottom:8px;'>
        SYSTEM INITIALISATION REQUIRED
      </h2>
      <p style='color:#607a99;margin-bottom:24px;'>
        Missing data files detected. Run the generation scripts below to initialise the system.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.warning("**Missing files:**\n" + "\n".join(f"• `{f}`" for f in missing))
    with c2:
        st.info(
            "**To generate all required files, run in order:**\n\n"
            "```bash\n"
            "python generate_villages.py      # → villages.csv\n"
            "python generate_outbreak.py      # → outbreak.csv + outbreaks_critical.csv\n"
            "python generate_matrices.py      # → matrix_normal.csv + matrix_monsoon.csv\n"
            "```"
        )
    st.stop()
    return False


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_villages(): return pd.read_csv("villages.csv")


def load_scenario(outbreak_path: str) -> pd.DataFrame:
    villages = load_villages()
    outbreak = pd.read_csv(outbreak_path)
    merged = villages.merge(
        outbreak[["Village_ID", "Disease_Type", "Active_Cases", "Severity_Score"]],
        on="Village_ID", how="left",
    )
    merged["Severity_Score"] = merged["Severity_Score"].fillna(0.0)
    merged["Active_Cases"]   = merged["Active_Cases"].fillna(0).astype(int)
    merged["Disease_Type"]   = merged["Disease_Type"].fillna("None")
    return merged


def allocate_resources(route_ids: list, df: pd.DataFrame, van_capacity: int) -> pd.DataFrame:
    """
    Kit heuristic:
      Severity ≥ 8  → 50 kits  (critical)
      Severity ≥ 4  → 20 kits  (warning)
      else          → 10 kits  (stable)
    Capped by remaining van capacity.
    """
    rows, remaining, lookup = [], van_capacity, df.set_index("Village_ID")
    for step, vid in enumerate(route_ids):
        if vid == "V01" or vid not in lookup.index or remaining <= 0:
            continue
        r     = lookup.loc[vid]
        sev   = float(r["Severity_Score"])
        kits  = 50 if sev >= 8 else (20 if sev >= 4 else 10)
        kits  = min(kits, remaining)
        nurses = max(2, math.ceil(kits / 10))
        remaining -= kits
        rows.append({
            "Stop": step, "Village_ID": vid,
            "Village": r["Village_Name"], "Disease": r["Disease_Type"],
            "Cases": int(r["Active_Cases"]), "Severity": round(sev, 2),
            "🧴 Kits": kits, "👩‍⚕️ Nurses": nurses,
        })
    return pd.DataFrame(rows), van_capacity - remaining


# ══════════════════════════════════════════════════════════════════════════════
# MAP HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _sev_color(sev: float) -> str:
    if sev > 7:  return "red"
    if sev >= 4: return "orange"
    return "green"


def _pulse_icon() -> str:
    """CSS pulsing icon HTML for severity = 10 villages."""
    return (
        "<div class='pulse-wrapper'>"
        "<div class='pulse-ring'></div>"
        "<div class='pulse-ring'></div>"
        "<div class='pulse-core'></div>"
        "</div>"
    )


def build_map(df: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=10, tiles="CartoDB dark_matter",
    )
    for _, row in df.iterrows():
        sev = float(row["Severity_Score"])
        tip = (
            f"<div style='font-family:monospace;font-size:12px;min-width:160px;'>"
            f"<b style='color:#00aaff'>{row['Village_Name']}</b><br>"
            f"🦠 {row['Disease_Type']}<br>"
            f"🤒 Cases: {int(row['Active_Cases'])}<br>"
            f"⚠️ Severity: {sev:.2f}</div>"
        )
        if row["Village_ID"] == "V01":
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                tooltip=folium.Tooltip(tip),
            ).add_to(m)
        elif sev == 10.0:
            # Pulsing CSS animation for maximum-severity hotspots
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.DivIcon(
                    html=_pulse_icon(),
                    icon_size=(22, 22), icon_anchor=(11, 11),
                ),
                tooltip=folium.Tooltip(tip),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6 + sev * 0.65,
                color=_sev_color(sev), fill=True,
                fill_color=_sev_color(sev), fill_opacity=0.8,
                tooltip=folium.Tooltip(tip),
            ).add_to(m)
    return m

VAN_COLOURS = ["#1A73E8", "#9C27B0", "#E53935", "#2E7D32", "#F57C00"]

def draw_route(result: dict, folium_map: folium.Map, df: pd.DataFrame, depot_id: str = "V01"):
    """
    Draws multi-van routes on the existing Folium map without erasing hotspots.
    """
    route_ids = result.get("route_ids", {})
    
    # Backwards compatibility: if solver returns a single list instead of a dict
    if isinstance(route_ids, list):
        if not route_ids:
            return
        route_ids = {"Medical_Van_1": route_ids}

    # Build lat/lon lookup
    coord_map = {
        row["Village_ID"]: (float(row["Latitude"]), float(row["Longitude"]))
        for _, row in df.iterrows()
    }

    # Plot each van's route
    for van_idx, (van_name, village_sequence) in enumerate(route_ids.items()):
        colour = VAN_COLOURS[van_idx % len(VAN_COLOURS)]

        coords = [coord_map[vid] for vid in village_sequence if vid in coord_map]

        if len(coords) < 2:
            continue  # nothing to draw

        # Route polyline
        folium.PolyLine(
            locations=coords,
            color=colour,
            weight=5,
            opacity=0.9,
            tooltip=f"<b>{van_name}</b>",
        ).add_to(folium_map)

    # ── Legend (simple HTML overlay) ─────────────────────────────────────────
    if route_ids:
        legend_items = "".join(
            f'<li><span style="background:{VAN_COLOURS[i % len(VAN_COLOURS)]}; '
            f'width:14px;height:14px;display:inline-block;border-radius:3px;'
            f'margin-right:6px;vertical-align:middle;"></span>{van_name}</li>'
            for i, van_name in enumerate(route_ids.keys())
        )
        legend_html = f"""
        <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                    background:#0d1526;color:white;padding:10px 14px;border-radius:8px;
                    border:1px solid #1e3a5f;box-shadow:2px 2px 6px rgba(0,0,0,0.5);font-size:13px;font-family:sans-serif;">
          <b>🚑 Active Fleet</b><br>
          <ul style="list-style:none;margin:6px 0 0;padding:0;">{legend_items}</ul>
        </div>
        """
        folium_map.get_root().html.add_child(folium.Element(legend_html))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def chart_severity_bar(df: pd.DataFrame, visited: set, skipped: set) -> go.Figure:
    plot = df[df["Village_ID"] != "V01"].copy().sort_values("Severity_Score")

    def bar_clr(row):
        vid = row["Village_ID"]
        sev = row["Severity_Score"]
        if vid in visited: return CLR["visited"]
        if vid in skipped: return CLR["skipped"]
        if sev > 7:  return CLR["critical"]
        if sev >= 4: return CLR["warning"]
        return CLR["stable"]

    plot["clr"] = plot.apply(bar_clr, axis=1)

    fig = go.Figure(go.Bar(
        x=plot["Severity_Score"], y=plot["Village_Name"],
        orientation="h", marker_color=plot["clr"],
        customdata=plot[["Disease_Type", "Active_Cases"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Severity: %{x:.2f}<br>"
            "Disease: %{customdata[0]}<br>Cases: %{customdata[1]}<extra></extra>"
        ),
    ))

    # Legend entries
    for label, color in [("✅ Visited", CLR["visited"]), ("⏭ Skipped", CLR["skipped"]),
                          ("🔴 Critical", CLR["critical"]), ("🟠 Warning", CLR["warning"])]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        **CHART_BASE, height=500,
        title=dict(text="SEVERITY MATRIX — VISITED vs SKIPPED", font=dict(size=13, color="#00aaff")),
        xaxis=dict(title="Severity Score", range=[0, 10.5], gridcolor="#1e3a5f", color="#607a99"),
        yaxis=dict(title="", color="#607a99"),
        legend=dict(orientation="h", y=-0.1, font=dict(size=10)),
        bargap=0.28,
    )
    return fig


def chart_kits_pie(resource_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        resource_df, names="Village", values="🧴 Kits", hole=0.52,
        color_discrete_sequence=["#00aaff","#00ccff","#00ff88","#00cc66",
                                  "#0099ee","#00ffaa","#33bbff","#66ddff"],
        title="KIT DISTRIBUTION",
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Kits: %{value}<br>%{percent}<extra></extra>",
        textfont=dict(family="JetBrains Mono", size=10),
    )
    fig.update_layout(
        **CHART_BASE, showlegend=False,
        title_font=dict(size=13, color="#00aaff"), height=340,
    )
    return fig


def chart_gauge_time(total_time: int, max_mins: int) -> go.Figure:
    pct   = min(total_time / max_mins, 1.0)
    color = CLR["critical"] if pct > 0.9 else (CLR["warning"] if pct > 0.7 else CLR["visited"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_time,
        delta={"reference": max_mins, "decreasing": {"color": CLR["visited"]},
               "increasing": {"color": CLR["critical"]}, "valueformat": ".0f"},
        number={"suffix": " min", "font": {"size": 30, "family": "Rajdhani", "color": CLR["stable"]}},
        title={"text": "TIME EXPENDED", "font": {"size": 12, "color": "#00aaff", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, max_mins], "tickcolor": "#607a99", "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#0d1526",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, max_mins*0.7],  "color": "rgba(0,255,136,0.06)"},
                {"range": [max_mins*0.7, max_mins*0.9], "color": "rgba(255,140,0,0.06)"},
                {"range": [max_mins*0.9, max_mins],     "color": "rgba(255,51,85,0.06)"},
            ],
            "threshold": {"line": {"color": CLR["critical"], "width": 3},
                          "thickness": 0.8, "value": max_mins},
        },
    ))
    fig.update_layout(**CHART_BASE, height=240)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def run_solver(matrix_path, outbreak_path, max_time_mins):
    try:
        from hour4_data_model import build_data_model
        from hour5_vrp_solver import solve_routing

        # 1. Load current scenario data
        scenario_df = load_scenario(outbreak_path)

        # 2. Build the new V2.0 data model
        data = build_data_model(merged_df=scenario_df, num_vehicles=2) 
        
        # 3. Overwrite the Haversine matrix with the scenario matrix 
        # (Ensures the 2.5x Monsoon flood penalty still works!)
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        data["time_matrix"] = matrix_df.values.tolist()
        
        # Pass Severity Scores into the data dictionary for the solver penalty
        data["prizes"] = scenario_df["Severity_Score"].tolist()

        # 4. Run the multi-van fleet solver
        result = solve_routing(data=data, fleet_size=2, max_time=max_time_mins)
        
        # 5. Format the output so the UI KPI cards don't crash
        if result.get("status") == "SUCCESS":
            # Extract all unique villages visited across all vans
            visited_nodes = set([vid for route in result["route_ids"].values() for vid in route])
            visited_nodes.discard("V01") # Remove depot from counts
            
            # Calculate total mitigated severity
            mitigated = scenario_df[scenario_df["Village_ID"].isin(visited_nodes)]["Severity_Score"].sum()
            result["total_severity_mitigated"] = round(mitigated, 2)
            
            # Calculate skipped villages
            all_villages = set(scenario_df["Village_ID"]) - {"V01"}
            result["skipped_ids"] = list(all_villages - visited_nodes)
            
            return result
        else:
            return None

    except Exception as e:
        st.error(f"❌ Solver error: {e}")
        return None
# ══════════════════════════════════════════════════════════════════════════════
# LIVE LOGISTICS FEED
# ══════════════════════════════════════════════════════════════════════════════
def render_logistics_feed(route_ids: list, df: pd.DataFrame, resource_df: pd.DataFrame):
    lookup  = df.set_index("Village_ID")
    kit_map = (resource_df.set_index("Village_ID")["🧴 Kits"].to_dict()
               if not resource_df.empty and "Village_ID" in resource_df.columns else {})
    with st.expander("📡 LIVE LOGISTICS FEED", expanded=True):
        for step, vid in enumerate(route_ids):
            if vid not in lookup.index: continue
            name = lookup.loc[vid, "Village_Name"]
            if vid == "V01" and step == 0:
                st.markdown(f"`00` 🏠 **DEPART** — BHEL Base Camp")
            elif vid == "V01":
                st.markdown(f"`{step:02d}` 🏠 **RETURN** — BHEL Base Camp ✅")
            else:
                sev  = float(lookup.loc[vid, "Severity_Score"])
                icon = "🔴" if sev > 7 else ("🟠" if sev >= 4 else "🟢")
                kits = kit_map.get(vid, 0)
                st.markdown(f"`{step:02d}` {icon} **{name}** — drop `{kits}` kits")
        if not route_ids:
            st.caption("Run the solver to generate turn-by-turn directions.")


# ══════════════════════════════════════════════════════════════════════════════
# ██████  MAIN APP  ██████
# ══════════════════════════════════════════════════════════════════════════════

# ── Gate: file validation ──────────────────────────────────────────────────────
validate_files()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Rajdhani,sans-serif;color:#00aaff;"
        "letter-spacing:0.1em;margin-bottom:0;'>⚙ COMMAND PANEL</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e3a5f;margin:8px 0 16px;'>", unsafe_allow_html=True)

    scenario_label = st.selectbox(
        "ENVIRONMENTAL SCENARIO",
        options=list(SCENARIOS.keys()), index=0,
    )
    scenario = SCENARIOS[scenario_label]
    st.markdown(
        f"<div style='background:{scenario['color']}12;border-left:3px solid {scenario['color']};"
        f"padding:8px 12px;border-radius:8px;font-family:JetBrains Mono,monospace;"
        f"font-size:0.75rem;color:{scenario['color']};margin-bottom:12px;'>"
        f"[ {scenario['badge']} ] {scenario['desc']}</div>",
        unsafe_allow_html=True,
    )

    max_hours   = st.slider("MAX OPERATING TIME (hours)", 2, 12, 8, 1)
    max_mins    = max_hours * 60
    van_capacity = st.slider("VAN KIT CAPACITY (total kits)", 50, 500, 200, 10)
    st.caption(f"Budget: {max_mins} min · Capacity: {van_capacity} kits")

    # ── ML Prediction Toggle (Hour 12 intercept) ──────────────────────────────
    if ML_AVAILABLE:
        st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>🔮 ROUTING INTELLIGENCE</p>",
            unsafe_allow_html=True,
        )
        predictive_mode = st.toggle(
            "Predictive Mode (Tomorrow's Hotspots)", 
            value=False,
            help="Uses a Random Forest ML model to forecast tomorrow's outbreak severity."
        )
        if predictive_mode:
            st.info("🧠 **Proactive routing enabled.** Optimising for tomorrow.", icon="🔮")
    else:
        predictive_mode = False
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
    compute_btn = st.button("🚀 COMPUTE OPTIMAL ROUTE", type="primary", use_container_width=True)
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>Severity Legend</p>",
        unsafe_allow_html=True,
    )
    for label, color in [("● CRITICAL  Severity > 7", "#ff3355"),
                          ("● WARNING   Severity 4–7", "#ff8c00"),
                          ("● STABLE    No outbreak", "#00ff88"),
                          ("⌂ BASE CAMP  Depot", "#00aaff")]:
        st.markdown(
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;"
            f"color:{color};'>{label}</span>", unsafe_allow_html=True,
        )
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;'>
  <h1 style='font-family:Rajdhani,sans-serif;font-size:2.4rem;font-weight:700;
    color:#e8f4ff;letter-spacing:0.06em;margin:0;'>
    🏥 RURAL HEALTH LOGISTICS
  </h1>
  <span style='font-family:JetBrains Mono,monospace;font-size:0.8rem;
    color:#00aaff;letter-spacing:0.15em;'>COMMAND CENTER v9.0</span>
</div>
<p style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#607a99;
  letter-spacing:0.04em;margin-bottom:0;'>
  MOBILE CLINIC ROUTING OPTIMISER · HYDERABAD DISTRICT · TELANGANA
</p>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
try:
    df = load_scenario(scenario["outbreak"])
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()


# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k, v in [("solver_result", None), ("last_scenario", None),
              ("last_max_mins", None), ("last_capacity", None), 
              ("pred_df", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if (st.session_state.last_scenario != scenario_label or
        st.session_state.last_max_mins != max_mins or
        st.session_state.last_capacity != van_capacity):
    st.session_state.solver_result = None
    st.session_state.last_scenario = scenario_label
    st.session_state.last_max_mins = max_mins
    st.session_state.last_capacity = van_capacity

if compute_btn:
    outbreak_target = scenario["outbreak"]
    
    # ── Predictive AI Intercept Logic ──────────────────────────────────────────
    if predictive_mode:
        with st.spinner("🔮 Generating AI severity predictions for tomorrow..."):
            ml_pipeline = load_model()
            pred_df = predict_tomorrow(ml_pipeline, df)
            
            # Format dataframe to match exactly what the solver expects
            temp_outbreak = pred_df[["Village_ID", "Disease_Type", "Predicted_Active_Cases", "Predicted_Severity_T_plus_1"]].copy()
            temp_outbreak = temp_outbreak.rename(columns={
                "Predicted_Active_Cases": "Active_Cases",
                "Predicted_Severity_T_plus_1": "Severity_Score"
            })
            
            # Save temporary file for the backend OR-Tools solver to read
            temp_outbreak.to_csv("temp_predictive_outbreak.csv", index=False)
            outbreak_target = "temp_predictive_outbreak.csv"
            
            # Save the prediction DF to session state so we can display it in the UI later
            st.session_state.pred_df = pred_df 
    else:
        st.session_state.pred_df = None
    # ─────────────────────────────────────────────────────────────────────────

    with st.spinner("⚙ SOLVING PRIZE-COLLECTING VRP — GUIDED LOCAL SEARCH…"):
        st.session_state.solver_result = run_solver(
            matrix_path=scenario["matrix"],
            outbreak_path=outbreak_target,
            max_time_mins=max_mins,
        )
        
    # If using predictive mode, we update the primary `df` using the temp file 
    # so the Map and UI widgets accurately reflect the predicted values, not today's values.
    if predictive_mode:
        df = load_scenario("temp_predictive_outbreak.csv")

# If already solved in predictive mode, ensure the df stays updated for UI redrawing
elif st.session_state.pred_df is not None:
    df = load_scenario("temp_predictive_outbreak.csv")


result    = st.session_state.solver_result

# Handle backwards compatibility if solver returns dict or list
if result:
    if isinstance(result.get("route_ids"), dict):
        # Flatten all routes into one list for the kit allocation logic
        route_ids = [vid for route in result["route_ids"].values() for vid in route]
    else:
        route_ids = result.get("route_ids", [])
else:
    route_ids = []

# ── DERIVED DATA ───────────────────────────────────────────────────────────────
resource_df, kits_used = (
    allocate_resources(route_ids, df, van_capacity) if route_ids
    else (pd.DataFrame(), 0)
)
visited_ids   = set(route_ids[1:-1]) if len(route_ids) > 2 else set()
skipped_ids   = set(result["skipped_ids"]) if result else set()
visited_count = len([x for x in visited_ids if x != "V01"]) # avoid double counting depot
total_nodes   = len(df) - 1


# ── TOAST (mission success) ────────────────────────────────────────────────────
if result and compute_btn:
    if visited_count > 0:
        st.success(
            f"✅ MISSION SUCCESS — {visited_count} villages reached · "
            f"{result['total_severity_mitigated']} severity points mitigated · "
            f"{kits_used}/{van_capacity} kits deployed"
        )
    else:
        st.error(
            "⚠ NO FEASIBLE ROUTE FOUND — Try increasing the time limit or reducing the scenario load."
        )


# ── KPI METRIC CARDS ───────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
    "color:#607a99;letter-spacing:0.12em;text-transform:uppercase;'>[ MISSION KPIs ]</p>",
    unsafe_allow_html=True,
)
k1, k2, k3, k4, k5 = st.columns(5)

if result:
    hours_used   = result["total_time"] / 60
    coverage_pct = round(visited_count / total_nodes * 100, 1) if total_nodes > 0 else 0
    k1.metric("VILLAGES REACHED",  f"{visited_count}/{total_nodes}", f"{coverage_pct}% coverage")
    k2.metric("TIME EXPENDED",     f"{hours_used:.1f}h", f"{max_hours-hours_used:.1f}h remaining")
    k3.metric("IMPACT SCORE",      result["total_severity_mitigated"], "severity mitigated")
    k4.metric("KITS DEPLOYED",     f"{kits_used}/{van_capacity}", f"{len(skipped_ids)} vill. skipped", delta_color="inverse")
    k5.metric("SOLVER STATUS",     result["status"])
else:
    k1.metric("VILLAGES REACHED",  f"—/{total_nodes}")
    k2.metric("TIME EXPENDED",     f"—/{max_hours}h")
    k3.metric("IMPACT SCORE",      "—")
    k4.metric("KITS DEPLOYED",     f"—/{van_capacity}")
    k5.metric("SOLVER STATUS",     "STANDBY")


# ── KIT DEPLETION PROGRESS BAR (sidebar) ─────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>KIT DEPLETION</p>",
        unsafe_allow_html=True,
    )
    pct_used = kits_used / van_capacity if van_capacity > 0 else 0
    st.progress(pct_used)
    st.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#00aaff;'>"
        f"{kits_used} used · {van_capacity - kits_used} remaining</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # Logistics feed
    if route_ids:
        render_logistics_feed(route_ids, df, resource_df)
    else:
        with st.expander("📡 LIVE LOGISTICS FEED"):
            st.caption("Run the solver to generate turn-by-turn directions.")

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
    st.caption("RHLC Command Center · Phase 9")


st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)

# ── ML PREDICTION TABLE (Shows only if Predictive Mode is ON) ──────────────────
if st.session_state.pred_df is not None:
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
        "color:#00aaff;text-transform:uppercase;letter-spacing:0.1em;'>[ 🔮 AI FORECAST: TOMORROW'S PREDICTED HOTSPOTS ]</p>",
        unsafe_allow_html=True,
    )
    pred_display = st.session_state.pred_df[["Village_ID", "Village_Name", "Disease_Type", "Predicted_Severity_T_plus_1", "Predicted_Active_Cases"]].copy()
    pred_display = pred_display.sort_values("Predicted_Severity_T_plus_1", ascending=False).reset_index(drop=True)
    st.dataframe(pred_display, use_container_width=True, hide_index=True, height=180)
    st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)


# ── MAP + SNAPSHOT ─────────────────────────────────────────────────────────────
fmap = build_map(df)
if result:
    draw_route(result, fmap, df)

col_map, col_snap = st.columns([3, 1])

with col_map:
    badge_html = (
        f"<span style='background:{scenario['color']}22;border:1px solid {scenario['color']};"
        f"color:{scenario['color']};padding:3px 10px;border-radius:6px;"
        f"font-family:JetBrains Mono,monospace;font-size:0.72rem;letter-spacing:0.1em;'>"
        f"[ {scenario['badge']} ]</span>"
    )
    st.markdown(
        f"<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
        f"color:#607a99;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;'>"
        f"OUTBREAK MAP &nbsp; {badge_html}</p>",
        unsafe_allow_html=True,
    )
    st_folium(fmap, width=None, height=540, returned_objects=[])

with col_snap:
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>[ SYSTEM SNAPSHOT ]</p>",
        unsafe_allow_html=True,
    )
    active_ob = (df["Severity_Score"] > 0).sum()
    critical  = (df["Severity_Score"] > 7).sum()
    warning   = ((df["Severity_Score"] >= 4) & (df["Severity_Score"] <= 7)).sum()

    st.metric("VILLAGES TRACKED", total_nodes)
    st.metric("ACTIVE OUTBREAKS", int(active_ob))
    st.metric("CRITICAL ZONES",   int(critical))
    st.metric("WARNING ZONES",    int(warning))
    st.metric("TOTAL CASES",      int(df["Active_Cases"].sum()))

    st.markdown("<hr style='border-color:#1e3a5f;margin:10px 0;'>", unsafe_allow_html=True)
    ob_table = (
        df[df["Severity_Score"] > 0]
        [["Village_Name", "Disease_Type", "Active_Cases", "Severity_Score"]]
        .sort_values("Severity_Score", ascending=False).reset_index(drop=True)
        .rename(columns={"Village_Name": "Village", "Disease_Type": "Disease",
                          "Active_Cases": "Cases", "Severity_Score": "Sev"})
    )
    st.dataframe(ob_table, use_container_width=True, hide_index=True, height=265)


st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)


# ── ANALYTICS SECTION ──────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
    "color:#607a99;letter-spacing:0.12em;text-transform:uppercase;'>[ ANALYTICS DASHBOARD ]</p>",
    unsafe_allow_html=True,
)

if result and visited_count > 0:
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ SEVERITY MATRIX", "🧴 RESOURCE ALLOCATION", "⏱ TIME ANALYSIS", "📋 ROUTE MANIFEST"])

    with tab1:
        st.plotly_chart(chart_severity_bar(df, visited_ids, skipped_ids), use_container_width=True)

    with tab2:
        if not resource_df.empty:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(chart_kits_pie(resource_df), use_container_width=True)
                total_kits   = int(resource_df["🧴 Kits"].sum())
                total_nurses = int(resource_df["👩‍⚕️ Nurses"].sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("KITS OUT",    total_kits)
                m2.metric("NURSES",      total_nurses)
                m3.metric("REMAINING",   van_capacity - total_kits)
            with c2:
                st.markdown(
                    "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>"
                    "PER-VILLAGE BREAKDOWN</p>", unsafe_allow_html=True,
                )
                st.caption("Critical ≥8 → 50 kits · Warning ≥4 → 20 kits · Stable → 10 kits")
                st.dataframe(
                    resource_df[["Stop","Village","Disease","Cases","Severity","🧴 Kits","👩‍⚕️ Nurses"]],
                    use_container_width=True, hide_index=True, height=340,
                )

    with tab3:
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(chart_gauge_time(result["total_time"], max_mins), use_container_width=True)
        with g2:
            st.markdown(
                "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>"
                "MISSION TIME LOG</p>", unsafe_allow_html=True,
            )
            drive_h  = result["total_time"] / 60
            remain_h = max(max_hours - drive_h, 0)
            fig_donut = px.pie(
                pd.DataFrame({"Segment":["Drive","Remaining"],"h":[drive_h, remain_h]}),
                names="Segment", values="h", hole=0.58,
                color_discrete_sequence=["#ff8c00", "#1e3a5f"],
            )
            fig_donut.update_traces(textinfo="percent+label", textfont=dict(family="JetBrains Mono", size=11))
            fig_donut.update_layout(**CHART_BASE, showlegend=False, height=220, title="TIME UTILISATION")
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#e8f4ff;'>"
                f"🚗 Drive: <b style='color:#ff8c00'>{drive_h:.2f}h</b> ({result['total_time']} min)<br>"
                f"📅 Budget: <b style='color:#00aaff'>{max_hours}h</b> ({max_mins} min)<br>"
                f"⏳ Spare: <b style='color:#00ff88'>{remain_h:.2f}h</b>"
                f"</div>", unsafe_allow_html=True,
            )

    with tab4:
        lookup     = df.set_index("Village_ID")
        route_rows = []
        
        # Handle dict format for multi-van routes
        active_route_dict = result.get("route_ids", {})
        if isinstance(active_route_dict, list):
            active_route_dict = {"Medical_Van_1": active_route_dict}
            
        for van_name, sequence in active_route_dict.items():
            for step, vid in enumerate(sequence):
                if vid not in lookup.index: continue
                r   = lookup.loc[vid]
                sev = float(r["Severity_Score"])
                tag = "🏠 DEPOT" if vid == "V01" else (
                    "🔴 CRITICAL" if sev > 7 else ("🟠 WARNING" if sev >= 4 else "🟢 STABLE")
                )
                route_rows.append({
                    "VAN": van_name,
                    "STEP": "—" if vid == "V01" else step,
                    "ID": vid, "VILLAGE": r["Village_Name"],
                    "SEV": "—" if vid == "V01" else round(sev, 2), "STATUS": tag,
                })
                
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                "<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                "color:#607a99;text-transform:uppercase;letter-spacing:0.1em;'>"
                "OPTIMISED ROUTE SEQUENCE</p>", unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True, height=460)
        with c2:
            if skipped_ids:
                st.markdown(
                    f"<p style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"color:#ff3355;text-transform:uppercase;letter-spacing:0.1em;'>"
                    f"SKIPPED ({len(skipped_ids)} VILLAGES — TIME EXCEEDED)</p>",
                    unsafe_allow_html=True,
                )
                slookup = df.set_index("Village_ID")
                skip_rows = [
                    {"ID": v, "VILLAGE": slookup.loc[v, "Village_Name"],
                     "SEV": round(float(slookup.loc[v, "Severity_Score"]), 2),
                     "DISEASE": slookup.loc[v, "Disease_Type"]}
                    for v in sorted(skipped_ids) if v in slookup.index
                ]
                st.dataframe(pd.DataFrame(skip_rows), use_container_width=True, hide_index=True, height=460)
            else:
                st.success("✅ ALL VILLAGES REACHED WITHIN TIME BUDGET")

elif result and visited_count == 0:
    st.error(
        "⚠ NO FEASIBLE ROUTE FOUND within the current constraints.\n\n"
        "**Try:** ↑ Increase time budget · ↓ Switch to a lighter scenario · Check solver logs."
    )
else:
    st.markdown(
        "<div style='background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;"
        "padding:48px;text-align:center;margin-top:8px;'>"
        "<div style='font-size:2rem;margin-bottom:12px;'>🚀</div>"
        "<h3 style='font-family:Rajdhani,sans-serif;color:#00aaff;letter-spacing:0.08em;'>"
        "AWAITING SOLVER EXECUTION</h3>"
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#607a99;'>"
        "Select a scenario · Set time budget · Click COMPUTE OPTIMAL ROUTE</p>"
        "</div>",
        unsafe_allow_html=True,
    )