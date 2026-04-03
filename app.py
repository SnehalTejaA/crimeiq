import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings("ignore")

from data_loader import (
    load_data, train_model, predict_crime,
    get_shap_for_input, get_cluster_profiles,
    FEATURES, FEATURE_LABELS, FEATURE_RANGES, TARGET, DATA_URL
)
from heatmap import build_heatmap
from llm_policy import generate_policy_recommendations, generate_cluster_narrative

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrimeIQ — NC Crime Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-title { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
  .metric-value { font-size: 28px; font-weight: 600; color: #212529; }
  .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 4px; }
  .pill-red    { background:#fde8e8; color:#a32d2d; border-radius:20px; padding:3px 12px; font-size:12px; }
  .pill-green  { background:#eaf3de; color:#3b6d11; border-radius:20px; padding:3px 12px; font-size:12px; }
  .pill-amber  { background:#faeeda; color:#854f0b; border-radius:20px; padding:3px 12px; font-size:12px; }
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/CrimeIQ-NC%20Intelligence-blue?style=for-the-badge", width=220)
    st.markdown("### 🔍 CrimeIQ")
    st.caption("AI-Powered Urban Crime Intelligence Platform")
    st.markdown("---")
    st.markdown("**Dataset:** Cornwell & Trumbull NC (1981–87)")
    st.markdown("**Counties:** 90 | **Observations:** 630")
    st.markdown("**Best model:** Random Forest (R² ≈ 0.905)")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("Use the tabs above to explore:")
    st.markdown("- 🗺️ Crime Heatmap")
    st.markdown("- 🎛️ What-If Simulator")
    st.markdown("- 🤖 AI Policy Advisor")
    st.markdown("- 📊 Analytics Dashboard")
    st.markdown("---")
    st.caption("Built for DTSC 5082 · Spring 2026 · Group 1")

# ── Load data & model (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Training Random Forest model…")
def get_model(data_hash):
    df = get_data()
    return train_model(df)

df = get_data()
model_bundle = get_model(hash(str(df.shape)))
baseline_crime = float(df[TARGET].mean())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 CrimeIQ — NC Crime Intelligence Platform")
st.markdown("*Analysing the Impact of Socioeconomic and Law Enforcement Factors on Crime Rates Across North Carolina Counties*")

# Top-level KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Mean Crime Rate</div>
        <div class="metric-value">{baseline_crime:.4f}</div>
        <div class="metric-sub">crmrte (crimes per person)</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Model R²</div>
        <div class="metric-value">{model_bundle['r2']}</div>
        <div class="metric-sub">Random Forest</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Counties</div>
        <div class="metric-value">90</div>
        <div class="metric-sub">NC counties, 1981–1987</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Features Used</div>
        <div class="metric-value">14</div>
        <div class="metric-sub">VIF-selected (all &lt; 10)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── feature_vals: initialised here so Tab 3 always has a valid dict ───────────
# Will be populated by sliders in Tab 2 on every render pass
feature_vals = {f: FEATURE_RANGES[f][2] for f in FEATURES}  # defaults

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️  Crime Heatmap",
    "🎛️  What-If Simulator",
    "🤖  AI Policy Advisor",
    "📊  Analytics Dashboard",
    "🎭  Scenario Simulation",
    "⚖️  Bias & Fairness Audit",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CRIME HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🗺️ Crime Rate Heatmap — North Carolina Counties")
    st.markdown("Hover over any county marker for its crime rate. Use the year filter to see changes over time.")

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        year_choice = st.selectbox(
            "Filter by year",
            options=["All years (1981–87)"] + sorted(df["year"].unique().tolist()),
            index=0,
        )
        year_val = None if "All" in str(year_choice) else int(year_choice)

        show_raw = st.checkbox("Show county stats table", value=False)

    with col_info:
        year_data = df if year_val is None else df[df["year"] == year_val]
        high_crime = year_data.groupby("county")[TARGET].mean().idxmax()
        low_crime  = year_data.groupby("county")[TARGET].mean().idxmin()
        mean_yr    = year_data[TARGET].mean()
        st.info(
            f"**Period mean crime rate:** {mean_yr:.4f}  \n"
            f"**Highest crime county:** #{high_crime}  \n"
            f"**Lowest crime county:** #{low_crime}"
        )

    # Crime rate bubble map using Plotly (reliable cross-version)
    from heatmap import NC_COUNTY_COORDS
    county_agg = (
        year_data.groupby("county")[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "mean_crmrte"})
    )
    county_agg["lat"] = county_agg["county"].map(lambda c: NC_COUNTY_COORDS.get(c, (35.5, -79.5))[0])
    county_agg["lon"] = county_agg["county"].map(lambda c: NC_COUNTY_COORDS.get(c, (35.5, -79.5))[1])
    county_agg["label"] = "County " + county_agg["county"].astype(str)

    fig_map = px.scatter_mapbox(
        county_agg,
        lat="lat", lon="lon",
        size="mean_crmrte",
        color="mean_crmrte",
        color_continuous_scale="Reds",
        hover_name="label",
        hover_data={"mean_crmrte": ":.5f", "lat": False, "lon": False},
        size_max=30,
        zoom=6,
        center={"lat": 35.5, "lon": -79.5},
        mapbox_style="carto-positron",
        labels={"mean_crmrte": "Crime Rate"},
    )
    fig_map.update_layout(
        height=500,
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title="Crime Rate"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    if show_raw:
        st.dataframe(
            year_data.groupby("county")[TARGET]
            .mean()
            .reset_index()
            .rename(columns={TARGET: "mean_crmrte"})
            .sort_values("mean_crmrte", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=300,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🎛️ What-If Crime Rate Simulator")
    st.markdown("Adjust the socioeconomic and law-enforcement sliders to see how the model predicts crime rate changes.")

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown("**Law Enforcement Factors**")
        for feat in ["prbarr", "prbconv", "prbpris", "avgsen", "polpc"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], mn, mx, dv, step,
                key=f"slider_{feat}"
            )

        st.markdown("**Socioeconomic Factors**")
        for feat in ["density", "taxpc", "pctmin80", "wcon", "wtuc", "wfed"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], mn, mx, dv, step,
                key=f"slider_{feat}"
            )

        st.markdown("**Regional Indicators**")
        for feat in ["west", "central", "urban"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.select_slider(
                FEATURE_LABELS[feat],
                options=[0, 1],
                value=int(dv),
                key=f"slider_{feat}"
            )

    with col_result:
        predicted = predict_crime(model_bundle, feature_vals)
        delta = predicted - baseline_crime
        delta_pct = delta / baseline_crime * 100
        pill_class = "pill-red" if delta > 0 else "pill-green"
        direction_label = f"▲ +{delta_pct:.1f}% above average" if delta > 0 else f"▼ {delta_pct:.1f}% below average"

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px">
            <div class="metric-title">Predicted Crime Rate</div>
            <div class="metric-value">{predicted:.5f}</div>
            <div class="metric-sub">crmrte (crimes per person)</div>
            <br>
            <span class="{pill_class}">{direction_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted,
            delta={"reference": baseline_crime, "valueformat": ".5f"},
            title={"text": "vs. State Average"},
            gauge={
                "axis": {"range": [0, df[TARGET].max() * 1.2]},
                "bar":  {"color": "#E24B4A" if delta > 0 else "#3B8BD4"},
                "steps": [
                    {"range": [0, baseline_crime * 0.75], "color": "#eaf3de"},
                    {"range": [baseline_crime * 0.75, baseline_crime * 1.25], "color": "#faeeda"},
                    {"range": [baseline_crime * 1.25, df[TARGET].max() * 1.2], "color": "#fde8e8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": baseline_crime,
                },
            },
            number={"valueformat": ".5f"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP waterfall for this prediction
        st.markdown("**What's driving this prediction?**")
        shap_vals = get_shap_for_input(model_bundle, feature_vals)
        shap_df = (
            pd.DataFrame({"feature": list(shap_vals.keys()), "shap": list(shap_vals.values())})
            .assign(feature=lambda x: x["feature"].map(FEATURE_LABELS))
            .sort_values("shap", key=abs, ascending=True)
        )
        colors = ["#E24B4A" if v > 0 else "#3B8BD4" for v in shap_df["shap"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig_shap.update_layout(
            xaxis_title="SHAP value (impact on crime rate)",
            height=420,
            margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AI POLICY ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 AI Policy Advisor")
    st.markdown(
        "Powered by **Claude (Anthropic)**. The AI analyses your SHAP feature drivers "
        "and generates specific, evidence-based policy recommendations for the county profile "
        "you configured in the What-If Simulator."
    )

    cluster_profiles = get_cluster_profiles(df)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        cluster_id = st.selectbox(
            "Select county cluster",
            options=[0, 1, 2],
            format_func=lambda c: {
                0: "Cluster 0 — High crime / low policing",
                1: "Cluster 1 — Moderate crime / avg resources",
                2: "Cluster 2 — Low crime / well-resourced",
            }[c],
        )
        run_llm = st.button("🤖 Generate Policy Recommendations", type="primary", use_container_width=True)

    with col_b:
        st.markdown("**Current what-if profile** (from Simulator tab)")
        if feature_vals:
            display_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURES],
                "Value":   [round(feature_vals.get(f, 0), 4) for f in FEATURES],
            })
            st.dataframe(display_df, use_container_width=True, height=250, hide_index=True)
        else:
            st.info("Configure sliders in the What-If Simulator tab first.")

    if run_llm:
        with st.spinner("Claude is analysing the data and writing recommendations…"):
            shap_vals = get_shap_for_input(model_bundle, feature_vals)
            predicted_now = predict_crime(model_bundle, feature_vals)
            rec_text = generate_policy_recommendations(
                shap_dict=shap_vals,
                cluster_id=cluster_id,
                predicted_crime=predicted_now,
                baseline_crime=baseline_crime,
                feature_labels=FEATURE_LABELS,
            )
        st.markdown("---")
        st.markdown("#### 📋 Policy Report")
        st.markdown(rec_text)

    st.markdown("---")
    st.markdown("#### 🗂️ Cluster Profiles (K-Means, k=3)")
    st.dataframe(
        cluster_profiles.rename(columns=FEATURE_LABELS).T,
        use_container_width=True,
        height=320,
    )

    if st.button("🤖 Generate Cluster Narrative", use_container_width=False):
        with st.spinner("Generating cluster analysis…"):
            narrative = generate_cluster_narrative(cluster_profiles)
        st.markdown(narrative)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Crime rate trend over years
    with col1:
        st.markdown("**Crime Rate Trend (1981–1987)**")
        trend = df.groupby("year")[TARGET].mean().reset_index()
        fig_trend = px.line(
            trend, x="year", y=TARGET,
            markers=True,
            labels={TARGET: "Mean Crime Rate", "year": "Year"},
            color_discrete_sequence=["#E24B4A"],
        )
        fig_trend.update_layout(
            height=280, margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Feature importance from RF
    with col2:
        st.markdown("**Feature Importances (Random Forest)**")
        fi = pd.Series(
            model_bundle["model"].feature_importances_,
            index=[FEATURE_LABELS[f] for f in FEATURES]
        ).sort_values(ascending=True)
        fig_fi = px.bar(
            fi, orientation="h",
            labels={"value": "Importance", "index": ""},
            color=fi.values,
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            height=280, margin=dict(t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    col3, col4 = st.columns(2)

    # SHAP summary (mean absolute)
    with col3:
        st.markdown("**Mean |SHAP| — Global Feature Impact**")
        mean_shap = np.abs(model_bundle["shap_values"]).mean(axis=0)
        shap_summary = pd.Series(
            mean_shap, index=[FEATURE_LABELS[f] for f in FEATURES]
        ).sort_values(ascending=True)
        fig_ms = px.bar(
            shap_summary, orientation="h",
            labels={"value": "Mean |SHAP|", "index": ""},
            color=shap_summary.values,
            color_continuous_scale="Oranges",
        )
        fig_ms.update_layout(
            height=280, margin=dict(t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ms, use_container_width=True)

    # Crime rate distribution
    with col4:
        st.markdown("**Crime Rate Distribution**")
        fig_dist = px.histogram(
            df, x=TARGET, nbins=40,
            labels={TARGET: "Crime Rate"},
            color_discrete_sequence=["#534AB7"],
        )
        fig_dist.add_vline(
            x=baseline_crime, line_dash="dash",
            line_color="#E24B4A", annotation_text="Mean",
        )
        fig_dist.update_layout(
            height=280, margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Scatter: density vs crime
    st.markdown("**Population Density vs Crime Rate**")
    fig_scatter = px.scatter(
        df, x="density", y=TARGET,
        color="year",
        opacity=0.6,
        labels={"density": "Population Density", TARGET: "Crime Rate", "year": "Year"},
        color_continuous_scale="Viridis",
        trendline="ols",
    )
    fig_scatter.update_layout(
        height=350, margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Model performance summary
    st.markdown("---")
    st.markdown("**Model Performance Summary**")
    perf_data = {
        "Model": ["Random Forest", "Gradient Boosting", "Ridge Regression"],
        "R²":    [model_bundle["r2"], 0.884, 0.712],
        "RMSE":  [model_bundle["rmse"], 0.000182, 0.000310],
        "Status": ["✅ Best (used)", "🟡 Runner-up", "🔵 Baseline"],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SCENARIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🎭 Scenario Simulation")
    st.markdown("Preset scenarios simulating systemic socioeconomic or law enforcement shifts.")

    scenarios = {
        "High Policing": {"polpc": 1.3, "prbarr": 1.2},
        "Police Reduction": {"polpc": 0.7, "prbarr": 0.8},
        "Economic Decline": {"taxpc": 0.7, "wcon": 0.9, "wtuc": 0.9},
        "Urban Growth": {"density": 1.2, "urban": "set_1"}
    }
    
    scenario_choice = st.selectbox("Select a scenario to simulate:", list(scenarios.keys()), key="scenario_select")
    mods = scenarios[scenario_choice]
    
    scen_vals = feature_vals.copy()
    diffs = {}
    for f, k in mods.items():
        old = scen_vals[f]
        if k == "set_1":
            new = 1.0
        else:
            new = old * k
        # Clamp within valid trained ranges
        mn, mx, _, _ = FEATURE_RANGES[f]
        new = max(mn, min(mx, new))
        scen_vals[f] = new
        
        # Track diff for the bar chart
        if old != 0:
            pct_diff = ((new - old) / old) * 100
        else:
            pct_diff = (new * 100)
        diffs[FEATURE_LABELS[f]] = pct_diff

    # Make predictions
    b_pred = predict_crime(model_bundle, feature_vals)
    s_pred = predict_crime(model_bundle, scen_vals)
    d_pred = s_pred - b_pred
    d_pct = (d_pred / b_pred * 100) if b_pred != 0 else 0

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("**Feature Modifications**")
        rows = []
        for f in FEATURES:
            changed = "Yes" if f in mods else "No"
            rows.append({
                "Feature": FEATURE_LABELS[f],
                "Baseline": feature_vals[f],
                "Scenario": scen_vals[f],
                "Changed": changed
            })
        df_comp = pd.DataFrame(rows)
        
        def highlight_amber(row):
            return ['background-color: #faeeda; color: black' if row['Changed'] == 'Yes' else '' for _ in row]
            
        st.dataframe(
            df_comp.style.apply(highlight_amber, axis=1),
            use_container_width=True, hide_index=True
        )

        if diffs:
            df_bar = pd.DataFrame(list(diffs.items()), columns=["Feature", "% Change"])
            fig_bar = px.bar(df_bar, x="% Change", y="Feature", orientation="h", title="Modifiers Applied")
            fig_bar.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("**Crime Rate Impact**")
        d1, d2 = st.columns(2)
        with d1:
            fg1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=b_pred,
                title={"text": "Baseline Rate"},
                gauge={"axis": {"range": [0, df[TARGET].max()]}, "bar": {"color": "#6c757d"}}
            ))
            fg1.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fg1, use_container_width=True)
            
        with d2:
            fg2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=s_pred,
                delta={"reference": b_pred, "valueformat": ".5f"},
                title={"text": "Scenario Rate"},
                gauge={"axis": {"range": [0, df[TARGET].max()]}, "bar": {"color": "#E24B4A" if d_pred > 0 else "#3B8BD4"}}
            ))
            fg2.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fg2, use_container_width=True)

        pill_clr = "pill-red" if d_pred > 0 else "pill-green"
        dir_lbl = f"▲ +{d_pct:.1f}% increase" if d_pred > 0 else f"▼ {abs(d_pct):.1f}% decrease"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Delta</div>
            <div class="metric-value" style="font-size:20px;">{d_pred:+.5f}</div>
            <span class="{pill_clr}">{dir_lbl}</span>
        </div>''', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — BIAS & FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### ⚖️ Bias & Fairness Audit")
    st.markdown("Analyzing predictive fairness across key subgroups in the test set.")
    
    # Extract test data
    X_t = model_bundle["X_test"]
    y_t = model_bundle["y_test"].values
    mod = model_bundle["model"]
    scl = model_bundle["scaler"]
    
    p_t = mod.predict(scl.transform(X_t))
    
    adf = X_t.copy()
    adf["actual"] = y_t
    adf["predicted"] = p_t
    
    med_min = adf["pctmin80"].median()
    
    groups = [
        ("Urban vs Rural", "urban", 1, 0, "Urban", "Rural"),
        ("Western vs Non-western", "west", 1, 0, "Western", "Non-western"),
        ("Central vs Non-central", "central", 1, 0, "Central", "Non-central"),
        ("High vs Low Minority", "pctmin80", "high", "low", "> Median", "<= Median")
    ]
    
    res = []
    
    for concept, col, v1, v2, l1, l2 in groups:
        if col == "pctmin80":
            m1 = adf[col] > med_min
            m2 = adf[col] <= med_min
        else:
            m1 = adf[col] == v1
            m2 = adf[col] == v2
            
        p1 = adf[m1]["predicted"].mean() if m1.sum() > 0 else 0
        a1 = adf[m1]["actual"].mean() if m1.sum() > 0 else 0
        p2 = adf[m2]["predicted"].mean() if m2.sum() > 0 else 0
        a2 = adf[m2]["actual"].mean() if m2.sum() > 0 else 0
        
        g = abs(p1 - p2)
        
        sent = (f"The model predicts an average crime rate of {p1:.4f} for {l1} counties, "
                f"compared to {p2:.4f} for {l2} counties—a gap of {g:.4f}.")
                
        res.append({
            "Comparison": concept,
            "Group 1": l1, "Pred 1": p1, "Act 1": a1,
            "Group 2": l2, "Pred 2": p2, "Act 2": a2,
            "Gap": g,
            "Sentence": sent
        })

    for r in res:
        with st.expander(f"{r['Comparison']} — Gap: {r['Gap']:.5f}", expanded=False):
            ca, cb = st.columns([1, 2])
            with ca:
                st.markdown(f"**{r['Group 1']}**\n- Predicted: {r['Pred 1']:.5f}\n- Actual: {r['Act 1']:.5f}")
                st.markdown(f"**{r['Group 2']}**\n- Predicted: {r['Pred 2']:.5f}\n- Actual: {r['Act 2']:.5f}")
                st.info(r['Sentence'])
            with cb:
                plotdf = pd.DataFrame({
                    "Subgroup": [r['Group 1'], r['Group 1'], r['Group 2'], r['Group 2']],
                    "Metric": ["Predicted", "Actual", "Predicted", "Actual"],
                    "Crime Rate": [r['Pred 1'], r['Act 1'], r['Pred 2'], r['Act 2']]
                })
                fig_f = px.bar(plotdf, y="Subgroup", x="Crime Rate", color="Metric", 
                               barmode="group", orientation="h",
                               color_discrete_sequence=["#3B8BD4", "#6c757d"])
                fig_f.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=10))
                st.plotly_chart(fig_f, use_container_width=True)
                
    st.markdown("#### Fairness Gap Summary")
    summ = pd.DataFrame(res)[["Comparison", "Group 1", "Group 2", "Gap"]].sort_values("Gap", ascending=False)
    st.dataframe(summ, use_container_width=True, hide_index=True)
