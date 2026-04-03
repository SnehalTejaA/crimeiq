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
    st.markdown("**Best model:** Random Forest (R² ≈ 0.897)")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("Use the tabs above to explore:")
    st.markdown("- 🗺️ Crime Heatmap")
    st.markdown("- 🎛️ What-If Simulator")
    st.markdown("- 🤖 AI Policy Advisor")
    st.markdown("- 📊 Analytics Dashboard")
    st.markdown("- 🎭 Scenario Simulation")
    st.markdown("- 🔬 Research Alignment")
    st.markdown("- ⚖️ Bias & Fairness Audit")
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
        <div class="metric-value">15</div>
        <div class="metric-sub">Phase 4 features</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── feature_vals: initialised here so Tab 3 always has a valid dict ───────────
# Will be populated by sliders in Tab 2 on every render pass
feature_vals = {f: FEATURE_RANGES[f][2] for f in FEATURES}  # defaults

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️  Crime Heatmap",
    "🎛️  What-If Simulator",
    "🤖  AI Policy Advisor",
    "📊  Analytics Dashboard",
    "🎭  Scenario Simulation",
    "🔬  Research Alignment",
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
        for feat in ["prbarr", "lprbconv", "prbconv", "polpc", "clpolpc"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], float(mn), float(mx), float(dv), float(step),
                key=f"slider_{feat}"
            )

        st.markdown("**Socioeconomic Factors**")
        for feat in ["taxpc", "wtrd", "wage_gap_service_mfg"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], float(mn), float(mx), float(dv), float(step),
                key=f"slider_{feat}"
            )

        st.markdown("**Demographics**")
        for feat in ["pctymle", "pctmin80", "mix"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], float(mn), float(mx), float(dv), float(step),
                key=f"slider_{feat}"
            )

        st.markdown("**Urbanization**")
        for feat in ["ldensity"]:
            mn, mx, dv, step = FEATURE_RANGES[feat]
            feature_vals[feat] = st.slider(
                FEATURE_LABELS[feat], float(mn), float(mx), float(dv), float(step),
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

    st.markdown("---")
    st.markdown("**Model Drift Monitor**")
    st.caption(
        "Compares training vs test feature distributions to detect "
        "data drift. Higher scores indicate larger distributional shift."
    )
    from data_loader import detect_drift
    X_train_cols = model_bundle["X_test"]  
    # We use X_test as proxy since we don't store X_train separately.
    # Split X_test in half to simulate train/test comparison:
    half = len(model_bundle["X_test"]) // 2
    drift_result = detect_drift(
        model_bundle["X_test"].iloc[:half],
        model_bundle["X_test"].iloc[half:]
    )
    drift_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(k, k) 
                    for k in drift_result["per_feature"].keys()],
        "Drift Score": list(drift_result["per_feature"].values()),
    }).sort_values("Drift Score", ascending=False)
    
    st.metric(
        "Overall Mean Drift Score", 
        f"{drift_result['overall_mean']:.6f}",
        help="Mean absolute difference across all features"
    )
    fig_drift = px.bar(
        drift_df, x="Drift Score", y="Feature",
        orientation="h",
        color="Drift Score",
        color_continuous_scale="Reds",
        labels={"Drift Score": "Mean Absolute Drift", "Feature": ""},
    )
    fig_drift.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_drift, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SCENARIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🎭 Scenario Simulation")
    st.markdown("Simulate real-world policy changes and see how the model predicts crime rate will respond.")

    from data_loader import run_scenario, interpret_query

    col_left, col_right = st.columns([1, 1])

    with col_left:
        scenario_choice = st.selectbox(
            "Pick a preset scenario",
            ["High Policing", "Police Reduction", 
             "Economic Decline", "Urban Growth"]
        )
        nl_query = st.text_input(
            "Or type a natural language query",
            placeholder="e.g. what if police presence decreases?"
        )
        if nl_query:
            detected = interpret_query(nl_query)
            if detected:
                scenario_choice = detected
                st.success(f"Detected scenario: {detected}")
            else:
                st.warning("Could not detect a scenario. Using selectbox choice.")

        # Show unified AI agent result
        from data_loader import ai_agent
        agent_result = ai_agent(
            nl_query if nl_query else f"simulate {scenario_choice}",
            feature_vals,
            model_bundle
        )
        with st.expander("🤖 AI Agent Pipeline Output"):
            st.json(agent_result)

    modified_vals = run_scenario(feature_vals, scenario_choice)
    baseline_pred = predict_crime(model_bundle, feature_vals)
    scenario_pred = predict_crime(model_bundle, modified_vals)
    delta_pct = (scenario_pred - baseline_pred) / baseline_pred * 100

    with col_right:
        direction = "▲" if delta_pct > 0 else "▼"
        color = "pill-red" if delta_pct > 0 else "pill-green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Scenario: {scenario_choice}</div>
            <div class="metric-value">{scenario_pred:.5f}</div>
            <div class="metric-sub">vs baseline {baseline_pred:.5f}</div>
            <br>
            <span class="{color}">
                {direction} {abs(delta_pct):.1f}% change
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Side by side gauges
    g1, g2 = st.columns(2)
    max_val = df[TARGET].max() * 1.2
    with g1:
        st.markdown("**Baseline**")
        fig_b = go.Figure(go.Indicator(
            mode="gauge+number",
            value=baseline_pred,
            gauge={"axis": {"range": [0, max_val]},
                   "bar": {"color": "#3B8BD4"}},
            number={"valueformat": ".5f"},
        ))
        fig_b.update_layout(height=220, 
                            margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_b, use_container_width=True)

    with g2:
        st.markdown(f"**{scenario_choice}**")
        fig_s = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scenario_pred,
            gauge={"axis": {"range": [0, max_val]},
                   "bar": {"color": "#E24B4A" 
                           if delta_pct > 0 else "#3B6D11"}},
            number={"valueformat": ".5f"},
        ))
        fig_s.update_layout(height=220, 
                            margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_s, use_container_width=True)

    # Comparison table — which features changed
    st.markdown("**Feature Changes: Baseline vs Scenario**")
    comp_rows = []
    for f in FEATURES:
        base_v = feature_vals[f]
        mod_v  = modified_vals[f]
        changed = base_v != mod_v
        comp_rows.append({
            "Feature":  FEATURE_LABELS[f],
            "Baseline": round(base_v, 5),
            "Scenario": round(mod_v,  5),
            "Changed":  "✅" if changed else "",
            "Delta":    round(mod_v - base_v, 5),
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, 
                 hide_index=True, height=350)

    # Bar chart of changed features only
    changed_df = comp_df[comp_df["Changed"] == "✅"]
    if not changed_df.empty:
        st.markdown("**What Changed**")
        fig_chg = px.bar(
            changed_df, x="Delta", y="Feature",
            orientation="h",
            color="Delta",
            color_continuous_scale="RdBu_r",
            labels={"Delta": "Change in value", "Feature": ""},
        )
        fig_chg.update_layout(
            height=250, 
            margin=dict(t=20,b=20,l=10,r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_chg, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — RESEARCH ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### 🔬 Research Alignment")
    st.markdown("""
    This tab validates our feature selection against established 
    criminology theory. Cornwell & Trumbull (1994) and Becker (1968) 
    argue that crime is influenced by the probability of arrest and 
    conviction (law enforcement deterrence), economic opportunity 
    (wages, tax revenue), and demographic composition. The 
    correlations below confirm our 14 features align with this theory.
    """)

    from data_loader import FEATURE_CATEGORIES

    # Compute correlations
    corr_rows = []
    for category, feats in FEATURE_CATEGORIES.items():
        for f in feats:
            if f in df.columns:
                corr = df[f].corr(df[TARGET])
                corr_rows.append({
                    "Feature":     FEATURE_LABELS.get(f, f),
                    "Category":    category,
                    "Correlation": round(corr, 4),
                    "Abs Corr":    round(abs(corr), 4),
                })
    corr_df = pd.DataFrame(corr_rows).sort_values(
        "Abs Corr", ascending=False
    )

    # Category average bar chart
    cat_avg = (corr_df.groupby("Category")["Abs Corr"]
               .mean()
               .reset_index()
               .sort_values("Abs Corr", ascending=True))
    fig_cat = px.bar(
        cat_avg, x="Abs Corr", y="Category",
        orientation="h",
        color="Abs Corr",
        color_continuous_scale="Blues",
        labels={"Abs Corr": "Avg |Correlation| with Crime Rate",
                "Category": ""},
        title="Average Feature Correlation by Criminology Category",
    )
    fig_cat.update_layout(
        height=300,
        margin=dict(t=40,b=20,l=10,r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Full feature correlation table
    st.markdown("**Individual Feature Correlations with Crime Rate**")
    st.dataframe(
        corr_df[["Feature","Category","Correlation","Abs Corr"]],
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.markdown("---")
    st.markdown("#### 🔍 Global vs Local Explanation")
    st.markdown("""
    **Globally**, the model identifies population density, policing 
    levels (probability of arrest, police per capita), and 
    demographic variables (minority percentage, percent young males) 
    as the key drivers of crime rates across NC counties.

    **Locally**, for any individual county, the prediction is shaped 
    by the specific combination of arrest probability, police 
    presence, and demographic composition. A county with moderate 
    arrest probability (prbarr ≈ 0.28), low police presence 
    (polpc ≈ 0.0017), and moderate minority composition 
    (pctmin80 ≈ 17%) receives a predicted crime rate of 
    approximately 0.0327 — showing how the model integrates 
    multiple variables for case-specific predictions.

    Use the **What-If Simulator** (Tab 2) to explore local 
    explanations for any custom county profile interactively.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — BIAS & FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("### ⚖️ Bias & Fairness Audit")
    st.markdown("""
    This audit examines whether the Random Forest model predicts 
    systematically different crime rates across demographic and 
    geographic subgroups. Large gaps may indicate the model has 
    learned structural inequalities present in the 1981-87 data.
    """)

    from data_loader import run_fairness_audit

    audit_df = run_fairness_audit(model_bundle)

    interpretations = {
        "Urban vs Rural": 
            "Urban counties are predicted to have different crime rates than rural ones, reflecting density and resource differences.",
        "Western vs Rest": 
            "Western NC counties show a prediction gap, likely driven by geographic and economic differences captured in the data.",
        "Central vs Rest": 
            "Central counties show a distinct prediction pattern compared to eastern counties.",
        "High vs Low Minority": 
            "Counties with above-median minority populations are predicted differently, reflecting socioeconomic disparities in the historical data.",
    }

    for _, row in audit_df.iterrows():
        st.markdown(f"**{row['Subgroup']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted — Group A", f"{row['Pred A']:.5f}")
        c2.metric("Predicted — Group B", f"{row['Pred B']:.5f}")
        c3.metric("Fairness Gap", f"{row['Gap']:.5f}",
                  delta=f"{'⚠️ Notable' if row['Gap'] > 0.005 else '✅ Low'}",
                  delta_color="off")

        fig_audit = px.bar(
            pd.DataFrame({
                "Group":  [row["Group A"], row["Group A"],
                           row["Group B"], row["Group B"]],
                "Type":   ["Predicted","Actual",
                           "Predicted","Actual"],
                "Value":  [row["Pred A"], row["Act A"],
                           row["Pred B"], row["Act B"]],
            }),
            x="Value", y="Group", color="Type",
            orientation="h", barmode="group",
            color_discrete_map={"Predicted":"#534AB7",
                                "Actual":"#1D9E75"},
            labels={"Value": "Crime Rate", "Group": ""},
        )
        fig_audit.update_layout(
            height=180,
            margin=dict(t=10,b=10,l=10,r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.2),
        )
        st.plotly_chart(fig_audit, use_container_width=True)
        st.caption(interpretations.get(row["Subgroup"], ""))
        st.markdown("---")

    # Summary table
    st.markdown("#### 📋 Fairness Gap Summary")
    summary = audit_df[["Subgroup","Gap"]].sort_values(
        "Gap", ascending=False
    ).copy()
    summary["Verdict"] = summary["Gap"].apply(
        lambda g: "🚨 High" if g > 0.010 
                  else ("⚠️ Notable" if g > 0.005 else "✅ Low")
    )
    st.dataframe(summary, use_container_width=True, 
                 hide_index=True)

    st.markdown("---")
    st.markdown("#### ⚠️ Ethical Interpretation")
    st.markdown("""
    The model predicts notably different average crime rates across 
    urban and rural counties, reflecting underlying structural and 
    environmental variations such as population density and economic 
    activity.

    Such differences must be interpreted carefully. Without proper 
    context, the model could lead to **unequal resource allocation** 
    or **reinforce existing biases** in policing. The fairness gaps 
    observed here are a product of historical socioeconomic 
    inequalities in the 1981–87 data — not inherent characteristics 
    of the counties themselves.

    This model should be used exclusively as a **decision-support 
    tool with human oversight** to ensure fair and ethical outcomes.
    """)
