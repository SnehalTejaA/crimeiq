# 🔍 CrimeIQ — NC Crime Intelligence Platform

> AI-Powered Urban Crime Intelligence Platform  
> DTSC 5082 · Spring 2026 · Group 1  
> University of North Texas

---

## What This Does

CrimeIQ integrates Phase 3 analytical work with Phase 4 real-world deployment features including scenario simulation, research alignment validation, and a bias & fairness audit.

### 7 Pillars

| Tab | What it shows |
|-----|--------------|
| 🗺️ Crime Heatmap | Folium choropleth of NC crime rates by county, filterable by year |
| 🎛️ What-If Simulator | Sliders for every feature → live predicted crime rate + SHAP waterfall |
| 🤖 AI Policy Advisor | Claude API (Anthropic) generates evidence-based policy recommendations from SHAP drivers |
| 📊 Analytics Dashboard | Trend charts, feature importance, SHAP summary, model performance |
| 🎭 Scenario Simulation | Preset scenarios simulating systemic socioeconomic or law enforcement shifts |
| 🔬 Research Alignment | Validates feature selection against established criminology theory |
| ⚖️ Bias & Fairness Audit | Analyzes predictive fairness across key demographic sub-groups |

---

## Setup

### 1. Clone / download
```bash
git clone https://github.com/SnehalTejaA/crimeiq-v2.git
cd crimeiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app loads data directly from GitHub on first run (cached after that).

---

## Project Structure

```text
crimeiq/
├── app.py            # Main Streamlit app — all 7 tabs
├── data_loader.py    # Data loading, feature engineering, 
│                     # model training, scenario simulation,
│                     # fairness audit, drift detection
├── heatmap.py        # NC county coordinates for Plotly map
├── llm_policy.py     # Groq API integration for policy recommendations
├── requirements.txt
└── README.md
```

---

## Dataset

**Cornwell & Trumbull NC County Crime Dataset (1981–1987)**
- 630 observations · 90 counties · 7 years
- Source: [GitHub repo](https://github.com/nishapattim05-del/crime-project-data)

**15 Features used (14 VIF-selected + Phase 4 extensions):**
`prbarr`, `prbconv`, `prbpris`, `avgsen`, `polpc`, `density`, `taxpc`,
`west`, `central`, `urban`, `pctmin80`, `wcon`, `wtuc`, `wfed`, `wage_gap_service_mfg`

**Target variable:** `crmrte` (crimes per person)

---

## Model

The Random Forest Regressor is trained fresh on app startup (cached via
`@st.cache_resource`). Performance mirrors Phase 3 findings:

| Metric | Value |
|--------|-------|
| R²     | ~0.905 |
| RMSE   | ~0.000178 |

---

## Phase 4 Extensions

- **Scenario Simulation Engine:** Compares custom baseline feature inputs against 4 predefined systemic scenarios (or Natural Language defined scenarios).
- **Research Alignment:** Evaluates variables against Becker (1968) and Cornwell & Trumbull (1994) foundational principles of deterrence and economic opportunity.
- **Bias & Fairness Audit:** Evaluates predictive differences across 4 subgroups (e.g. Urban vs Rural, High vs Low Minority).
- **Model Drift Monitor:** Computes and visualizes out-of-distribution shifts between continuous feature vectors using Mean Absolute Difference.
- **AI Agent Pipeline:** A unified backend python wrapper that intercepts user queries, detects the requisite scenario automatically, simulates it against the Random Forest, and returns the unified response dictionary.

---

## API Deployment Concept

POST `/predict`
Input:  `{ "prbarr": 0.3, "polpc": 0.002, ... all 14 features }`
Output: `{ "predicted_crime_rate": 0.032 }`

POST `/simulate`
Input:  `{ "scenario": "High Policing", "features": {...} }`
Output: `{ "baseline": 0.031, "scenario": 0.028, "delta_pct": -9.7 }`

POST `/agent`
Input:  `{ "query": "what if police presence decreases?", "features": {...} }`
Output: `{ "scenario_detected": "Police Reduction", "predicted_crime_rate": 0.034 }`

---

## Production Readiness

The model supports two deployment modes:

**1. Batch Processing**
The system can run periodically (daily or weekly) to generate crime
predictions for all counties and support long-term planning. Outputs
are stored and used for trend analysis and resource allocation across
law enforcement agencies.

**2. Real-Time Prediction**
The model is integrated into an API where users provide inputs such
as police levels or economic conditions and receive instant
predictions. This supports operational decision-making and live
scenario simulation through the CrimeIQ interface.

This flexibility allows the model to support both strategic planning
and real-time decision-making. The model achieved R²≈0.905,
indicating strong predictive performance. To ensure long-term
reliability, data drift detection is built in (Tab 4 — Analytics
Dashboard) to identify changes in input distributions over time.
If performance degrades, the model can be retrained with updated
data.

---

## Team

| Member | Role |
|--------|------|
| Snehal Teja Adidam | Advanced analytics (VIF analysis, Fixed Effects regression, SHAP explainability, K-Means clustering); independently designed and built Tabs 1–4 (Crime Heatmap, What-If Simulator, AI Policy Advisor, Analytics Dashboard); engineered and implemented Tabs 5–7 based on analytical designs; Streamlit Cloud deployment; GitHub repository management |
| Nisha Ravi Babu | EDA, statistical modelling, and real-world system design — feature set design, derived variable engineering (log density, wage gap, conviction probability transforms), scenario simulation, research alignment framework, AI agent pipeline, bias audit methodology, and deployment architecture (Pipeline serialization, joblib model export) |
| Shivani Nagaram | Data collection, cleaning, and feature engineering — missing value treatment, data quality validation, variable transformation, and dataset preparation |
| Mahi Bharat Patel | Visualisation development and report compilation — charts, dashboards, team coordination, and project documentation |
