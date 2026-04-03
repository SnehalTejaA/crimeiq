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
| 🤖 AI Policy Advisor | Groq API generates evidence-based policy recommendations from SHAP drivers |
| 📊 Analytics Dashboard | Trend charts, feature importance, SHAP summary, model performance |
| 🎭 Scenario Simulation | Preset scenarios simulating systemic socioeconomic or law enforcement shifts |
| 🔬 Research Alignment | Validates feature selection against established criminology theory |
| ⚖️ Bias & Fairness Audit | Analyzes predictive fairness across key demographic sub-groups |

---

## Setup

### 1. Clone / download
```bash
git clone https://github.com/SnehalTejaA/crimeiq.git
cd crimeiq
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the app
```
streamlit run app.py
```

The app loads data directly from GitHub on first run (cached after that).

---

## Project Structure

```
crimeiq/
├── app.py            # Main Streamlit app — all 4 tabs
├── data_loader.py    # Data loading, model training, prediction helpers
├── heatmap.py        # Folium heatmap builder
├── llm_policy.py     # Claude API integration for policy recommendations
├── requirements.txt
└── README.md
```

---

## Dataset

**Cornwell & Trumbull NC County Crime Dataset (1981–1987)**
- 630 observations · 90 counties · 7 years
- Source: [GitHub repo](https://github.com/nishapattim05-del/crime-project-data)

**14 VIF-selected features used (all VIF < 10):**
`prbarr`, `prbconv`, `prbpris`, `avgsen`, `polpc`, `density`, `taxpc`,
`west`, `central`, `urban`, `pctmin80`, `wcon`, `wtuc`, `wfed`

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

## AI Policy Advisor

The policy recommendation engine sends the SHAP feature drivers for the
current What-If profile to the Claude Sonnet API and receives a structured
policy brief with:

1. Situation summary
2. Top 3 actionable policy recommendations (tied to SHAP drivers)
3. Risk factors to monitor
4. One honest data limitation

---

## Team

| Member | Role |
|--------|------|
| Snehal Teja Adidam | Advanced analytics (VIF, Fixed Effects, SHAP, K-Means) |
| Nisha Ravi Babu | EDA and statistical modelling |
| Shivani Nagaram | Data cleaning and feature engineering |
| Mahi Bharat Patel | Visualisations and report compilation |
