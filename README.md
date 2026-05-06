# ACME Horizon — Revenue Forecasting Engine

A sales intelligence and revenue forecasting analytics platform built for **ACME Horizon**. The application combines clean, validated revenue data, a Scikit-learn ML forecasting pipeline, and an interactive Streamlit dashboard to give RevOps, Finance, and Sales leadership a single pane of glass on the next quarter.

---

## Project Overview

The engine answers four business questions in one place:

1. **How much revenue did we book** by region, product line, sales team, and subscription plan?
2. **How much will we book next quarter?** — driven by an ML model trained on historical bookings, pipeline state, and customer health features.
3. **Who is at risk?** — surfacing the customers and teams most likely to drag the next quarter's number.
4. **How confident are we?** — calibrating realised vs. forecasted revenue at every confidence band.

The app ships with `data/revenue_forecasting_sales_intelligence.csv` (152 records spanning regions, products, sales teams, and subscription plans) so you can run it end-to-end immediately.

---

## Architecture

```
acme-revenue-forecasting-engine/
│
├── app.py                       # Streamlit entry point: layout, tabs, metric cards
├── requirements.txt             # Pinned dependencies (Streamlit, Pandas, Plotly, sklearn)
├── README.md                    # You are here
│
├── data/
│   └── revenue_forecasting_sales_intelligence.csv
│
├── models/
│   ├── __init__.py
│   └── forecasting_model.py     # RevenueForecastModel: sklearn pipelines + risk scoring
│
└── utils/
    ├── __init__.py
    ├── data_loader.py           # Load, validate, clean, enrich, filter the dataset
    ├── analytics.py             # KPI + chart aggregations + advanced analytics
    ├── forecasting.py           # Thin orchestration wrapper around the ML model
    └── visualizations.py        # Plotly chart factories with the ACME visual theme
```

### Data flow

```
CSV ──► data_loader.load_revenue_data ──► cleaned DataFrame
                                            │
                       ┌────────────────────┼─────────────────────┐
                       ▼                    ▼                     ▼
              utils.analytics       utils.forecasting       utils.visualizations
              (KPIs, tables)        (model bundle)          (Plotly figures)
                       └────────────────────┴─────────────────────┘
                                            ▼
                                          app.py
                                     (Streamlit UI)
```

Heavy work is cached:

- `@st.cache_data` on the cleaned DataFrame.
- `@st.cache_resource` on the trained ML pipeline (re-trains automatically when the CSV signature changes).

---

## ML Forecasting Explanation

The forecasting engine is a single ``RevenueForecastModel`` that bundles:

| Sub-model        | Type                            | Target                                              |
|------------------|---------------------------------|-----------------------------------------------------|
| Revenue regressor | `GradientBoostingRegressor`      | `forecasted_revenue_next_quarter` (€)               |
| Quota classifier  | `GradientBoostingClassifier`     | `quota_attainment_pct >= 100` (binary)              |
| Risk scorer       | Composite analytical score      | Probability-weighted forecast gap × churn signal     |

**Features used** (one-hot encoded for categoricals, standard-scaled for numerics):

- *Categorical:* `region`, `country`, `sales_team`, `subscription_plan`, `product_line`, `sales_stage`, `industry`.
- *Numeric:* `monthly_recurring_revenue_eur`, `expansion_revenue_eur`, `renewal_revenue_eur`, `pipeline_value_eur`, `closed_won_value_eur`, `sales_target_eur`, `quota_attainment_pct`, `customer_growth_score`, `renewal_probability_pct`, `forecast_confidence_score`, `year`, `month_num`.

**Pipeline:**

```
ColumnTransformer(
    cat → OneHotEncoder(handle_unknown="ignore"),
    num → StandardScaler(),
) ─► GradientBoosting{Regressor, Classifier}
```

**Train / evaluate:** `train_test_split` (80 / 20), seeded with `random_state=42`. We surface MAE (€), R², and quota classifier accuracy directly in the dashboard so users can see model health.

**Revenue risk score:** an account-level blend of (a) the forecast gap (sum of forecasted vs. predicted), (b) the classifier's probability of *missing* quota, and (c) realised churn — min-max normalised and combined with weights `0.5 / 0.3 / 0.2`. The result is sorted to surface the top 15 ML-flagged accounts.

A second, *heuristic* risk view (`utils.analytics.revenue_risk_customers`) is available alongside the ML view for cross-checking.

---

## Tech Stack

| Layer          | Technology                                |
|----------------|-------------------------------------------|
| UI / dashboard | **Streamlit** with custom CSS            |
| Data wrangling | **Pandas** + **NumPy**                    |
| Visualisation  | **Plotly** (Express + Graph Objects)      |
| ML             | **Scikit-learn** (GradientBoosting + Pipelines + ColumnTransformer) |
| Language       | **Python 3.10+**                          |

---

## Setup Guide

### 1. Clone & enter the project

```bash
git clone https://github.com/<your-org>/acme-revenue-forecasting-engine.git
cd acme-revenue-forecasting-engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open the URL Streamlit prints in your browser (usually `http://localhost:8501`).

### 5. Swap in your own dataset

Drop a new CSV with the same schema into `data/` and overwrite `revenue_forecasting_sales_intelligence.csv`, or pass an explicit path:

```python
from utils.data_loader import load_revenue_data
df = load_revenue_data("/path/to/your.csv")
```

---

## Dashboard Features

### Top metrics

- **Total Revenue** — MRR + Expansion + Renewal − Churn for the filtered slice.
- **Forecasted Revenue (Next Quarter)** — sum of `forecasted_revenue_next_quarter`.
- **Avg Quota Attainment** — mean of `quota_attainment_pct`.
- **Churned Revenue** — sum of `churned_revenue_eur`.
- **ML model health** — MAE, R², quota classifier accuracy, train/test sizes.

### Charts

- **Revenue by Region** — grouped bar of total vs. forecasted revenue.
- **Forecast vs Actual Revenue** — month-over-month comparison line chart.
- **Pipeline Distribution** — donut chart by `sales_stage`.
- **Quota Attainment Heatmap** — region × team matrix on a red-yellow-green scale.
- **Product Revenue Trends** — multi-line chart per `product_line`.
- **Revenue Growth Trend** — bars with overlaid MoM growth %.
- **Churn Impact** — gross vs. churn per region.
- **Forecast Confidence** — bucketed forecast vs. actual revenue.
- **Top ML Features** — gradient-boosted feature importances.

### Tables

- **High Revenue Accounts** — top 15 customers by booked revenue.
- **Revenue Risk Customers** — heuristic risk ranking (churn + renewal + growth).
- **Underperforming Sales Teams** — average quota attainment below threshold.
- **Subscription Plan Mix** — aggregated metrics per plan.
- **ML-Predicted Risk Accounts** — top 15 accounts flagged by the ML model.

### Filters (sidebar)

- Region · Product · Sales Team · Subscription Plan (multi-select, all combine).

### Advanced analytics

- Revenue growth trend with rolling 3-month average.
- Churn impact by region with churn-rate %.
- Forecast confidence calibration (forecast vs. realised per confidence bucket).

---

## Screenshots

> _Add screenshots once the dashboard is running. Suggested captures:_

1. `docs/screenshot-overview.png` — Hero banner, top metric cards, Revenue by Region.
2. `docs/screenshot-pipeline.png` — Pipeline donut + Quota Attainment heatmap.
3. `docs/screenshot-forecast.png` — Forecast vs Actual + Top ML Features.
4. `docs/screenshot-tables.png` — High-revenue accounts and risk tables.
5. `docs/screenshot-advanced.png` — Revenue growth + churn + confidence panels.

To capture them:

```bash
streamlit run app.py
# Use your OS screenshot tool, save into docs/, then reference here.
```

---

## Repository Hygiene

- `utils/` and `models/` are pure Python modules — easy to unit-test or call from a notebook.
- All ML hyperparameters live in `models/forecasting_model.py` so they are version-controlled.
- No global state outside the Streamlit cache layer.
- `requirements.txt` pins minimum versions for reproducibility.

---

## License

Internal demo project for ACME Horizon. Adapt freely for your own RevOps team.
