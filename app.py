"""ACME Revenue Forecasting Engine — Streamlit dashboard entry point.

Run locally with::

    streamlit run app.py

The app composes the analytical helpers in :mod:`utils` and the ML pipeline in
:mod:`models.forecasting_model` into a single executive-style dashboard. All
heavy work (data loading, model training) is cached so interactions stay snappy.
"""

from __future__ import annotations

import streamlit as st

from utils import analytics
from utils.data_loader import (
    aggregate_by,
    apply_filters,
    get_filter_options,
    load_revenue_data,
    validate_revenue_calculations,
)
from utils.forecasting import ForecastBundle, build_forecasts
from utils import visualizations as viz


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ACME Horizon — Revenue Forecasting Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    /* Tighten default Streamlit padding for a denser dashboard look. */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Headline gradient for the dashboard title. */
    .acme-hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #06b6d4 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.18);
    }
    .acme-hero h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .acme-hero p  { color: #dbeafe; margin: 0.2rem 0 0 0; font-size: 0.95rem; }

    /* Metric cards. */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .metric-card .label { color: #475569; font-size: 0.78rem; text-transform: uppercase;
                          letter-spacing: 0.06em; font-weight: 600; }
    .metric-card .value { color: #0f172a; font-size: 1.55rem; font-weight: 700;
                          margin-top: 0.25rem; }
    .metric-card .delta { color: #2563eb; font-size: 0.82rem; margin-top: 0.15rem; }
    .metric-card.warn  .value { color: #b45309; }
    .metric-card.danger .value { color: #b91c1c; }
    .metric-card.good   .value { color: #047857; }

    /* Section headings. */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1rem 0 0.4rem 0;
        border-left: 4px solid #2563eb;
        padding-left: 0.6rem;
    }

    /* Data tables. */
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 10px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_load():
    """Cache the cleaned dataframe across reruns for instant filter changes."""
    return load_revenue_data()


@st.cache_resource(show_spinner=False)
def _cached_forecast(_signature: str) -> ForecastBundle:
    """Train the ML pipeline once per dataset version.

    The ``_signature`` argument is a content fingerprint (row count + sum) so
    that swapping in a new CSV invalidates the cached model.
    """
    df = _cached_load()
    return build_forecasts(df)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def render_hero() -> None:
    """Top hero banner."""
    st.markdown(
        """
        <div class="acme-hero">
            <h1>ACME Horizon — Revenue Forecasting Engine</h1>
            <p>Sales intelligence, ML-driven forecasts, and revenue risk monitoring across regions, products and teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str = "", flavour: str = "") -> str:
    """Return the HTML for a single metric card."""
    klass = f"metric-card {flavour}".strip()
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (
        f'<div class="{klass}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"{delta_html}"
        "</div>"
    )


def render_top_metrics(metrics: dict, ml_metrics) -> None:
    """Render the 4 headline metric cards plus 2 ML cards."""
    cols = st.columns(4)
    cards = [
        ("Total Revenue", f"€{metrics['total_revenue']:,.0f}",
         f"Pipeline: €{metrics['pipeline_value']:,.0f}", ""),
        ("Forecasted Revenue (Next Q)", f"€{metrics['forecasted_revenue']:,.0f}",
         f"Avg Confidence: {metrics['avg_confidence']:.2f}", "good"),
        ("Avg Quota Attainment",
         f"{metrics['avg_quota_attainment']:.1f}%",
         "Target: 100%",
         "good" if metrics["avg_quota_attainment"] >= 80
         else "warn" if metrics["avg_quota_attainment"] >= 50 else "danger"),
        ("Churned Revenue", f"€{metrics['churned_revenue']:,.0f}",
         "Lower is better", "danger"),
    ]
    for col, (label, value, delta, flavour) in zip(cols, cards):
        col.markdown(render_metric_card(label, value, delta, flavour), unsafe_allow_html=True)

    if ml_metrics is not None:
        st.markdown("<div class='section-title'>ML Model Health</div>", unsafe_allow_html=True)
        ml_cols = st.columns(4)
        ml_cards = [
            ("Forecast MAE", f"€{ml_metrics.mae_eur:,.0f}", "Mean abs error per record", "warn"),
            ("Forecast R²", f"{ml_metrics.r2:.3f}", "Higher is better", "good"),
            ("Quota Classifier Accuracy",
             f"{ml_metrics.quota_classifier_accuracy * 100:.1f}%",
             "Train/test 80/20", "good"),
            ("Train / Test Records",
             f"{ml_metrics.train_size} / {ml_metrics.test_size}",
             "Records used for ML", ""),
        ]
        for col, (label, value, delta, flavour) in zip(ml_cols, ml_cards):
            col.markdown(render_metric_card(label, value, delta, flavour), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App body
# ---------------------------------------------------------------------------

def main() -> None:
    render_hero()

    # Load data + model bundle (both cached).
    with st.spinner("Loading revenue dataset…"):
        df = _cached_load()
    signature = f"{len(df)}-{df['total_revenue_eur'].sum():.2f}"
    with st.spinner("Training ML forecasting model…"):
        bundle = _cached_forecast(signature)

    # ----------------------------- Sidebar ---------------------------------
    options = get_filter_options(df)
    st.sidebar.header("Filters")
    sel_regions = st.sidebar.multiselect("Region", options["regions"], default=[])
    sel_products = st.sidebar.multiselect("Product", options["products"], default=[])
    sel_teams = st.sidebar.multiselect("Sales Team", options["teams"], default=[])
    sel_plans = st.sidebar.multiselect("Subscription Plan", options["plans"], default=[])

    st.sidebar.markdown("---")
    st.sidebar.header("Data Quality")
    quality = validate_revenue_calculations(df)
    st.sidebar.metric("Records", f"{quality['rows']:,}")
    st.sidebar.metric("Regions / Products / Teams",
                      f"{quality['regions']} / {quality['products']} / {quality['teams']}")
    st.sidebar.metric("Max Revenue Reconciliation Δ",
                      f"€{quality['max_revenue_diff_eur']:.2f}")
    if quality["negative_revenue_rows"]:
        st.sidebar.warning(f"{quality['negative_revenue_rows']} rows have negative net revenue (heavy churn months).")
    else:
        st.sidebar.success("Revenue reconciliation passed.")

    # Apply filters to the working frames.
    df_f = apply_filters(df, sel_regions, sel_products, sel_teams, sel_plans)
    pred_f = apply_filters(
        bundle.predictions, sel_regions, sel_products, sel_teams, sel_plans
    )
    if df_f.empty:
        st.warning("No records match the selected filters. Adjust them in the sidebar.")
        return

    # ------------------------- Headline metrics ----------------------------
    metrics = analytics.compute_top_metrics(df_f)
    render_top_metrics(metrics, bundle.metrics)

    # --------------------------- Tabs / charts -----------------------------
    tab_overview, tab_pipeline, tab_forecast, tab_tables, tab_advanced = st.tabs(
        ["Revenue Overview", "Pipeline & Quota", "Forecast & ML",
         "Account Tables", "Advanced Analytics"]
    )

    with tab_overview:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("<div class='section-title'>Revenue by Region</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                viz.revenue_by_region_chart(analytics.revenue_by_region(df_f)),
                use_container_width=True,
            )
        with c2:
            st.markdown("<div class='section-title'>Forecast vs Actual Revenue</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                viz.forecast_vs_actual_chart(analytics.forecast_vs_actual(df_f)),
                use_container_width=True,
            )

        st.markdown("<div class='section-title'>Product Revenue Trends</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            viz.product_revenue_trend_chart(analytics.revenue_by_product(df_f)),
            use_container_width=True,
        )

    with tab_pipeline:
        c1, c2 = st.columns([1, 1.1])
        with c1:
            st.markdown("<div class='section-title'>Pipeline Distribution</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                viz.pipeline_distribution_chart(analytics.pipeline_distribution(df_f)),
                use_container_width=True,
            )
        with c2:
            st.markdown("<div class='section-title'>Quota Attainment Heatmap</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                viz.quota_attainment_heatmap(analytics.quota_attainment_matrix(df_f)),
                use_container_width=True,
            )

        st.markdown("<div class='section-title'>Subscription Plan Mix</div>",
                    unsafe_allow_html=True)
        plan_agg = aggregate_by(df_f, "subscription_plan")
        st.dataframe(
            plan_agg.style.format(
                {
                    "total_revenue_eur": "€{:,.0f}",
                    "mrr_eur": "€{:,.0f}",
                    "expansion_eur": "€{:,.0f}",
                    "renewal_eur": "€{:,.0f}",
                    "churn_eur": "€{:,.0f}",
                    "pipeline_eur": "€{:,.0f}",
                    "closed_won_eur": "€{:,.0f}",
                    "forecast_eur": "€{:,.0f}",
                    "avg_quota_pct": "{:.1f}%",
                    "avg_confidence": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    with tab_forecast:
        st.markdown(
            "Predicting next-quarter revenue for every account/record using a "
            "Gradient Boosting regressor trained on historical bookings, "
            "pipeline state and customer health features."
        )

        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("<div class='section-title'>Quota Attainment Trend (Predicted)</div>",
                        unsafe_allow_html=True)
            quota_trend = bundle.model.quota_attainment_trend(pred_f) \
                if not pred_f.empty else bundle.quota_trend
            st.plotly_chart(
                viz.forecast_vs_actual_chart(
                    quota_trend.assign(
                        actual_revenue=quota_trend["actual_quota_avg"],
                        forecast_revenue=quota_trend["predicted_quota_prob"] * 100,
                    )
                ),
                use_container_width=True,
            )
        with c2:
            st.markdown("<div class='section-title'>Top ML Features</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                viz.feature_importance_chart(bundle.feature_importances),
                use_container_width=True,
            )

        st.markdown("<div class='section-title'>ML-Predicted Revenue Risk Accounts</div>",
                    unsafe_allow_html=True)
        risk_df = bundle.model.risk_accounts(pred_f, top_n=15) \
            if not pred_f.empty else bundle.risk_accounts
        st.dataframe(
            risk_df.style.format(
                {
                    "mrr": "€{:,.0f}",
                    "forecast": "€{:,.0f}",
                    "predicted": "€{:,.0f}",
                    "forecast_gap_eur": "€{:,.0f}",
                    "miss_quota_prob": "{:.2f}",
                    "churn": "€{:,.0f}",
                    "ml_risk_score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    with tab_tables:
        st.markdown("<div class='section-title'>High Revenue Accounts</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            analytics.high_revenue_accounts(df_f, top_n=15).style.format(
                {
                    "total_revenue": "€{:,.0f}",
                    "mrr": "€{:,.0f}",
                    "expansion": "€{:,.0f}",
                    "renewal": "€{:,.0f}",
                    "avg_renewal_prob": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        st.markdown("<div class='section-title'>Revenue Risk Customers (Heuristic)</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            analytics.revenue_risk_customers(df_f, top_n=15).style.format(
                {
                    "mrr": "€{:,.0f}",
                    "churn": "€{:,.0f}",
                    "avg_renewal_prob": "{:.1f}%",
                    "avg_growth_score": "{:.1f}",
                    "avg_quota": "{:.1f}%",
                    "risk_score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("<div class='section-title'>Underperforming Sales Teams</div>",
                    unsafe_allow_html=True)
        under = analytics.underperforming_teams(df_f, threshold_pct=50.0)
        if under.empty:
            st.success("All sales teams in the current view are tracking ≥ 50% quota attainment.")
        else:
            st.dataframe(
                under.style.format(
                    {
                        "avg_quota": "{:.1f}%",
                        "total_revenue": "€{:,.0f}",
                        "forecast_revenue": "€{:,.0f}",
                        "avg_confidence": "{:.2f}",
                        "gap_to_quota_pct": "{:.1f}%",
                    }
                ),
                use_container_width=True,
            )

    with tab_advanced:
        st.markdown("<div class='section-title'>Revenue Growth Trend</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            viz.revenue_growth_chart(analytics.revenue_growth_trend(df_f)),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-title'>Churn Impact Analysis</div>",
                        unsafe_allow_html=True)
            churn_df = analytics.churn_impact_analysis(df_f)
            st.plotly_chart(viz.churn_impact_chart(churn_df), use_container_width=True)
            st.dataframe(
                churn_df.style.format(
                    {
                        "gross_revenue": "€{:,.0f}",
                        "expansion": "€{:,.0f}",
                        "renewal": "€{:,.0f}",
                        "churn": "€{:,.0f}",
                        "gross_total": "€{:,.0f}",
                        "net_revenue": "€{:,.0f}",
                        "churn_rate_pct": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )
        with c2:
            st.markdown("<div class='section-title'>Forecast Confidence Analysis</div>",
                        unsafe_allow_html=True)
            conf_df = analytics.forecast_confidence_analysis(df_f)
            st.plotly_chart(viz.confidence_accuracy_chart(conf_df), use_container_width=True)
            st.dataframe(
                conf_df.style.format(
                    {
                        "forecast_revenue": "€{:,.0f}",
                        "actual_revenue": "€{:,.0f}",
                        "avg_confidence": "{:.2f}",
                        "accuracy_ratio": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

    st.markdown("---")
    st.caption(
        "© ACME Horizon — Revenue Intelligence Platform. Data refreshes are cached "
        "in-memory; restart Streamlit to pick up new CSV drops."
    )


if __name__ == "__main__":
    main()
