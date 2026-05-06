"""High-level analytics helpers for the dashboard.

Each function in this module returns a tidy ``pandas.DataFrame`` (or scalar
dict) that the Streamlit layer can render directly. Keeping the analytical
logic here means the UI stays thin and the same metrics can be reused by
notebooks, tests, or downstream services.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------

def compute_top_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute the four headline KPIs displayed at the top of the dashboard."""
    if df.empty:
        return {
            "total_revenue": 0.0,
            "forecasted_revenue": 0.0,
            "avg_quota_attainment": 0.0,
            "churned_revenue": 0.0,
            "pipeline_value": 0.0,
            "avg_confidence": 0.0,
        }

    return {
        "total_revenue": float(df["total_revenue_eur"].sum()),
        "forecasted_revenue": float(df["forecasted_revenue_next_quarter"].sum()),
        "avg_quota_attainment": float(df["quota_attainment_pct"].mean()),
        "churned_revenue": float(df["churned_revenue_eur"].sum()),
        "pipeline_value": float(df["pipeline_value_eur"].sum()),
        "avg_confidence": float(df["forecast_confidence_score"].mean()),
    }


# ---------------------------------------------------------------------------
# Aggregations used by the charts
# ---------------------------------------------------------------------------

def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Total / forecast / churn rolled up to the region level."""
    return (
        df.groupby("region", dropna=False)
        .agg(
            total_revenue=("total_revenue_eur", "sum"),
            forecast_revenue=("forecasted_revenue_next_quarter", "sum"),
            churn=("churned_revenue_eur", "sum"),
            pipeline=("pipeline_value_eur", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue per product line for trend charts."""
    return (
        df.groupby(["month_label", "product_line"], dropna=False)
        .agg(total_revenue=("total_revenue_eur", "sum"))
        .reset_index()
        .sort_values("month_label")
    )


def forecast_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    """Compare booked revenue against the forecasted next-quarter revenue.

    We aggregate to the month level so the resulting line chart is readable
    irrespective of dataset size.
    """
    grouped = (
        df.groupby("month_label", dropna=False)
        .agg(
            actual_revenue=("total_revenue_eur", "sum"),
            forecast_revenue=("forecasted_revenue_next_quarter", "sum"),
        )
        .reset_index()
        .sort_values("month_label")
    )
    grouped["variance"] = grouped["actual_revenue"] - grouped["forecast_revenue"]
    grouped["variance_pct"] = np.where(
        grouped["forecast_revenue"] > 0,
        (grouped["variance"] / grouped["forecast_revenue"]) * 100.0,
        0.0,
    )
    return grouped


def pipeline_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribute pipeline value across ``sales_stage`` for the funnel chart."""
    return (
        df.groupby("sales_stage", dropna=False)
        .agg(
            pipeline_value=("pipeline_value_eur", "sum"),
            opportunities=("record_id", "count"),
            closed_won=("closed_won_value_eur", "sum"),
            closed_lost=("closed_lost_value_eur", "sum"),
        )
        .reset_index()
        .sort_values("pipeline_value", ascending=False)
    )


def quota_attainment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a region x sales_team matrix of average quota attainment.

    Returned in *long* form. The visualisation layer pivots it into a 2D
    heatmap.
    """
    return (
        df.groupby(["region", "sales_team"], dropna=False)
        .agg(avg_quota=("quota_attainment_pct", "mean"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def high_revenue_accounts(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top N customers by total booked revenue."""
    grouped = (
        df.groupby(["customer_name", "industry", "region"], dropna=False)
        .agg(
            total_revenue=("total_revenue_eur", "sum"),
            mrr=("monthly_recurring_revenue_eur", "sum"),
            expansion=("expansion_revenue_eur", "sum"),
            renewal=("renewal_revenue_eur", "sum"),
            avg_renewal_prob=("renewal_probability_pct", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
    )
    return grouped


def revenue_risk_customers(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Customers most likely to drag revenue down next quarter.

    The risk score blends three signals (each on a 0-1 scale) and weights
    them. The intent is to highlight accounts that are *both* large and
    fragile, not just the smallest customers.
    """
    risk = (
        df.groupby(["customer_name", "industry", "region"], dropna=False)
        .agg(
            mrr=("monthly_recurring_revenue_eur", "sum"),
            churn=("churned_revenue_eur", "sum"),
            avg_renewal_prob=("renewal_probability_pct", "mean"),
            avg_growth_score=("customer_growth_score", "mean"),
            avg_quota=("quota_attainment_pct", "mean"),
            records=("record_id", "count"),
        )
        .reset_index()
    )

    # Normalise each signal to [0, 1] before combining.
    churn_norm = _safe_minmax(risk["churn"])
    renewal_risk = (100.0 - risk["avg_renewal_prob"].clip(0, 100)) / 100.0
    growth_risk = (100.0 - risk["avg_growth_score"].clip(0, 100)) / 100.0
    mrr_norm = _safe_minmax(risk["mrr"])

    # Weight churn highest because it's already realised loss, then renewal
    # probability, then growth trajectory. Multiply by mrr_norm so the score
    # scales with account size.
    risk["risk_score"] = (
        (0.5 * churn_norm + 0.3 * renewal_risk + 0.2 * growth_risk)
        * (0.4 + 0.6 * mrr_norm)
    ).round(3)

    return (
        risk.sort_values("risk_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def underperforming_teams(df: pd.DataFrame, threshold_pct: float = 50.0) -> pd.DataFrame:
    """Sales teams with average quota attainment below ``threshold_pct``."""
    grouped = (
        df.groupby(["sales_team", "region"], dropna=False)
        .agg(
            avg_quota=("quota_attainment_pct", "mean"),
            total_revenue=("total_revenue_eur", "sum"),
            forecast_revenue=("forecasted_revenue_next_quarter", "sum"),
            avg_confidence=("forecast_confidence_score", "mean"),
            reps=("account_executive", "nunique"),
            records=("record_id", "count"),
        )
        .reset_index()
    )
    grouped["gap_to_quota_pct"] = (100.0 - grouped["avg_quota"]).round(2)
    return (
        grouped[grouped["avg_quota"] < threshold_pct]
        .sort_values("avg_quota")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Advanced analytics
# ---------------------------------------------------------------------------

def revenue_growth_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month growth rate for total booked revenue."""
    monthly = (
        df.groupby("month_label", dropna=False)
        .agg(total_revenue=("total_revenue_eur", "sum"))
        .reset_index()
        .sort_values("month_label")
    )
    monthly["mom_growth_pct"] = (
        monthly["total_revenue"].pct_change() * 100.0
    ).round(2)
    monthly["rolling_3m_revenue"] = (
        monthly["total_revenue"].rolling(window=3, min_periods=1).mean().round(2)
    )
    return monthly


def churn_impact_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much churn ate into gross revenue per region."""
    grouped = (
        df.groupby("region", dropna=False)
        .agg(
            gross_revenue=("monthly_recurring_revenue_eur", "sum"),
            expansion=("expansion_revenue_eur", "sum"),
            renewal=("renewal_revenue_eur", "sum"),
            churn=("churned_revenue_eur", "sum"),
        )
        .reset_index()
    )
    grouped["gross_total"] = (
        grouped["gross_revenue"] + grouped["expansion"] + grouped["renewal"]
    )
    grouped["net_revenue"] = grouped["gross_total"] - grouped["churn"]
    grouped["churn_rate_pct"] = np.where(
        grouped["gross_total"] > 0,
        (grouped["churn"] / grouped["gross_total"]) * 100.0,
        0.0,
    ).round(2)
    return grouped.sort_values("churn", ascending=False).reset_index(drop=True)


def forecast_confidence_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of forecast confidence buckets and their realised revenue."""
    bins = [0, 0.5, 0.7, 0.85, 1.0001]
    labels = ["Low (<0.5)", "Moderate (0.5-0.7)", "High (0.7-0.85)", "Very High (0.85+)"]
    bucket = pd.cut(
        df["forecast_confidence_score"], bins=bins, labels=labels, include_lowest=True
    )
    out = (
        df.assign(confidence_bucket=bucket)
        .groupby("confidence_bucket", dropna=False, observed=False)
        .agg(
            records=("record_id", "count"),
            forecast_revenue=("forecasted_revenue_next_quarter", "sum"),
            actual_revenue=("total_revenue_eur", "sum"),
            avg_confidence=("forecast_confidence_score", "mean"),
        )
        .reset_index()
    )
    out["accuracy_ratio"] = np.where(
        out["forecast_revenue"] > 0,
        (out["actual_revenue"] / out["forecast_revenue"]).round(3),
        0.0,
    )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_minmax(s: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1] without blowing up on constants."""
    s = s.astype(float)
    span = s.max() - s.min()
    if span <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / span
