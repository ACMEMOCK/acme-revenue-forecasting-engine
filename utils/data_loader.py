"""Data loading and cleaning utilities for the ACME Revenue Forecasting Engine.

This module owns the full ingestion pipeline: reading the raw CSV from disk,
normalising column types, deriving canonical revenue fields, and validating
that the numbers reconcile against expected business rules.

The output of :func:`load_revenue_data` is a single, analysis-ready
``pandas.DataFrame`` consumed by every other module in the project (analytics,
forecasting, visualisations, and the Streamlit app).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

# Default location of the raw revenue intelligence dataset. We resolve it
# relative to the repository root so the app works regardless of the CWD
# Streamlit is launched from.
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "revenue_forecasting_sales_intelligence.csv",
)

# Columns that must always be present in the source CSV. Anything missing here
# is a hard failure because downstream analytics depend on each field.
REQUIRED_COLUMNS: List[str] = [
    "record_id",
    "region",
    "country",
    "sales_team",
    "account_executive",
    "customer_name",
    "industry",
    "reporting_month",
    "subscription_plan",
    "monthly_recurring_revenue_eur",
    "expansion_revenue_eur",
    "renewal_revenue_eur",
    "churned_revenue_eur",
    "pipeline_value_eur",
    "closed_won_value_eur",
    "closed_lost_value_eur",
    "sales_target_eur",
    "quota_attainment_pct",
    "forecasted_revenue_next_quarter",
    "forecast_confidence_score",
    "customer_growth_score",
    "renewal_probability_pct",
    "product_line",
    "sales_stage",
]

# Numeric columns coerced to float64 with NaN-safe defaulting.
NUMERIC_COLUMNS: List[str] = [
    "monthly_recurring_revenue_eur",
    "expansion_revenue_eur",
    "renewal_revenue_eur",
    "churned_revenue_eur",
    "pipeline_value_eur",
    "closed_won_value_eur",
    "closed_lost_value_eur",
    "sales_target_eur",
    "quota_attainment_pct",
    "forecasted_revenue_next_quarter",
    "forecast_confidence_score",
    "customer_growth_score",
    "renewal_probability_pct",
]


def load_revenue_data(path: str | None = None) -> pd.DataFrame:
    """Load and prepare the revenue intelligence dataset.

    Parameters
    ----------
    path:
        Optional override for the CSV path. Defaults to the bundled
        ``data/revenue_forecasting_sales_intelligence.csv``.

    Returns
    -------
    pandas.DataFrame
        Cleaned, typed, and enriched dataset ready for analytics.
    """
    csv_path = path or DEFAULT_DATA_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Revenue dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_schema(df)
    df = _clean(df)
    df = _enrich(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error if the source CSV is missing required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(missing)}"
        )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes, fill NaNs, and drop obviously invalid rows."""
    df = df.copy()

    # Strip whitespace on string-like categorical fields to avoid duplicate
    # categories such as ``"EMEA"`` vs ``"EMEA "``.
    for col in [
        "region",
        "country",
        "sales_team",
        "account_executive",
        "customer_name",
        "industry",
        "subscription_plan",
        "product_line",
        "sales_stage",
    ]:
        df[col] = df[col].astype(str).str.strip()

    # Coerce numeric columns. Any non-numeric junk becomes NaN, then 0.0 so
    # aggregations remain well-defined.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Quota attainment is sometimes recorded as a percentage point value
    # (e.g. 53.5) and other times as a fraction. Normalise everything to a
    # percentage point scale (0-200+).
    df["quota_attainment_pct"] = df["quota_attainment_pct"].astype(float)

    # Parse reporting month as a proper monthly Period -> Timestamp so we can
    # do time-series math.
    df["reporting_month"] = pd.to_datetime(
        df["reporting_month"], format="%Y-%m", errors="coerce"
    )
    df = df.dropna(subset=["reporting_month"]).reset_index(drop=True)

    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived business metrics used across the dashboard."""
    df = df.copy()

    # Total revenue per record = MRR + expansion + renewal - churn.
    # This is the single number we treat as "actual revenue booked" for that
    # record/month and reconcile against the forecasted revenue.
    df["total_revenue_eur"] = (
        df["monthly_recurring_revenue_eur"]
        + df["expansion_revenue_eur"]
        + df["renewal_revenue_eur"]
        - df["churned_revenue_eur"]
    ).round(2)

    # Net new revenue excludes renewal so we can show "growth" cleanly.
    df["net_new_revenue_eur"] = (
        df["monthly_recurring_revenue_eur"]
        + df["expansion_revenue_eur"]
        - df["churned_revenue_eur"]
    ).round(2)

    # Pipeline conversion = closed-won / pipeline_value, guarded against /0.
    df["pipeline_conversion_pct"] = np.where(
        df["pipeline_value_eur"] > 0,
        (df["closed_won_value_eur"] / df["pipeline_value_eur"]) * 100.0,
        0.0,
    ).round(2)

    # Pull out the year and quarter for time-based aggregations.
    df["year"] = df["reporting_month"].dt.year
    df["quarter"] = df["reporting_month"].dt.to_period("Q").astype(str)
    df["month_label"] = df["reporting_month"].dt.strftime("%Y-%m")

    # A simple "at risk" flag for the risk customer table:
    #   - non-trivial churn that month, OR
    #   - renewal probability < 60 with positive MRR, OR
    #   - quota attainment < 25% with sales target > 0.
    df["is_revenue_risk"] = (
        (df["churned_revenue_eur"] > 0)
        | ((df["renewal_probability_pct"] < 60) & (df["monthly_recurring_revenue_eur"] > 0))
        | ((df["quota_attainment_pct"] < 25) & (df["sales_target_eur"] > 0))
    )

    return df


def validate_revenue_calculations(df: pd.DataFrame) -> Dict[str, float | int]:
    """Run sanity checks over the dataset and return a small report.

    The returned dict is rendered as a small data-quality panel inside the
    Streamlit sidebar so users can trust the numbers.
    """
    expected_total = (
        df["monthly_recurring_revenue_eur"]
        + df["expansion_revenue_eur"]
        + df["renewal_revenue_eur"]
        - df["churned_revenue_eur"]
    )
    diff = (df["total_revenue_eur"] - expected_total).abs()

    return {
        "rows": int(len(df)),
        "regions": int(df["region"].nunique()),
        "products": int(df["product_line"].nunique()),
        "teams": int(df["sales_team"].nunique()),
        "max_revenue_diff_eur": float(diff.max()),
        "negative_revenue_rows": int((df["total_revenue_eur"] < 0).sum()),
        "missing_targets": int((df["sales_target_eur"] <= 0).sum()),
    }


def aggregate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate the canonical revenue metrics by an arbitrary dimension.

    Supports any categorical column on ``df``, e.g. ``region``,
    ``product_line``, ``sales_team``, ``subscription_plan``.
    """
    if group_col not in df.columns:
        raise KeyError(f"Cannot aggregate by unknown column: {group_col}")

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            total_revenue_eur=("total_revenue_eur", "sum"),
            mrr_eur=("monthly_recurring_revenue_eur", "sum"),
            expansion_eur=("expansion_revenue_eur", "sum"),
            renewal_eur=("renewal_revenue_eur", "sum"),
            churn_eur=("churned_revenue_eur", "sum"),
            pipeline_eur=("pipeline_value_eur", "sum"),
            closed_won_eur=("closed_won_value_eur", "sum"),
            forecast_eur=("forecasted_revenue_next_quarter", "sum"),
            avg_quota_pct=("quota_attainment_pct", "mean"),
            avg_confidence=("forecast_confidence_score", "mean"),
            records=("record_id", "count"),
        )
        .reset_index()
        .sort_values("total_revenue_eur", ascending=False)
    )
    return grouped


def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return the unique values used to populate the sidebar filter widgets."""
    return {
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "products": sorted(df["product_line"].dropna().unique().tolist()),
        "teams": sorted(df["sales_team"].dropna().unique().tolist()),
        "plans": sorted(df["subscription_plan"].dropna().unique().tolist()),
    }


def apply_filters(
    df: pd.DataFrame,
    regions: List[str] | None = None,
    products: List[str] | None = None,
    teams: List[str] | None = None,
    plans: List[str] | None = None,
) -> pd.DataFrame:
    """Apply the dashboard filter selections in a single pass."""
    out = df
    if regions:
        out = out[out["region"].isin(regions)]
    if products:
        out = out[out["product_line"].isin(products)]
    if teams:
        out = out[out["sales_team"].isin(teams)]
    if plans:
        out = out[out["subscription_plan"].isin(plans)]
    return out.reset_index(drop=True)
