"""Plotly chart factories for the ACME Revenue Forecasting dashboard.

Every function in this module returns a fully-styled
:class:`plotly.graph_objects.Figure`. Centralising the styling here keeps
the dashboard visually consistent (fonts, palette, margins, hover format)
while the app layer just composes the figures.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ACME Horizon brand-aligned palette. Using a single, ordered palette keeps
# the dashboard visually cohesive across charts.
PALETTE = [
    "#2563EB",  # primary blue
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#64748B",  # slate
]

PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(148, 163, 184, 0.2)"


def _apply_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the shared visual theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title or "", x=0.0, xanchor="left", font=dict(size=16)),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        legend=dict(orientation="h", y=-0.2, x=0, xanchor="left"),
        colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def revenue_by_region_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar of total revenue and forecast per region."""
    fig = go.Figure()
    fig.add_bar(
        name="Total Revenue",
        x=df["region"],
        y=df["total_revenue"],
        marker_color=PALETTE[0],
        hovertemplate="%{x}<br>Revenue: €%{y:,.0f}<extra></extra>",
    )
    fig.add_bar(
        name="Forecast Revenue",
        x=df["region"],
        y=df["forecast_revenue"],
        marker_color=PALETTE[1],
        hovertemplate="%{x}<br>Forecast: €%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="€", separatethousands=True)
    return _apply_theme(fig, "Revenue by Region")


def forecast_vs_actual_chart(df: pd.DataFrame) -> go.Figure:
    """Combined line chart comparing actual vs forecast revenue over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month_label"],
            y=df["actual_revenue"],
            name="Actual Revenue",
            mode="lines+markers",
            line=dict(width=3, color=PALETTE[0]),
            hovertemplate="%{x}<br>Actual: €%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["month_label"],
            y=df["forecast_revenue"],
            name="Forecast Revenue",
            mode="lines+markers",
            line=dict(width=3, dash="dash", color=PALETTE[1]),
            hovertemplate="%{x}<br>Forecast: €%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="€", separatethousands=True)
    return _apply_theme(fig, "Forecast vs Actual Revenue")


def pipeline_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of pipeline value distributed across sales stages."""
    fig = px.pie(
        df,
        values="pipeline_value",
        names="sales_stage",
        hole=0.55,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}<br>Pipeline: €%{value:,.0f}<extra></extra>",
    )
    return _apply_theme(fig, "Pipeline Distribution by Stage")


def quota_attainment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Region x team heatmap of average quota attainment."""
    pivot = (
        df.pivot_table(
            index="region",
            columns="sales_team",
            values="avg_quota",
            aggfunc="mean",
        )
        .fillna(0.0)
        .round(1)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=100,
            zmin=0,
            zmax=max(150, float(pivot.values.max()) if pivot.size else 150),
            colorbar=dict(title="Quota %"),
            hovertemplate="Region: %{y}<br>Team: %{x}<br>Quota: %{z:.1f}%<extra></extra>",
        )
    )
    return _apply_theme(fig, "Quota Attainment Heatmap (Region x Team)")


def product_revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly revenue per product line."""
    fig = px.line(
        df,
        x="month_label",
        y="total_revenue",
        color="product_line",
        markers=True,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: €%{y:,.0f}<extra></extra>")
    fig.update_yaxes(tickprefix="€", separatethousands=True)
    return _apply_theme(fig, "Product Revenue Trends")


def revenue_growth_chart(df: pd.DataFrame) -> go.Figure:
    """Bar+line chart: monthly revenue (bar) overlaid with MoM growth %."""
    fig = go.Figure()
    fig.add_bar(
        name="Revenue",
        x=df["month_label"],
        y=df["total_revenue"],
        marker_color=PALETTE[0],
        hovertemplate="%{x}<br>Revenue: €%{y:,.0f}<extra></extra>",
    )
    fig.add_trace(
        go.Scatter(
            x=df["month_label"],
            y=df["mom_growth_pct"],
            name="MoM Growth %",
            mode="lines+markers",
            yaxis="y2",
            line=dict(width=3, color=PALETTE[3]),
            hovertemplate="%{x}<br>Growth: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Revenue (€)", tickprefix="€", separatethousands=True),
        yaxis2=dict(
            title="MoM Growth %",
            overlaying="y",
            side="right",
            ticksuffix="%",
            showgrid=False,
        ),
    )
    return _apply_theme(fig, "Revenue Growth Trend")


def churn_impact_chart(df: pd.DataFrame) -> go.Figure:
    """Side-by-side bar of gross revenue vs churn per region."""
    fig = go.Figure()
    fig.add_bar(
        name="Gross Revenue",
        x=df["region"],
        y=df["gross_total"],
        marker_color=PALETTE[1],
        hovertemplate="%{x}<br>Gross: €%{y:,.0f}<extra></extra>",
    )
    fig.add_bar(
        name="Churned Revenue",
        x=df["region"],
        y=df["churn"],
        marker_color=PALETTE[3],
        hovertemplate="%{x}<br>Churn: €%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="€", separatethousands=True)
    return _apply_theme(fig, "Churn Impact by Region")


def confidence_accuracy_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: forecast confidence buckets vs realised accuracy ratio."""
    fig = go.Figure()
    fig.add_bar(
        name="Forecasted Revenue",
        x=df["confidence_bucket"].astype(str),
        y=df["forecast_revenue"],
        marker_color=PALETTE[4],
        hovertemplate="%{x}<br>Forecast: €%{y:,.0f}<extra></extra>",
    )
    fig.add_bar(
        name="Actual Revenue",
        x=df["confidence_bucket"].astype(str),
        y=df["actual_revenue"],
        marker_color=PALETTE[0],
        hovertemplate="%{x}<br>Actual: €%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="€", separatethousands=True)
    return _apply_theme(fig, "Forecast Confidence vs Realised Revenue")


def feature_importance_chart(importances: Dict[str, float], top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of the top model features."""
    if not importances:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importances available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return _apply_theme(fig, "Model Feature Importance")

    pairs = list(importances.items())[:top_n][::-1]
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig = go.Figure(
        data=go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=PALETTE[5],
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )
    return _apply_theme(fig, "Top ML Features Driving the Forecast")
