"""Thin orchestration layer around :class:`models.forecasting_model.RevenueForecastModel`.

The Streamlit app imports a single helper from this module
(:func:`build_forecasts`) so the UI never has to deal with model lifecycle
details directly. Caching is done on the Streamlit side via
``@st.cache_resource``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from models.forecasting_model import (
    ModelMetrics,
    RevenueForecastModel,
    get_feature_importances,
)


@dataclass
class ForecastBundle:
    """Container for everything the dashboard needs after training.

    Attributes
    ----------
    model:
        The fitted :class:`RevenueForecastModel`.
    predictions:
        Per-record dataframe with predicted revenue and quota probability.
    quota_trend:
        Monthly aggregation of predicted quota attainment.
    risk_accounts:
        Top accounts at risk per the ML risk score.
    metrics:
        Training/test sizes, MAE, R^2 and quota classifier accuracy.
    feature_importances:
        Top model features (already truncated to the most informative ones).
    """

    model: RevenueForecastModel
    predictions: pd.DataFrame
    quota_trend: pd.DataFrame
    risk_accounts: pd.DataFrame
    metrics: ModelMetrics
    feature_importances: Dict[str, float]


def build_forecasts(df: pd.DataFrame, random_state: int = 42) -> ForecastBundle:
    """Train the forecasting model and produce all derived artefacts.

    Parameters
    ----------
    df:
        Cleaned dataset returned by :func:`utils.data_loader.load_revenue_data`.
    random_state:
        Seed forwarded to :class:`RevenueForecastModel`.
    """
    model = RevenueForecastModel(random_state=random_state)
    metrics = model.fit(df)
    predictions = model.predict_next_quarter(df)
    return ForecastBundle(
        model=model,
        predictions=predictions,
        quota_trend=model.quota_attainment_trend(predictions),
        risk_accounts=model.risk_accounts(predictions, top_n=15),
        metrics=metrics,
        feature_importances=get_feature_importances(model),
    )
