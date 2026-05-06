"""Scikit-learn revenue forecasting model.

We frame the forecasting task as a supervised regression problem where each
row in the source dataset becomes a training example:

    features = [region, country, sales_team, subscription_plan, product_line,
                sales_stage, MRR, expansion, renewal, pipeline, target,
                quota attainment, growth score, renewal probability,
                forecast confidence, year, month]
    target   = forecasted_revenue_next_quarter

A :class:`~sklearn.ensemble.GradientBoostingRegressor` is wrapped in a
``Pipeline`` together with a :class:`~sklearn.compose.ColumnTransformer`
that one-hot encodes categorical fields and scales numerics. The trained
pipeline is exposed via the :class:`RevenueForecastModel` class which also
implements a quota attainment classifier and a revenue-risk scorer.

The intent is *not* to compete with a production-grade forecasting service
but to demonstrate a clean, reproducible ML workflow that the dashboard can
expose to business users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES: List[str] = [
    "region",
    "country",
    "sales_team",
    "subscription_plan",
    "product_line",
    "sales_stage",
    "industry",
]

NUMERIC_FEATURES: List[str] = [
    "monthly_recurring_revenue_eur",
    "expansion_revenue_eur",
    "renewal_revenue_eur",
    "pipeline_value_eur",
    "closed_won_value_eur",
    "sales_target_eur",
    "quota_attainment_pct",
    "customer_growth_score",
    "renewal_probability_pct",
    "forecast_confidence_score",
    "year",
    "month_num",
]


@dataclass
class ModelMetrics:
    """Container for the metrics surfaced in the dashboard sidebar."""

    train_size: int
    test_size: int
    mae_eur: float
    r2: float
    quota_classifier_accuracy: float


class RevenueForecastModel:
    """End-to-end forecasting + quota + risk model bundle.

    Parameters
    ----------
    random_state:
        Seed for any stochastic steps. Surfaced so unit tests can reproduce
        results.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.regressor: Pipeline | None = None
        self.classifier: Pipeline | None = None
        self.metrics: ModelMetrics | None = None
        self._feature_columns: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ModelMetrics:
        """Train the regression + classification pipelines.

        The regressor predicts the *next-quarter* revenue per record. The
        classifier predicts whether a given record will *meet* quota
        (``quota_attainment_pct >= 100``).
        """
        feature_df = self._prepare_features(df)
        y_reg = df["forecasted_revenue_next_quarter"].astype(float).values
        y_cls = (df["quota_attainment_pct"] >= 100).astype(int).values

        x_train, x_test, y_train, y_test = train_test_split(
            feature_df,
            y_reg,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.regressor = self._build_regressor()
        self.regressor.fit(x_train, y_train)
        preds = self.regressor.predict(x_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # Train quota classifier on the full feature frame; we only need a
        # quick proxy of "are we likely to hit quota?" for the dashboard.
        self.classifier = self._build_classifier()
        cls_x_train, cls_x_test, cls_y_train, cls_y_test = train_test_split(
            feature_df,
            y_cls,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_cls if len(np.unique(y_cls)) > 1 else None,
        )
        self.classifier.fit(cls_x_train, cls_y_train)
        cls_acc = float(self.classifier.score(cls_x_test, cls_y_test))

        self.metrics = ModelMetrics(
            train_size=int(len(x_train)),
            test_size=int(len(x_test)),
            mae_eur=mae,
            r2=r2,
            quota_classifier_accuracy=cls_acc,
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_quarter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict next-quarter revenue for every record in ``df``.

        Returns a copy of ``df`` enriched with two new columns:
        ``predicted_revenue_next_quarter`` and ``predicted_meets_quota``.
        """
        if self.regressor is None or self.classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        feature_df = self._prepare_features(df)
        out = df.copy()
        out["predicted_revenue_next_quarter"] = self.regressor.predict(feature_df)
        out["predicted_revenue_next_quarter"] = out[
            "predicted_revenue_next_quarter"
        ].clip(lower=0).round(2)
        out["predicted_meets_quota"] = self.classifier.predict(feature_df)
        out["quota_attainment_probability"] = self.classifier.predict_proba(
            feature_df
        )[:, 1].round(3)
        return out

    def quota_attainment_trend(self, df_pred: pd.DataFrame) -> pd.DataFrame:
        """Roll the per-record quota predictions up to a monthly trend."""
        return (
            df_pred.groupby("month_label", dropna=False)
            .agg(
                actual_quota_avg=("quota_attainment_pct", "mean"),
                predicted_quota_prob=("quota_attainment_probability", "mean"),
                records=("record_id", "count"),
            )
            .reset_index()
            .sort_values("month_label")
        )

    def risk_accounts(self, df_pred: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Rank accounts most likely to underdeliver next quarter.

        Combines the regression *gap* (forecast vs. predicted) with the
        quota classifier's probability of *missing* quota and normalises
        against current MRR so we surface meaningful, sizeable risks.
        """
        grouped = (
            df_pred.groupby(["customer_name", "region", "industry"], dropna=False)
            .agg(
                mrr=("monthly_recurring_revenue_eur", "sum"),
                forecast=("forecasted_revenue_next_quarter", "sum"),
                predicted=("predicted_revenue_next_quarter", "sum"),
                miss_quota_prob=(
                    "quota_attainment_probability",
                    lambda s: float(1.0 - np.mean(s)),
                ),
                churn=("churned_revenue_eur", "sum"),
            )
            .reset_index()
        )
        grouped["forecast_gap_eur"] = (
            grouped["forecast"] - grouped["predicted"]
        ).round(2)
        # Risk score: weighted combo of forecast gap, miss-quota probability,
        # and churn already booked. Higher is worse.
        gap_norm = _safe_minmax(grouped["forecast_gap_eur"].clip(lower=0))
        churn_norm = _safe_minmax(grouped["churn"])
        grouped["ml_risk_score"] = (
            0.5 * gap_norm + 0.3 * grouped["miss_quota_prob"] + 0.2 * churn_norm
        ).round(3)
        return (
            grouped.sort_values("ml_risk_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slice the dataset down to the feature columns the model expects.

        We also derive ``month_num`` and ``year`` directly from
        ``reporting_month`` so seasonality is encoded.
        """
        out = df.copy()
        if "reporting_month" in out.columns:
            out["month_num"] = pd.to_datetime(out["reporting_month"]).dt.month
            if "year" not in out.columns:
                out["year"] = pd.to_datetime(out["reporting_month"]).dt.year
        else:
            out["month_num"] = 1
            out["year"] = 0

        for col in self._feature_columns:
            if col not in out.columns:
                out[col] = 0 if col in NUMERIC_FEATURES else "unknown"

        return out[self._feature_columns]

    def _build_regressor(self) -> Pipeline:
        """Construct the regression pipeline."""
        pre = self._build_preprocessor()
        return Pipeline(
            steps=[
                ("preprocess", pre),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _build_classifier(self) -> Pipeline:
        """Construct the quota-attainment classification pipeline."""
        pre = self._build_preprocessor()
        return Pipeline(
            steps=[
                ("preprocess", pre),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _build_preprocessor(self) -> ColumnTransformer:
        """Categorical OHE + numeric scaling, version-agnostic for sklearn."""
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        return ColumnTransformer(
            transformers=[
                ("cat", ohe, CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES),
            ]
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _safe_minmax(s: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1] without blowing up on constants."""
    s = s.astype(float)
    span = s.max() - s.min()
    if span <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / span


def get_feature_importances(model: RevenueForecastModel) -> Dict[str, float]:
    """Return a mapping of *expanded* feature name -> importance.

    Useful for surfacing model interpretability inside the Streamlit app.
    Falls back to the numeric/categorical column names when the underlying
    estimator does not expose ``feature_importances_``.
    """
    if model.regressor is None:
        return {}

    estimator = model.regressor.named_steps["model"]
    pre: ColumnTransformer = model.regressor.named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out().tolist()
    except Exception:  # pragma: no cover - sklearn very old
        names = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None or len(importances) != len(names):
        return {}
    pairs = sorted(zip(names, importances), key=lambda kv: kv[1], reverse=True)
    return {name: float(score) for name, score in pairs[:20]}
