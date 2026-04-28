"""Data drift monitoring and optional Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

try:
    from ft_engineering import (
        ARTIFACTS_DIR,
        DATA_PATH,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        load_data,
    )
    from model_deploy import predict_records
    from model_training_evaluation import load_trained_model
except ImportError:
    from .ft_engineering import (
        ARTIFACTS_DIR,
        DATA_PATH,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        load_data,
    )
    from .model_deploy import predict_records
    from .model_training_evaluation import load_trained_model


DRIFT_REPORT_CSV = REPORTS_DIR / "drift_report.csv"
DRIFT_REPORT_JSON = REPORTS_DIR / "drift_report.json"


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """Calculate PSI for a numerical variable."""
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.nunique() <= 1 or actual.empty:
        return 0.0
    _, bin_edges = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    expected_pct = pd.cut(expected, bin_edges).value_counts(normalize=True).sort_index()
    actual_pct = pd.cut(actual, bin_edges).value_counts(normalize=True).sort_index()
    expected_pct, actual_pct = expected_pct.align(actual_pct, fill_value=0.0001)
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def categorical_drift(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Return chi-square p-value and Jensen-Shannon distance."""
    expected_counts = expected.fillna("missing").astype(str).value_counts()
    actual_counts = actual.fillna("missing").astype(str).value_counts()
    expected_counts, actual_counts = expected_counts.align(actual_counts, fill_value=0)
    contingency = np.vstack([expected_counts.values, actual_counts.values])
    _, p_value, _, _ = chi2_contingency(contingency + 1e-9)
    expected_dist = expected_counts / expected_counts.sum()
    actual_dist = actual_counts / actual_counts.sum()
    js_distance = float(jensenshannon(expected_dist, actual_dist))
    return float(p_value), js_distance


def build_reference_and_current(
    path: str | Path = DATA_PATH, simulate_shift: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into historical/current windows and simulate recent drift."""
    df = add_features(clean_data(load_data(path)))
    cutoff = df["signup_month"].median()
    reference = df[df["signup_month"] <= cutoff].copy()
    current = df[df["signup_month"] > cutoff].copy()
    if simulate_shift:
        current = current.copy()
        current["sessions_week"] = (current["sessions_week"] * 0.70).round().astype(int)
        current["avg_session_min"] = (current["avg_session_min"] * 0.80).round(2)
        current["support_tickets_3m"] = (current["support_tickets_3m"] + 1).clip(upper=6)
        current["discount_pct_3m"] = (current["discount_pct_3m"] * 1.35).clip(upper=1)
        current["low_engagement_flag"] = (
            current["sessions_week"] * current["avg_session_min"]
            < reference["engagement_minutes_week"].median()
        ).astype(int)
        current["engagement_minutes_week"] = (
            current["sessions_week"] * current["avg_session_min"]
        )
    return reference, current


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.20,
    js_threshold: float = 0.10,
) -> pd.DataFrame:
    """Compare reference and current populations and flag drift."""
    rows = []
    excluded = {TARGET, "customer_id", "signup_month"}
    for column in [c for c in reference.columns if c not in excluded]:
        if pd.api.types.is_numeric_dtype(reference[column]):
            psi = population_stability_index(reference[column], current[column])
            ks_stat, ks_p_value = ks_2samp(reference[column], current[column])
            rows.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "psi": psi,
                    "ks_stat": float(ks_stat),
                    "ks_p_value": float(ks_p_value),
                    "js_distance": np.nan,
                    "chi2_p_value": np.nan,
                    "drift_detected": bool(psi > psi_threshold or ks_stat > ks_threshold),
                }
            )
        else:
            chi2_p_value, js_distance = categorical_drift(reference[column], current[column])
            rows.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "psi": np.nan,
                    "ks_stat": np.nan,
                    "ks_p_value": np.nan,
                    "js_distance": js_distance,
                    "chi2_p_value": chi2_p_value,
                    "drift_detected": bool(js_distance > js_threshold or chi2_p_value < 0.05),
                }
            )
    return pd.DataFrame(rows).sort_values("drift_detected", ascending=False)


def generate_monitoring_report(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Create drift reports and prediction samples for operational evidence."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    reference, current = build_reference_and_current(path)
    load_trained_model()
    predictions = predict_records(current.drop(columns=[TARGET]).head(500).to_dict("records"))
    pd.DataFrame(predictions).to_csv(REPORTS_DIR / "current_predictions_sample.csv", index=False)

    report = detect_drift(reference, current)
    report.to_csv(DRIFT_REPORT_CSV, index=False)
    payload = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "drifted_features": report.loc[report["drift_detected"], "feature"].tolist(),
        "summary": report.to_dict(orient="records"),
    }
    DRIFT_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reference.to_csv(ARTIFACTS_DIR / "reference_sample.csv", index=False)
    return report


def streamlit_dashboard() -> None:
    """Render the Streamlit monitoring view."""
    import streamlit as st

    st.set_page_config(page_title="CustomerChurnX Monitoring", layout="wide")
    st.title("CustomerChurnX - Monitoreo de Data Drift")
    report = generate_monitoring_report()
    drift_count = int(report["drift_detected"].sum())
    st.metric("Variables con drift", drift_count)
    st.dataframe(report, use_container_width=True)
    st.bar_chart(report.set_index("feature")["psi"].fillna(0))
    if drift_count:
        st.warning("Se recomienda revisar el pipeline y evaluar reentrenamiento.")
    else:
        st.success("No se detectan alertas críticas de drift.")


if __name__ == "__main__":
    report_df = generate_monitoring_report()
    print(report_df.to_string(index=False))
    print(f"CSV report: {DRIFT_REPORT_CSV}")
    print(f"JSON report: {DRIFT_REPORT_JSON}")
