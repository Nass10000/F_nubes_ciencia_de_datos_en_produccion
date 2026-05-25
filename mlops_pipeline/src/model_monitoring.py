"""Calcula drift de datos y genera el dashboard de monitoreo."""

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
        DATE_COLUMN,
        ID_COLUMN,
        MODEL_EXCLUDED_COLUMNS,
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
        DATE_COLUMN,
        ID_COLUMN,
        MODEL_EXCLUDED_COLUMNS,
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
    """Calcula el PSI para medir cambio de distribucion en una variable numerica."""
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.nunique() <= 1 or actual.empty:
        return 0.0
    _, bin_edges = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    expected_pct = pd.cut(expected, bin_edges).value_counts(normalize=True).sort_index()
    actual_pct = pd.cut(actual, bin_edges).value_counts(normalize=True).sort_index()
    # Estos valores pequenos evitan divisiones por cero en los buckets.
    expected_pct, actual_pct = expected_pct.align(actual_pct, fill_value=0.0001)
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def categorical_drift(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Mide drift en variables categoricas con chi-cuadrado y Jensen-Shannon."""
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
    """Separa la base en ventana historica y ventana reciente para monitoreo."""
    df = add_features(clean_data(load_data(path)))
    cutoff = df[DATE_COLUMN].median()
    # Lo historico actua como referencia y lo reciente como ventana actual.
    reference = df[df[DATE_COLUMN] <= cutoff].copy()
    current = df[df[DATE_COLUMN] > cutoff].copy()
    if simulate_shift:
        # En modo demo se fuerzan cambios para mostrar un caso claro de drift.
        current = current.copy()
        current["salario_cliente"] = (current["salario_cliente"] * 0.75).round()
        current["puntaje_datacredito"] = (current["puntaje_datacredito"] * 0.90).round()
        current["saldo_mora"] = (current["saldo_mora"].fillna(0) + 250000).clip(lower=0)
        current["huella_consulta"] = (current["huella_consulta"] + 3).clip(lower=0)
        current = add_features(current)
    return reference, current


def resolve_monitoring_mode(mode: str) -> bool:
    """Traduce el modo elegido a si se debe simular drift o no."""
    options = {
        "real": False,
        "simulated": True,
        "demo": True,
    }
    try:
        return options[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported monitoring mode: {mode}") from exc


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.20,
    js_threshold: float = 0.10,
) -> pd.DataFrame:
    """Compara referencia y ventana actual para marcar variables con drift."""
    rows = []
    # Solo se monitorean variables de entrada reales; se excluyen target, IDs y fugas.
    for column in [c for c in reference.columns if c not in MODEL_EXCLUDED_COLUMNS]:
        if pd.api.types.is_numeric_dtype(reference[column]):
            psi = population_stability_index(reference[column], current[column])
            reference_values = reference[column].dropna()
            current_values = current[column].dropna()
            if reference_values.empty or current_values.empty:
                ks_stat, ks_p_value = 0.0, 1.0
            else:
                ks_stat, ks_p_value = ks_2samp(reference_values, current_values)
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


def generate_monitoring_report(
    path: str | Path = DATA_PATH,
    mode: str = "real",
) -> pd.DataFrame:
    """Genera el reporte de drift y una muestra de predicciones actuales."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    simulate_shift = resolve_monitoring_mode(mode)
    reference, current = build_reference_and_current(path, simulate_shift=simulate_shift)
    load_trained_model()
    # Se guarda una muestra de predicciones como evidencia de uso operativo.
    predictions = predict_records(current.drop(columns=[TARGET]).head(500).to_dict("records"))
    pd.DataFrame(predictions).to_csv(REPORTS_DIR / "current_predictions_sample.csv", index=False)

    report = detect_drift(reference, current)
    report.to_csv(DRIFT_REPORT_CSV, index=False)
    summary_records = json.loads(report.to_json(orient="records"))
    payload = {
        "mode": mode,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "drifted_features": report.loc[report["drift_detected"], "feature"].tolist(),
        "summary": summary_records,
    }
    DRIFT_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reference.to_csv(ARTIFACTS_DIR / "reference_sample.csv", index=False)
    return report


def streamlit_dashboard() -> None:
    """Muestra en Streamlit el dashboard de monitoreo de drift."""
    import streamlit as st

    st.set_page_config(page_title="Riesgo crediticio - Monitoring", layout="wide")
    st.title("Riesgo crediticio - Monitoreo de Data Drift")
    # Permite alternar entre el caso real y un caso demo con drift simulado.
    mode = st.radio(
        "Modo de monitoreo",
        options=["real", "simulated"],
        format_func=lambda value: "Real" if value == "real" else "Simulado",
        horizontal=True,
    )
    report = generate_monitoring_report(mode=mode)
    drift_count = int(report["drift_detected"].sum())
    if mode == "real":
        st.caption("Compara ventanas historicas y recientes de la base sin alterar los datos.")
    else:
        st.caption("Aplica una alteracion controlada para demostrar un escenario con drift.")
    st.metric("Variables con drift", drift_count)
    st.dataframe(report, width="stretch")
    st.bar_chart(report.set_index("feature")["psi"].fillna(0))
    if drift_count:
        st.warning("Se recomienda revisar el pipeline y evaluar reentrenamiento.")
    else:
        st.success("No se detectan alertas criticas de drift.")


if __name__ == "__main__":
    report_df = generate_monitoring_report()
    print(report_df.to_string(index=False))
    print(f"CSV report: {DRIFT_REPORT_CSV}")
    print(f"JSON report: {DRIFT_REPORT_JSON}")
