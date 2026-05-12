from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ft_engineering import (
    TARGET,
    add_features,
    calculate_low_engagement_threshold,
    clean_data,
    load_data,
)
from model_deploy import predict_records
from model_monitoring import generate_monitoring_report, resolve_monitoring_mode


def test_dataset_schema_and_target():
    """Verifica que la base tenga la estructura minima necesaria."""
    df = clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv"))
    assert TARGET in df.columns
    assert df["customer_id"].is_unique
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_feature_engineering_creates_expected_columns():
    """Comprueba que la ingenieria de variables cree las columnas esperadas."""
    df = add_features(clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv")))
    expected = {
        "engagement_minutes_week",
        "tickets_per_tenure",
        "late_payment_flag",
        "high_discount_flag",
        "early_tenure_flag",
        "low_engagement_flag",
    }
    assert expected.issubset(df.columns)


def test_low_engagement_flag_uses_reference_threshold():
    """Valida que el umbral de bajo engagement sea consistente con entrenamiento."""
    df = clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv"))
    threshold = calculate_low_engagement_threshold(df)
    sample = df.head(1).copy()
    sample["sessions_week"] = 3
    sample["avg_session_min"] = 8.5

    featured = add_features(sample, low_engagement_threshold=threshold)

    assert featured["low_engagement_flag"].iloc[0] == 1


def test_prediction_contract():
    """Valida la forma minima que debe devolver una prediccion."""
    record = {
        "signup_month": 10,
        "age": 40,
        "tenure_months": 12,
        "region": "North",
        "channel": "web",
        "plan": "Plus",
        "sessions_week": 3,
        "avg_session_min": 8.5,
        "notif_click_rate": 0.1,
        "support_tickets_3m": 1,
        "discount_pct_3m": 0.05,
        "late_payments_6m": 0,
        "auto_renew": 1,
    }
    result = predict_records([record])[0]
    assert set(result) == {"customer_id", "prediction", "churn_probability"}
    assert result["prediction"] in {0, 1}
    assert 0 <= result["churn_probability"] <= 1


def test_monitoring_modes_are_distinct():
    """Comprueba que el modo real y el modo simulado de monitoreo no se mezclen."""
    real_report = generate_monitoring_report(mode="real")
    simulated_report = generate_monitoring_report(mode="simulated")

    assert resolve_monitoring_mode("real") is False
    assert resolve_monitoring_mode("simulated") is True
    assert int(real_report["drift_detected"].sum()) <= int(simulated_report["drift_detected"].sum())
