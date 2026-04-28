from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ft_engineering import TARGET, add_features, clean_data, load_data
from model_deploy import predict_records


def test_dataset_schema_and_target():
    df = clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv"))
    assert TARGET in df.columns
    assert df["customer_id"].is_unique
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_feature_engineering_creates_expected_columns():
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


def test_prediction_contract():
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
