from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ft_engineering import (
    DATE_COLUMN,
    ID_COLUMN,
    TARGET,
    add_features,
    clean_data,
    load_data,
)
import model_monitoring
from model_deploy import predict_records
from model_monitoring import resolve_monitoring_mode


def test_dataset_schema_and_target():
    """Verifica que la base correcta del PI tenga estructura y objetivo validos."""
    df = clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv"))

    assert len(df) == 10763
    assert TARGET == "Pago_atiempo"
    assert TARGET in df.columns
    assert ID_COLUMN in df.columns
    assert df[ID_COLUMN].is_unique
    assert set(df[TARGET].unique()).issubset({0, 1})
    assert df[DATE_COLUMN].notna().all()


def test_feature_engineering_creates_expected_columns():
    """Comprueba que la ingenieria financiera cree las columnas del modelo."""
    df = add_features(clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv")))
    expected = {
        "prestamo_year",
        "prestamo_month",
        "cuota_salario_ratio",
        "capital_salario_ratio",
        "carga_total_salario_ratio",
        "capital_por_mes",
        "mora_ratio",
        "saldo_principal_ratio",
        "has_mora",
        "has_codeudor_mora",
        "creditos_total",
        "datacredito_income_gap",
    }
    assert expected.issubset(df.columns)


def test_prediction_contract():
    """Valida la forma minima que debe devolver una prediccion."""
    df = clean_data(load_data(PROJECT_DIR / "Base_de_datos.csv"))
    record = (
        df.drop(columns=[TARGET, ID_COLUMN])
        .head(1)
        .assign(fecha_prestamo=lambda data: data[DATE_COLUMN].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")[0]
    )

    result = predict_records([record])[0]

    assert set(result) == {
        "loan_id",
        "prediction",
        "pago_atiempo_probability",
        "riesgo_no_pago_probability",
    }
    assert result["prediction"] in {0, 1}
    assert 0 <= result["pago_atiempo_probability"] <= 1
    assert 0 <= result["riesgo_no_pago_probability"] <= 1


def test_monitoring_modes_are_distinct(tmp_path, monkeypatch):
    """Comprueba que el modo real y el modo simulado de monitoreo no se mezclen."""
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(model_monitoring, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(model_monitoring, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(model_monitoring, "DRIFT_REPORT_CSV", reports_dir / "drift_report.csv")
    monkeypatch.setattr(model_monitoring, "DRIFT_REPORT_JSON", reports_dir / "drift_report.json")

    real_report = model_monitoring.generate_monitoring_report(mode="real")
    simulated_report = model_monitoring.generate_monitoring_report(mode="simulated")

    assert resolve_monitoring_mode("real") is False
    assert resolve_monitoring_mode("simulated") is True
    assert "drift_detected" in real_report.columns
    assert int(real_report["drift_detected"].sum()) <= int(
        simulated_report["drift_detected"].sum()
    )
