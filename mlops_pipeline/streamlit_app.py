"""Streamlit app for CustomerChurnX predictions and monitoring."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from src.model_deploy import predict_records
from src.model_monitoring import generate_monitoring_report


API_URL = os.getenv("CUSTOMER_CHURNX_API_URL", "http://127.0.0.1:5000/predict")
REGIONS = ["North", "South", "East", "West", "Center"]
CHANNELS = ["web", "store", "app"]
PLANS = ["Basic", "Plus", "Premium"]


def build_customer_record() -> dict:
    """Render the prediction form and return a model-ready record."""
    left, right = st.columns(2)
    with left:
        signup_month = st.number_input("Mes de registro", 1, 24, 10)
        age = st.number_input("Edad", 18, 100, 40)
        tenure_months = st.number_input("Meses como cliente", 0, 120, 12)
        region = st.selectbox("Region", REGIONS)
        channel = st.selectbox("Canal", CHANNELS)
        plan = st.selectbox("Plan", PLANS, index=1)
    with right:
        sessions_week = st.number_input("Sesiones por semana", 0, 50, 3)
        avg_session_min = st.number_input("Minutos promedio por sesion", 0.0, 120.0, 8.5)
        notif_click_rate = st.slider("Tasa de clic en notificaciones", 0.0, 1.0, 0.10)
        support_tickets_3m = st.number_input("Tickets soporte ultimos 3 meses", 0, 20, 1)
        discount_pct_3m = st.slider("Descuento ultimos 3 meses", 0.0, 1.0, 0.05)
        late_payments_6m = st.number_input("Pagos atrasados ultimos 6 meses", 0, 12, 0)
        auto_renew = st.checkbox("Renovacion automatica", value=True)

    return {
        "signup_month": int(signup_month),
        "age": int(age),
        "tenure_months": int(tenure_months),
        "region": region,
        "channel": channel,
        "plan": plan,
        "sessions_week": int(sessions_week),
        "avg_session_min": float(avg_session_min),
        "notif_click_rate": float(notif_click_rate),
        "support_tickets_3m": int(support_tickets_3m),
        "discount_pct_3m": float(discount_pct_3m),
        "late_payments_6m": int(late_payments_6m),
        "auto_renew": int(auto_renew),
    }


def request_api_predictions(records: list[dict]) -> list[dict]:
    """Call the FastAPI prediction endpoint used by the Docker service."""
    payload = json.dumps({"records": records}).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["results"]


def run_predictions(records: list[dict]) -> tuple[list[dict], str]:
    """Predict through the API first, then fall back to local inference."""
    try:
        return request_api_predictions(records), "API FastAPI"
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        st.warning(
            f"No se pudo conectar con la API en {API_URL}. "
            f"Se usa el modelo local. Detalle: {exc}"
        )
        return predict_records(records), "modelo local"


def render_single_prediction() -> None:
    """Render single-customer prediction controls and result."""
    st.subheader("Prediccion individual")
    record = build_customer_record()
    if st.button("Predecir churn", type="primary"):
        predictions, source = run_predictions([record])
        result = predictions[0]
        probability = result["churn_probability"]
        prediction_text = "Riesgo de churn" if result["prediction"] == 1 else "Sin churn esperado"
        st.caption(f"Origen de la prediccion: {source}")
        st.metric("Resultado", prediction_text)
        st.metric("Probabilidad de churn", f"{probability:.2%}")
        st.json({"entrada": record, "salida": result})


def render_batch_prediction() -> None:
    """Render CSV upload prediction workflow."""
    st.subheader("Prediccion por CSV")
    uploaded_file = st.file_uploader("Carga un CSV con columnas del modelo", type=["csv"])
    if uploaded_file is None:
        return

    input_df = pd.read_csv(uploaded_file)
    st.dataframe(input_df.head(20), width="stretch")
    if st.button("Predecir archivo CSV"):
        records = input_df.drop(columns=["churn"], errors="ignore").to_dict(orient="records")
        prediction_records, source = run_predictions(records)
        st.caption(f"Origen de la prediccion: {source}")
        predictions = pd.DataFrame(prediction_records)
        st.dataframe(predictions, width="stretch")
        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicciones",
            data=csv,
            file_name="predicciones_churn.csv",
            mime="text/csv",
        )


def render_monitoring() -> None:
    """Render drift monitoring controls and result."""
    st.subheader("Monitoreo de data drift")
    report = generate_monitoring_report()
    drift_count = int(report["drift_detected"].sum())
    st.metric("Variables con drift", drift_count)
    st.dataframe(report.round(4), width="stretch")
    chart_data = report.set_index("feature")["psi"].fillna(0).sort_values(ascending=False)
    st.bar_chart(chart_data)
    if drift_count:
        st.warning("Se recomienda revisar el pipeline y evaluar reentrenamiento.")
    else:
        st.success("No se detectan alertas criticas de drift.")


def main() -> None:
    st.set_page_config(page_title="CustomerChurnX", layout="wide")
    st.title("CustomerChurnX")
    prediction_tab, monitoring_tab = st.tabs(["Prediccion", "Monitoreo"])
    with prediction_tab:
        render_single_prediction()
        st.divider()
        render_batch_prediction()
    with monitoring_tab:
        render_monitoring()


main()
