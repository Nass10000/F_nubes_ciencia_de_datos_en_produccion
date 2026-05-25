"""Interfaz Streamlit para prediccion y monitoreo del PI de riesgo crediticio."""

from __future__ import annotations

import json
import os
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from src.model_deploy import predict_records
from src.model_monitoring import generate_monitoring_report


API_URL = os.getenv("CREDIT_RISK_API_URL", "http://127.0.0.1:5000/predict")
TIPOS_CREDITO = ["4", "6", "7", "9", "10", "68"]
TIPOS_LABORALES = ["Empleado", "Independiente"]
TENDENCIAS_INGRESOS = ["Creciente", "Estable", "Decreciente", "Desconocido"]


def build_credit_record() -> dict:
    """Construye una solicitud de credito con los campos usados por el modelo."""
    left, right = st.columns(2)
    with left:
        tipo_credito = st.selectbox("Tipo de credito", TIPOS_CREDITO)
        fecha_prestamo = st.date_input("Fecha del prestamo", value=date(2025, 6, 1))
        capital_prestado = st.number_input("Capital prestado", 0.0, 100_000_000.0, 3_500_000.0)
        plazo_meses = st.number_input("Plazo en meses", 1, 120, 12)
        edad_cliente = st.number_input("Edad del cliente", 18, 100, 40)
        tipo_laboral = st.selectbox("Tipo laboral", TIPOS_LABORALES)
        salario_cliente = st.number_input("Salario del cliente", 0.0, 100_000_000.0, 3_000_000.0)
        total_otros_prestamos = st.number_input(
            "Total otros prestamos", 0.0, 100_000_000.0, 500_000.0
        )
        cuota_pactada = st.number_input("Cuota pactada", 0.0, 20_000_000.0, 300_000.0)
        puntaje = st.number_input("Puntaje interno", 0.0, 100.0, 80.0)
        puntaje_datacredito = st.number_input("Puntaje Datacredito", 0.0, 1000.0, 720.0)
    with right:
        cant_creditosvigentes = st.number_input("Creditos vigentes", 0, 100, 3)
        huella_consulta = st.number_input("Consultas recientes", 0, 100, 2)
        saldo_mora = st.number_input("Saldo en mora", 0.0, 100_000_000.0, 0.0)
        saldo_total = st.number_input("Saldo total", 0.0, 100_000_000.0, 50_000.0)
        saldo_principal = st.number_input("Saldo principal", 0.0, 100_000_000.0, 50_000.0)
        saldo_mora_codeudor = st.number_input("Mora codeudor", 0.0, 100_000_000.0, 0.0)
        creditos_sector_financiero = st.number_input("Creditos sector financiero", 0, 100, 2)
        creditos_sector_cooperativo = st.number_input("Creditos sector cooperativo", 0, 100, 0)
        creditos_sector_real = st.number_input("Creditos sector real", 0, 100, 1)
        promedio_ingresos_datacredito = st.number_input(
            "Promedio ingresos Datacredito", 0.0, 100_000_000.0, 2_800_000.0
        )
        tendencia_ingresos = st.selectbox("Tendencia de ingresos", TENDENCIAS_INGRESOS)

    return {
        "tipo_credito": tipo_credito,
        "fecha_prestamo": fecha_prestamo.isoformat(),
        "capital_prestado": float(capital_prestado),
        "plazo_meses": int(plazo_meses),
        "edad_cliente": int(edad_cliente),
        "tipo_laboral": tipo_laboral,
        "salario_cliente": float(salario_cliente),
        "total_otros_prestamos": float(total_otros_prestamos),
        "cuota_pactada": float(cuota_pactada),
        "puntaje": float(puntaje),
        "puntaje_datacredito": float(puntaje_datacredito),
        "cant_creditosvigentes": int(cant_creditosvigentes),
        "huella_consulta": int(huella_consulta),
        "saldo_mora": float(saldo_mora),
        "saldo_total": float(saldo_total),
        "saldo_principal": float(saldo_principal),
        "saldo_mora_codeudor": float(saldo_mora_codeudor),
        "creditos_sectorFinanciero": int(creditos_sector_financiero),
        "creditos_sectorCooperativo": int(creditos_sector_cooperativo),
        "creditos_sectorReal": int(creditos_sector_real),
        "promedio_ingresos_datacredito": float(promedio_ingresos_datacredito),
        "tendencia_ingresos": tendencia_ingresos,
    }


def request_api_predictions(records: list[dict]) -> list[dict]:
    """Envia registros a FastAPI y devuelve sus predicciones."""
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
    """Usa FastAPI si esta disponible; si no, predice con el modelo local."""
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
    """Muestra el formulario principal de prediccion individual."""
    st.subheader("Prediccion individual")
    record = build_credit_record()
    if st.button("Predecir pago", type="primary"):
        predictions, source = run_predictions([record])
        result = predictions[0]
        pago_prob = result["pago_atiempo_probability"]
        riesgo_prob = result["riesgo_no_pago_probability"]
        prediction_text = (
            "Pago a tiempo esperado" if result["prediction"] == 1 else "Riesgo de no pago"
        )
        st.caption(f"Origen de la prediccion: {source}")
        st.metric("Resultado", prediction_text)
        st.metric("Probabilidad de pago a tiempo", f"{pago_prob:.2%}")
        st.metric("Riesgo de no pago", f"{riesgo_prob:.2%}")
        st.json({"entrada": record, "salida": result})


def render_batch_prediction() -> None:
    """Permite subir un CSV y generar predicciones por lote."""
    st.subheader("Prediccion por CSV")
    uploaded_file = st.file_uploader("Carga un CSV con columnas del modelo", type=["csv"])
    if uploaded_file is None:
        return

    input_df = pd.read_csv(uploaded_file)
    st.dataframe(input_df.head(20), width="stretch")
    if st.button("Predecir archivo CSV"):
        records = input_df.drop(columns=["Pago_atiempo"], errors="ignore").to_dict(
            orient="records"
        )
        prediction_records, source = run_predictions(records)
        st.caption(f"Origen de la prediccion: {source}")
        predictions = pd.DataFrame(prediction_records)
        st.dataframe(predictions, width="stretch")
        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicciones",
            data=csv,
            file_name="predicciones_credito.csv",
            mime="text/csv",
        )


def render_monitoring() -> None:
    """Muestra las metricas de data drift del avance 3."""
    st.subheader("Monitoreo de data drift")
    mode = st.radio(
        "Modo de monitoreo",
        options=["real", "simulated"],
        format_func=lambda value: "Real" if value == "real" else "Simulado",
        horizontal=True,
        key="monitoring_mode",
    )
    report = generate_monitoring_report(mode=mode)
    drift_count = int(report["drift_detected"].sum())
    if mode == "real":
        st.caption("Compara la ventana historica contra la ventana reciente de la base.")
    else:
        st.caption("Altera variables de forma controlada para demostrar deteccion de drift.")
    st.metric("Variables con drift", drift_count)
    st.dataframe(report.round(4), width="stretch")
    st.bar_chart(report.set_index("feature")["psi"].fillna(0).sort_values(ascending=False))
    if drift_count:
        st.warning("Se recomienda revisar el pipeline y evaluar reentrenamiento.")
    else:
        st.success("No se detectan alertas criticas de drift.")


def main() -> None:
    """Organiza la app en pestanas para prediccion y monitoreo."""
    st.set_page_config(page_title="PI Riesgo Crediticio", layout="wide")
    st.title("PI Riesgo Crediticio")
    prediction_tab, monitoring_tab = st.tabs(["Prediccion", "Monitoreo"])
    with prediction_tab:
        render_single_prediction()
        st.divider()
        render_batch_prediction()
    with monitoring_tab:
        render_monitoring()


main()
