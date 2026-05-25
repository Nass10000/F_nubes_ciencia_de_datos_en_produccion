"""Expone el modelo por API y permite correr predicciones por lote."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ft_engineering import (
        DATA_PATH,
        ID_COLUMN,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        load_data,
    )
    from model_training_evaluation import load_trained_model
except ImportError:
    from .ft_engineering import (
        DATA_PATH,
        ID_COLUMN,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        load_data,
    )
    from .model_training_evaluation import load_trained_model

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # Allows local batch inference without the API dependency.
    FastAPI = None
    HTTPException = Exception
    BaseModel = object


class PredictionRequest(BaseModel):
    """Define el formato de entrada para una o varias solicitudes de credito."""

    records: list[dict[str, Any]] | None = None
    record: dict[str, Any] | None = None


def _prepare_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Normaliza los registros de entrada al formato esperado por el pipeline."""
    # La idea es que la API trate los datos igual que en entrenamiento.
    df = pd.DataFrame(records)
    if TARGET not in df.columns:
        df[TARGET] = 0
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"REQ{i:06d}" for i in range(len(df))]
    return add_features(clean_data(df))


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Devuelve si se espera pago a tiempo y su probabilidad por registro."""
    if not records:
        raise ValueError("At least one record is required")
    model = load_trained_model()
    prepared = _prepare_records(records)
    # La clase final se obtiene usando 0.5 como umbral de decision.
    probabilities = model.predict_proba(prepared.drop(columns=[TARGET, ID_COLUMN]))[
        :, 1
    ]
    predictions = (probabilities >= 0.5).astype(int)
    return [
        {
            "loan_id": loan_id,
            "prediction": int(prediction),
            "pago_atiempo_probability": round(float(probability), 4),
            "riesgo_no_pago_probability": round(float(1 - probability), 4),
        }
        for loan_id, prediction, probability in zip(
            prepared[ID_COLUMN], predictions, probabilities
        )
    ]


def batch_predict(input_path: str | Path, output_path: str | Path) -> Path:
    """Lee la base de entrada, genera predicciones y guarda el resultado."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    results = predict_records(df.drop(columns=[TARGET]).to_dict(orient="records"))
    output = pd.DataFrame(results)
    output_path = Path(output_path)
    output.to_csv(output_path, index=False)
    return output_path


if FastAPI is not None:
    # FastAPI publica el modelo como un servicio sencillo de prediccion.
    app = FastAPI(
        title="Credit Risk Payment Prediction API",
        version="1.0.0",
        description="API para predecir si un credito se pagara a tiempo.",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    @app.post("/prediccion")
    def predict(payload: PredictionRequest) -> dict[str, Any]:
        """Atiende peticiones de prediccion individual o por lote."""
        try:
            # El endpoint acepta un solo registro o una lista de registros.
            records = payload.records or ([payload.record] if payload.record else [])
            return {"results": predict_records(records)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
else:
    app = None


def main() -> None:
    """Permite ejecutar prediccion batch desde la terminal."""
    parser = argparse.ArgumentParser(description="Batch inference for credit payment risk.")
    parser.add_argument("--input", default=str(DATA_PATH), help="Input data path.")
    parser.add_argument(
        "--output",
        default=str(REPORTS_DIR / "batch_predictions.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    output_path = batch_predict(args.input, args.output)
    print(f"Predictions saved at: {output_path}")


if __name__ == "__main__":
    main()
