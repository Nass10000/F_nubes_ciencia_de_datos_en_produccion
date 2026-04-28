"""FastAPI deployment entrypoint and batch inference helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ft_engineering import DATA_PATH, REPORTS_DIR, add_features, clean_data, load_data
    from model_training_evaluation import load_trained_model
except ImportError:
    from .ft_engineering import DATA_PATH, REPORTS_DIR, add_features, clean_data, load_data
    from .model_training_evaluation import load_trained_model

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # Allows local batch inference without the API dependency.
    FastAPI = None
    HTTPException = Exception
    BaseModel = object
    Field = None


class PredictionRequest(BaseModel):
    """Flexible request schema for one or many customer records."""

    records: list[dict[str, Any]] | None = None
    record: dict[str, Any] | None = None


def _prepare_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "churn" not in df.columns:
        df["churn"] = 0
    if "customer_id" not in df.columns:
        df["customer_id"] = [f"REQ{i:06d}" for i in range(len(df))]
    return add_features(clean_data(df))


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return churn prediction and probability for incoming records."""
    if not records:
        raise ValueError("At least one record is required")
    model = load_trained_model()
    prepared = _prepare_records(records)
    probabilities = model.predict_proba(prepared.drop(columns=["churn", "customer_id"]))[
        :, 1
    ]
    predictions = (probabilities >= 0.5).astype(int)
    return [
        {
            "customer_id": customer_id,
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
        }
        for customer_id, prediction, probability in zip(
            prepared["customer_id"], predictions, probabilities
        )
    ]


def batch_predict(input_path: str | Path, output_path: str | Path) -> Path:
    """Run batch predictions from a CSV file and persist the result."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    results = predict_records(df.drop(columns=["churn"]).to_dict(orient="records"))
    output = pd.DataFrame(results)
    output_path = Path(output_path)
    output.to_csv(output_path, index=False)
    return output_path


if FastAPI is not None:
    app = FastAPI(
        title="CustomerChurnX Churn Prediction API",
        version="1.0.0",
        description="API para disponibilizar el mejor modelo supervisado del PI M5.",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    @app.post("/prediccion")
    def predict(payload: PredictionRequest) -> dict[str, Any]:
        try:
            records = payload.records or ([payload.record] if payload.record else [])
            return {"results": predict_records(records)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
else:
    app = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference for CustomerChurnX.")
    parser.add_argument("--input", default=str(DATA_PATH), help="Input CSV path.")
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
