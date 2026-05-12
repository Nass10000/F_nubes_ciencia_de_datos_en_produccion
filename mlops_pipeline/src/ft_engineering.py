"""Utilidades de limpieza y creacion de variables para CustomerChurnX."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Base_de_datos.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
TARGET = "churn"
ID_COLUMN = "customer_id"
RANDOM_STATE = 42


def calculate_low_engagement_threshold(df: pd.DataFrame) -> float:
    """Calcula el umbral base de engagement semanal usado en las features."""
    engagement_minutes = df["sessions_week"] * df["avg_session_min"]
    return float(engagement_minutes.median())


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carga la base principal y valida que tenga las columnas obligatorias."""
    df = pd.read_csv(path)
    # Esta validacion evita seguir el pipeline con una base incorrecta o incompleta.
    required_columns = {
        ID_COLUMN,
        "signup_month",
        "age",
        "tenure_months",
        "region",
        "channel",
        "plan",
        "sessions_week",
        "avg_session_min",
        "notif_click_rate",
        "support_tickets_3m",
        "discount_pct_3m",
        "late_payments_6m",
        "auto_renew",
        TARGET,
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia la base: corrige tipos, elimina duplicados y ajusta rangos validos."""
    clean = df.copy()
    clean = clean.drop_duplicates(subset=[ID_COLUMN]).reset_index(drop=True)

    # Estas columnas se convierten a numericas porque alimentan el modelo.
    numeric_columns = [
        "signup_month",
        "age",
        "tenure_months",
        "sessions_week",
        "avg_session_min",
        "notif_click_rate",
        "support_tickets_3m",
        "discount_pct_3m",
        "late_payments_6m",
        "auto_renew",
        TARGET,
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["signup_month"] = clean["signup_month"].clip(lower=1, upper=24)
    clean["age"] = clean["age"].clip(lower=18, upper=100)
    clean["tenure_months"] = clean["tenure_months"].clip(lower=0)
    clean["sessions_week"] = clean["sessions_week"].clip(lower=0)
    clean["avg_session_min"] = clean["avg_session_min"].clip(lower=0)
    clean["notif_click_rate"] = clean["notif_click_rate"].clip(lower=0, upper=1)
    clean["discount_pct_3m"] = clean["discount_pct_3m"].clip(lower=0, upper=1)
    clean["auto_renew"] = clean["auto_renew"].fillna(0).astype(int).clip(lower=0, upper=1)
    clean[TARGET] = clean[TARGET].fillna(0).astype(int).clip(lower=0, upper=1)

    for column in ["region", "channel", "plan"]:
        clean[column] = clean[column].fillna("Unknown").astype(str).str.strip()

    return clean


def add_features(
    df: pd.DataFrame, low_engagement_threshold: float | None = None
) -> pd.DataFrame:
    """Crea variables derivadas que ayudan al modelo a detectar churn."""
    features = df.copy()
    # Estas features resumen uso, soporte y comportamiento de pago del cliente.
    features["engagement_minutes_week"] = (
        features["sessions_week"] * features["avg_session_min"]
    )
    features["tickets_per_tenure"] = features["support_tickets_3m"] / (
        features["tenure_months"].clip(lower=1)
    )
    features["late_payment_flag"] = (features["late_payments_6m"] > 0).astype(int)
    features["high_discount_flag"] = (features["discount_pct_3m"] >= 0.20).astype(int)
    features["early_tenure_flag"] = (features["tenure_months"] < 6).astype(int)
    if low_engagement_threshold is None:
        low_engagement_threshold = float(features["engagement_minutes_week"].median())
    features["low_engagement_flag"] = (
        features["engagement_minutes_week"] < low_engagement_threshold
    ).astype(int)
    return features


def get_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Separa las columnas del modelo en numericas y categoricas."""
    excluded = {TARGET, ID_COLUMN}
    candidate_columns = [column for column in df.columns if column not in excluded]
    numerical_columns = [
        column for column in candidate_columns if is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column for column in candidate_columns if column not in numerical_columns
    ]
    return numerical_columns, categorical_columns


def build_preprocessor(
    numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Construye el preprocesamiento que se aplicara antes de entrenar o predecir."""
    # Las numericas se imputan y escalan; las categoricas se imputan y codifican.
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def make_training_dataset(
    path: str | Path = DATA_PATH, test_size: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Prepara el dataset de entrenamiento y prueba junto al preprocesador."""
    clean_df = clean_data(load_data(path))
    threshold = calculate_low_engagement_threshold(clean_df)
    df = add_features(clean_df, low_engagement_threshold=threshold)
    numerical_columns, categorical_columns = get_feature_columns(df)
    preprocessor = build_preprocessor(numerical_columns, categorical_columns)
    x = df[numerical_columns + categorical_columns]
    y = df[TARGET]
    # Stratify mantiene la proporcion de churn en train y test.
    return (*train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    ), preprocessor)


def save_reference_sample(path: str | Path = DATA_PATH) -> Path:
    """Guarda una muestra historica que luego se usa como referencia de monitoreo."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_df = clean_data(load_data(path))
    threshold = calculate_low_engagement_threshold(clean_df)
    df = add_features(clean_df, low_engagement_threshold=threshold)
    # La mitad historica de la base se usa como linea base para drift.
    reference = df[df["signup_month"] <= df["signup_month"].median()].copy()
    output_path = ARTIFACTS_DIR / "reference_sample.csv"
    reference.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    clean_dataframe = clean_data(load_data())
    dataframe = add_features(
        clean_dataframe,
        low_engagement_threshold=calculate_low_engagement_threshold(clean_dataframe),
    )
    save_reference_sample()
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns):,}")
    print(f"Target rate: {dataframe[TARGET].mean():.3f}")
