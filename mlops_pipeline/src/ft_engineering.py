"""Utilidades de limpieza y creacion de variables para riesgo crediticio."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile
import xml.etree.ElementTree as ET

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
TARGET = "Pago_atiempo"
ID_COLUMN = "loan_id"
DATE_COLUMN = "fecha_prestamo"
RANDOM_STATE = 42
MODEL_EXCLUDED_COLUMNS = {
    TARGET,
    ID_COLUMN,
    DATE_COLUMN,
    "puntaje",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "mora_ratio",
    "saldo_principal_ratio",
    "has_mora",
    "has_codeudor_mora",
}

REQUIRED_COLUMNS = {
    "tipo_credito",
    DATE_COLUMN,
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    TARGET,
}


def _column_index(cell_ref: str) -> int:
    """Convierte una referencia de Excel como A1 o AA10 a indice numerico."""
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index - 1


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    """Lee el Excel oficial cuando openpyxl no esta instalado localmente."""
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def cell_value(cell, shared_strings: list[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "s":
            value = cell.find("a:v", namespace)
            return shared_strings[int(value.text)] if value is not None else ""
        if cell_type == "inlineStr":
            return "".join(text.text or "" for text in cell.findall(".//a:t", namespace))
        value = cell.find("a:v", namespace)
        return value.text if value is not None and value.text is not None else ""

    with ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", namespace))
                )

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("a:sheetData/a:row", namespace):
            values = []
            for cell in row.findall("a:c", namespace):
                index = _column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")
                values[index] = cell_value(cell, shared_strings)
            rows.append(values)

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(rows[1:], columns=rows[0])


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carga la base oficial y valida que tenga las columnas obligatorias."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except ImportError:
            df = _read_xlsx_without_openpyxl(path)
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia la base: corrige tipos, fechas, duplicados y rangos validos."""
    clean = df.copy()
    clean = clean.replace(r"^\s*$", pd.NA, regex=True)
    clean = clean.drop_duplicates().reset_index(drop=True)
    if ID_COLUMN not in clean.columns:
        clean.insert(0, ID_COLUMN, [f"REQ{i:06d}" for i in range(len(clean))])

    numeric_columns = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "puntaje",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "promedio_ingresos_datacredito",
        TARGET,
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    numeric_dates = pd.to_numeric(clean[DATE_COLUMN], errors="coerce")
    serial_dates = pd.to_datetime(
        numeric_dates, unit="D", origin="1899-12-30", errors="coerce"
    )
    parsed_dates = pd.to_datetime(
        clean[DATE_COLUMN].where(numeric_dates.isna()), errors="coerce"
    )
    clean[DATE_COLUMN] = serial_dates.where(numeric_dates.notna(), parsed_dates)

    clean["plazo_meses"] = clean["plazo_meses"].clip(lower=1)
    clean["edad_cliente"] = clean["edad_cliente"].clip(lower=18)
    clean["capital_prestado"] = clean["capital_prestado"].clip(lower=0)
    clean["salario_cliente"] = clean["salario_cliente"].clip(lower=0)
    clean["total_otros_prestamos"] = clean["total_otros_prestamos"].clip(lower=0)
    clean["cuota_pactada"] = clean["cuota_pactada"].clip(lower=0)
    clean["saldo_mora"] = clean["saldo_mora"].clip(lower=0)
    clean["saldo_total"] = clean["saldo_total"].clip(lower=0)
    clean[TARGET] = clean[TARGET].fillna(0).astype(int).clip(lower=0, upper=1)

    clean["tipo_credito"] = (
        pd.to_numeric(clean["tipo_credito"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "Desconocido")
    )
    for column in ["tipo_laboral", "tendencia_ingresos"]:
        clean[column] = clean[column].fillna("Desconocido").astype(str).str.strip()
    clean["tendencia_ingresos"] = clean["tendencia_ingresos"].where(
        clean["tendencia_ingresos"].isin(["Creciente", "Decreciente", "Estable"]),
        "Desconocido",
    )

    return clean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables financieras que ayudan a predecir pago a tiempo."""
    features = df.copy()
    salary = features["salario_cliente"].clip(lower=1)
    plazo = features["plazo_meses"].clip(lower=1)
    saldo_total = features["saldo_total"].clip(lower=1)

    features["prestamo_year"] = features[DATE_COLUMN].dt.year
    features["prestamo_month"] = features[DATE_COLUMN].dt.month
    features["cuota_salario_ratio"] = features["cuota_pactada"] / salary
    features["capital_salario_ratio"] = features["capital_prestado"] / salary
    features["otros_prestamos_salario_ratio"] = features["total_otros_prestamos"] / salary
    features["carga_total_salario_ratio"] = (
        features["cuota_pactada"] + features["total_otros_prestamos"]
    ) / salary
    features["capital_por_mes"] = features["capital_prestado"] / plazo
    features["mora_ratio"] = features["saldo_mora"] / saldo_total
    features["saldo_principal_ratio"] = features["saldo_principal"] / saldo_total
    features["has_mora"] = (features["saldo_mora"] > 0).astype(int)
    features["has_codeudor_mora"] = (features["saldo_mora_codeudor"] > 0).astype(int)
    features["creditos_total"] = (
        features["creditos_sectorFinanciero"]
        + features["creditos_sectorCooperativo"]
        + features["creditos_sectorReal"]
    )
    features["datacredito_income_gap"] = (
        features["promedio_ingresos_datacredito"] - features["salario_cliente"]
    )
    return features


def get_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Separa las columnas del modelo en numericas y categoricas."""
    candidate_columns = [
        column for column in df.columns if column not in MODEL_EXCLUDED_COLUMNS
    ]
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
    """Construye el preprocesamiento que se aplica antes de entrenar o predecir."""
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
    df = add_features(clean_data(load_data(path)))
    numerical_columns, categorical_columns = get_feature_columns(df)
    preprocessor = build_preprocessor(numerical_columns, categorical_columns)
    x = df[numerical_columns + categorical_columns]
    y = df[TARGET]
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
    df = add_features(clean_data(load_data(path)))
    cutoff = df[DATE_COLUMN].median()
    reference = df[df[DATE_COLUMN] <= cutoff].copy()
    output_path = ARTIFACTS_DIR / "reference_sample.csv"
    reference.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    dataframe = add_features(clean_data(load_data()))
    save_reference_sample()
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns):,}")
    print(f"On-time payment rate: {dataframe[TARGET].mean():.3f}")
