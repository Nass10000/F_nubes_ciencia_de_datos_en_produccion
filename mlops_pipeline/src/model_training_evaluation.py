"""Entrena, evalua y guarda los modelos supervisados del proyecto."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from ft_engineering import (
        ARTIFACTS_DIR,
        DATA_PATH,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        get_feature_columns,
        load_data,
        make_training_dataset,
        save_reference_sample,
    )
except ImportError:
    from .ft_engineering import (
        ARTIFACTS_DIR,
        DATA_PATH,
        REPORTS_DIR,
        TARGET,
        add_features,
        clean_data,
        get_feature_columns,
        load_data,
        make_training_dataset,
        save_reference_sample,
    )


MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
METRICS_PATH = REPORTS_DIR / "metrics_summary.csv"


def build_model(model_name: str, preprocessor) -> Pipeline:
    """Arma un pipeline completo con preprocesamiento mas algoritmo."""
    # Se prueban tres modelos candidatos para luego elegir el mejor.
    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimators[model_name]),
        ]
    )


def summarize_classification(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    y_proba,
) -> dict[str, float | str]:
    """Calcula las metricas principales para comparar modelos."""
    # Todas las opciones se evalúan con el mismo conjunto de metricas.
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_evaluate(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Entrena los modelos, elige el mejor por balanced accuracy y guarda artefactos."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, preprocessor = make_training_dataset(data_path)
    model_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    rows = []
    trained_models = {}

    # Todos se entrenan sobre el mismo split para compararlos de forma justa.
    for model_name in model_names:
        model = build_model(model_name, preprocessor)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        rows.append(summarize_classification(model_name, y_test, y_pred, y_proba))
        trained_models[model_name] = model

    metrics = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "roc_auc"], ascending=False
    )
    metrics.to_csv(METRICS_PATH, index=False)

    # Balanced accuracy evita elegir un modelo que ignore la clase minoritaria.
    best_model_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)

    # Estos reportes sirven como evidencia tecnica del PI.
    y_pred = best_model.predict(x_test)
    report = classification_report(y_test, y_pred)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, y_pred)).to_csv(
        REPORTS_DIR / "confusion_matrix.csv", index=False
    )

    full_df = add_features(clean_data(load_data(data_path)))
    numerical_columns, categorical_columns = get_feature_columns(full_df)
    # El schema guarda como se entreno el modelo para reutilizarlo al predecir.
    schema = {
        "target": TARGET,
        "best_model": str(best_model_name),
        "selection_metric": "balanced_accuracy",
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "input_columns": numerical_columns + categorical_columns,
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    save_reference_sample(data_path)
    return metrics


def load_trained_model(path: str | Path = MODEL_PATH):
    """Carga el modelo guardado y lo entrena si todavia no existe."""
    path = Path(path)
    # Esto evita que la app falle si los artefactos aun no han sido generados.
    if not path.exists():
        train_and_evaluate()
    return joblib.load(path)


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(summary.to_string(index=False))
    print(f"Best model saved at: {MODEL_PATH}")
