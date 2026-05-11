# CustomerChurnX - PI M5

Proyecto integrador de Fundamentos de nube y ciencia de datos en produccion. El objetivo es predecir el churn de clientes usando datos historicos y dejar un flujo reproducible con carga, EDA, feature engineering, entrenamiento, despliegue y monitoreo.

## Estructura

```text
mlops_pipeline/
  src/
    Cargar_datos.ipynb
    comprension_eda.ipynb
    ft_engineering.py
    model_training_evaluation.py
    model_deploy.py
    model_monitoring.py
  Base_de_datos.csv
  requirements.txt
  .gitignore
  readme.md
```

Tambien se incluyen `Dockerfile`, `.dockerignore`, `tests/`, `artifacts/` y `reports/` para cubrir despliegue, validacion y evidencia tecnica.

## Caso de Negocio

CustomerChurnX necesita anticipar que clientes tienen mayor probabilidad de abandonar el servicio. La variable objetivo es `churn`, y las senales disponibles incluyen antiguedad, plan, canal, region, uso semanal, tickets de soporte, descuentos, mora y renovacion automatica.

## Instalacion

```bash
cd mlops_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion

```bash
python src/ft_engineering.py
python src/model_training_evaluation.py
python src/model_deploy.py --input Base_de_datos.csv --output reports/batch_predictions.csv
python src/model_monitoring.py
```

## Resultados de Modelado

Se compararon Logistic Regression, Random Forest y Gradient Boosting. El mejor modelo por ROC-AUC fue `logistic_regression`.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.639 | 0.373 | 0.701 | 0.486 | 0.726 |
| random_forest | 0.695 | 0.412 | 0.582 | 0.482 | 0.709 |
| gradient_boosting | 0.753 | 0.483 | 0.172 | 0.254 | 0.701 |

El modelo se guarda en `artifacts/best_model.joblib` y las metricas en `reports/metrics_summary.csv`.

## API y Docker

La API expone:

- `GET /health`
- `POST /predict`
- `POST /prediccion`

Ejemplo Docker:

```bash
docker build -t customer-churnx .
docker run --rm -p 5000:5000 customer-churnx
```

Con `-p 5000:5000`, el puerto del host coincide con el puerto interno de Uvicorn. Con `-p 8000:5000`, la API queda disponible en el host por el puerto 8000. Con `-p 5000:8000`, no funcionara si la app sigue escuchando internamente en 5000.

## Streamlit

La app web permite cargar datos de un cliente, predecir su probabilidad de churn, cargar un CSV para prediccion batch y revisar monitoreo de drift.

```bash
streamlit run streamlit_app.py
```

## Monitoreo

`model_monitoring.py` compara una ventana historica contra un lote actual simulado. Calcula PSI, KS, chi-cuadrado y Jensen-Shannon. El reporte queda en:

- `reports/drift_report.csv`
- `reports/drift_report.json`

Variables con drift detectado en la corrida validada: `low_engagement_flag`, `sessions_week`, `avg_session_min`, `support_tickets_3m` y `engagement_minutes_week`.

Para abrir solo el dashboard de monitoreo desde el modulo:

```bash
streamlit run src/model_monitoring.py
```

## Gitflow

El repositorio debe tener tres ramas:

- `developer`: desarrollo y cambios principales.
- `certification`: version validada antes de produccion.
- `master`: entrega estable final.

La estructura base representa `V1.0.0`; los notebooks `V1.0.1`; feature engineering y modelado `V1.1.0`; monitoreo y despliegue cubren los avances posteriores.
