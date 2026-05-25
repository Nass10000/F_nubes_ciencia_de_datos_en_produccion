# PI M5 - Riesgo crediticio

Este repositorio contiene el Proyecto Integrador del modulo de Fundamentos de nube y ciencia de datos en produccion. El caso es financiero: a partir de informacion historica de creditos se entrena un modelo para estimar si una solicitud tendra `Pago_atiempo`.

La base usada por el proyecto es `Base_de_datos.csv`, con 10,763 registros y 23 columnas originales.

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
  Dockerfile
  tests/
  artifacts/
  reports/
  readme.md
```

## Avance 1 - Versionamiento y EDA

Se completo la estructura solicitada, el archivo `requirements.txt` y los notebooks:

- `src/Cargar_datos.ipynb`: carga la base, valida columnas obligatorias, revisa nulos, duplicados, tipos y distribucion de `Pago_atiempo`.
- `src/comprension_eda.ipynb`: desarrolla EDA inicial, univariable, bivariable y multivariable. Tambien documenta variables categoricas, numericas, ordinales, nominales, dicotomicas y politomicas, reglas de validacion y features sugeridas.

La limpieza unifica nulos, convierte `fecha_prestamo` desde serial de Excel a fecha real, corrige tipos numericos y crea `loan_id` como identificador tecnico.

## Avance 2 - Feature Engineering y Modelado

`src/ft_engineering.py` crea las variables derivadas y el preprocesamiento:

- `cuota_salario_ratio`
- `capital_salario_ratio`
- `otros_prestamos_salario_ratio`
- `carga_total_salario_ratio`
- `capital_por_mes`
- `creditos_total`
- `datacredito_income_gap`
- `prestamo_year` y `prestamo_month`

Para evitar fuga de informacion, el modelo excluye variables que pueden revelar el resultado despues del credito, como `puntaje`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor` y sus derivados de mora.

`src/model_training_evaluation.py` compara Logistic Regression, Random Forest y Gradient Boosting. Debido al desbalance de la clase objetivo, el mejor modelo se selecciona por `balanced_accuracy`.

| Modelo | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6470 | 0.6051 | 0.9674 | 0.6514 | 0.7786 | 0.6628 |
| random_forest | 0.9090 | 0.5609 | 0.9585 | 0.9454 | 0.9519 | 0.6733 |
| gradient_boosting | 0.9494 | 0.5076 | 0.9533 | 0.9956 | 0.9740 | 0.6831 |

Artefactos generados:

- `artifacts/best_model.joblib`
- `artifacts/feature_schema.json`
- `artifacts/reference_sample.csv`
- `reports/metrics_summary.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.csv`

## Avance 3 - Monitoreo y Streamlit

`src/model_monitoring.py` compara una ventana historica contra una ventana reciente y calcula:

- PSI para variables numericas.
- Kolmogorov-Smirnov para variables numericas.
- Jensen-Shannon y chi-cuadrado para variables categoricas.

Los reportes quedan en:

- `reports/drift_report.csv`
- `reports/drift_report.json`
- `reports/current_predictions_sample.csv`

La aplicacion `streamlit_app.py` tiene dos pestanas:

- `Prediccion`: formulario individual y carga CSV por lote.
- `Monitoreo`: tabla y grafico de data drift en modo real o simulado.

Tambien se incluye `.github/workflows/monitoring.yml` para ejecutar el reporte de drift de forma programada o manual desde GitHub Actions.

## Avance 4 - API, Docker y Prediccion Batch

`src/model_deploy.py` expone el modelo con FastAPI:

- `GET /health`
- `POST /predict`
- `POST /prediccion`

Tambien permite prediccion por lote desde terminal:

```bash
python src/model_deploy.py --input Base_de_datos.csv --output reports/batch_predictions.csv
```

El `Dockerfile` empaqueta dependencias, codigo, modelo y API con Uvicorn.

```bash
docker build -t pi-riesgo-crediticio .
docker run --rm -p 5000:5000 pi-riesgo-crediticio
```

Con la API activa, la documentacion Swagger queda disponible en:

```text
http://127.0.0.1:5000/docs
```

## Instalacion local

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
streamlit run streamlit_app.py
```

## Pruebas

```bash
pytest -q tests
```

Los tests validan:

- esquema y objetivo `Pago_atiempo`;
- cantidad correcta de registros;
- columnas creadas por feature engineering;
- contrato de salida de prediccion;
- modos de monitoreo real y simulado.

## Gitflow

El flujo esperado del PI usa tres ramas:

- `developer`: desarrollo activo.
- `certification`: validacion antes de entrega estable.
- `master`: version final estable.

Los archivos `lectures_txt/` y `materials/` estan ignorados para que no se suban al repositorio.
