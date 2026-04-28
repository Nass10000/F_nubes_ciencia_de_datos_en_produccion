FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/model_training_evaluation.py

EXPOSE 5000

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "5000"]
