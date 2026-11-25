FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.15.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/template_aware/ ./models/template_aware/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
