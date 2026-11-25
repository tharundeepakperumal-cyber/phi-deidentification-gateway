# PHI De-identification Gateway

A machine-learning-based gateway for detecting and de-identifying Protected Health Information (PHI) in clinical text.

## Features

- DistilBERT-based PHI detection
- Template-aware model training
- Person name detection
- Phone number detection
- Email detection
- Medical record number detection
- PHI replacement with anonymized placeholders
- Controlled rehydration
- API key authentication
- Secure logging
- FastAPI REST API

## Technology Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers
- DistilBERT
- FastAPI
- Presidio
- Pytest

## Installation

Create a virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

Configure environment:

    cp .env.example .env

## Dataset Preparation

    python -m src.prepare_dataset

## Model Training

    python -m src.train_template_split

## Model Evaluation

    python -m src.evaluation

## Model Inference

    python -m src.test_model

## Automated Testing

    python -m pytest -v

## Start API

    uvicorn src.api:app --host 127.0.0.1 --port 8000

Health endpoint:

    http://127.0.0.1:8000/health

## Evaluation

The current synthetic dataset contains 1,000 clinical notes.

Template-aware model validation F1: 0.999.

The automated test suite has 32 passing tests and 2 passing subtests.

These results are from synthetic development data and do not establish production clinical accuracy.

## Security

- Keep API keys in environment variables.
- Never commit .env files.
- Protect re-identification mappings.
- Do not commit real patient records or PHI.
- Validate security before production deployment.

## Disclaimer

This is a development prototype and is not independently validated or certified for production clinical use or regulatory compliance.

## Docker Deployment

Build the Docker image:

    docker build -t phi-deidentification-gateway:latest .

Run the application:

    docker run -d --name phi-gateway \
      -p 8000:8000 \
      -e PHI_API_KEY="your-secret-key" \
      phi-deidentification-gateway:latest

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Application status |
| GET | /health | Health check |
| GET | /model-status | Model verification |
| POST | /deidentify | PHI detection and masking |

The de-identification endpoint requires the X-API-Key header.

## Verified Tests

- 32 tests passed and 2 subtests passed.
- Docker image built successfully.
- Docker health endpoint returned HTTP 200.
- Model files verified inside Docker.
- PERSON, MRN, PHONE and EMAIL detection verified.
- Missing API key correctly rejected with HTTP 401.

## Performance

FP32 CPU: 89.45 records/second.
FP32 MPS: 194.74 records/second.
INT8 CPU: 19.50 records/second.

FP32 remains the selected model configuration.

Performance and accuracy evaluations use synthetic clinical data. Independent clinical validation is still required before production use.

## Disclaimer

This is a development prototype, not a certified production clinical system.
