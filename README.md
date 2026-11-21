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

The automated test suite contains 17 tests.

These results are from synthetic development data and do not establish production clinical accuracy.

## Security

- Keep API keys in environment variables.
- Never commit .env files.
- Protect re-identification mappings.
- Do not commit real patient records or PHI.
- Validate security before production deployment.

## Disclaimer

This is a development prototype and is not independently validated or certified for production clinical use or regulatory compliance.
