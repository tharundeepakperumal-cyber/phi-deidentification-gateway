
import os
import secrets
import logging
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from src.model_inference import deidentify_with_model
from src.secure_logger import setup_secure_logging

load_dotenv()
API_KEY = os.getenv("PHI_API_KEY")

logger = setup_secure_logging()
logger.propagate = False

app = FastAPI(
    title="PHI De-identification Gateway",
    description="API for detecting and removing PHI from clinical notes.",
    version="1.0.0",
)


class ClinicalNoteRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Clinical note text to de-identify",
    )


class DeidentificationResponse(BaseModel):
    deidentified_text: str
    deidentifier_text: str
    detected_entities: dict


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return True


@app.get("/")
def home():
    return {"message": "PHI De-identification Gateway is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}



@app.post(
    "/deidentify",
    response_model=DeidentificationResponse,
    dependencies=[Depends(verify_api_key)],
)
def deidentify_note(request: ClinicalNoteRequest):
    try:
        masked_text, mapping = deidentify_with_model(request.text)

        counts = Counter()

        for placeholder in mapping:
            entity_type = placeholder.split("_")[0].strip("[")

            if entity_type == "NAME":
                entity_type = "PERSON"

            counts[entity_type] += 1

        logger.info(
            "De-identification completed. Entity counts: %s",
            dict(counts),
        )

        return {
            "deidentified_text": masked_text,
            "deidentifier_text": masked_text,
            "detected_entities": dict(counts),
        }

    except Exception:
        logger.error("De-identification processing failed")
        raise HTTPException(
            status_code=500,
            detail="Internal de-identification processing error",
        ) from None



logger.info("PHI De-identification Gateway initialized")
