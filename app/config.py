"""Application configuration and environment variable management."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_S3_URI = os.getenv("MODEL_S3_URI")  # e.g. s3://bucket/models
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
AWS_REGION = os.getenv("AWS_REGION")
DECISION_THRESHOLD_ENV = os.getenv("DECISION_THRESHOLD")  # optional override
ALLOW_START_WITHOUT_MODEL = os.getenv("ALLOW_START_WITHOUT_MODEL", "false").lower() == "true"
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

# Local fallback paths (used if S3 not configured or fetch fails)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/champion_model.pkl")
LOCAL_PREPROCESSOR_PATH = os.getenv("LOCAL_PREPROCESSOR_PATH", "models/preprocessor.pkl")

APP_VERSION = "1.0.0"
