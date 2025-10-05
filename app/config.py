"""Application configuration and environment variable management."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# AWS / S3 model fetching removed – artifacts now sourced from local paths or Hugging Face Hub only.
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")  # retained for potential future tagging (not used for HF snapshot)
DECISION_THRESHOLD_ENV = os.getenv("DECISION_THRESHOLD")  # optional override
ALLOW_START_WITHOUT_MODEL = os.getenv("ALLOW_START_WITHOUT_MODEL", "false").lower() == "true"
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")  # e.g. j2damax/hotel-cancel-model

# Local fallback paths (used if artifacts baked into image or mounted)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/champion_model.pkl")
LOCAL_PREPROCESSOR_PATH = os.getenv("LOCAL_PREPROCESSOR_PATH", "models/preprocessor.pkl")

APP_VERSION = "1.0.0"
