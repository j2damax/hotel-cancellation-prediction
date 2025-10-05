"""Model + preprocessor loading utilities (S3 + local fallback)."""
from __future__ import annotations
import os, json, time
import joblib
from typing import Optional, Tuple

from . import config
from src.preprocessing import PreprocessingPipeline

model = None
preprocessor: Optional[PreprocessingPipeline] = None
model_version: Optional[str] = None
champion_meta_threshold: Optional[float] = None
_last_reload_time: float | None = None


def _resolve_git_sha() -> str | None:
    git_sha = os.getenv('GIT_SHA')
    if git_sha:
        return git_sha[:12]
    head_path = os.path.join('.git','HEAD')
    try:
        if os.path.exists(head_path):
            with open(head_path) as hf:
                ref = hf.read().strip()
            if ref.startswith('ref:'):
                ref_file = ref.split(' ',1)[1]
                ref_path = os.path.join('.git', ref_file)
                if os.path.exists(ref_path):
                    with open(ref_path) as rf:
                        return rf.read().strip()[:12]
            else:
                return ref[:12]
    except Exception:
        return None
    return None


def load_model() -> None:
    """Idempotent loading routine. Prefers S3 if configured."""
    global model, preprocessor, model_version, champion_meta_threshold, _last_reload_time
    # Attempt S3 fetch if configured
    if config.MODEL_S3_URI:
        try:
            import boto3
            if not config.MODEL_S3_URI.startswith('s3://'):
                raise ValueError('MODEL_S3_URI must start with s3://')
            remainder = config.MODEL_S3_URI[len('s3://'):]
            bucket, *rest = remainder.split('/', 1)
            base_prefix = rest[0].rstrip('/') if rest else ''
            s3 = boto3.client('s3', region_name=config.AWS_REGION) if config.AWS_REGION else boto3.client('s3')
            resolved_version = config.MODEL_VERSION
            if config.MODEL_VERSION == 'latest':
                latest_key = f"{base_prefix}/latest.txt" if base_prefix else 'latest.txt'
                latest_obj = s3.get_object(Bucket=bucket, Key=latest_key)
                resolved_version = latest_obj['Body'].read().decode().strip()
            version_prefix = f"{base_prefix}/{resolved_version}" if base_prefix else resolved_version
            cache_dir = os.path.join('models','remote', resolved_version)
            os.makedirs(cache_dir, exist_ok=True)
            def _dl(name):
                key = f"{version_prefix}/{name}"
                local_path = os.path.join(cache_dir, name)
                if not os.path.exists(local_path):
                    s3.download_file(bucket, key, local_path)
                return local_path
            model_path = _dl('champion_model.pkl')
            preproc_path = _dl('preprocessor.pkl')
            meta_path = _dl('champion_meta.json')
            # Load artifacts
            model_candidate = joblib.load(model_path)
            if hasattr(model_candidate, 'predict'):
                model = model_candidate
            try:
                preprocessor = PreprocessingPipeline.load(preproc_path)
            except Exception:
                preprocessor = None
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if 'decision_threshold' in meta:
                    champion_meta_threshold = meta['decision_threshold']
            except Exception:
                pass
            model_version = resolved_version
            _last_reload_time = time.time()
            git_sha = _resolve_git_sha()
            print(f"Loaded model (S3) version={model_version} git_sha={git_sha}")
        except Exception as e:
            print(f"S3 load failed: {e}. Falling back to local artifacts.")
    # Local fallback
    if model is None and os.path.exists(config.LOCAL_MODEL_PATH):
        try:
            model_candidate = joblib.load(config.LOCAL_MODEL_PATH)
            if hasattr(model_candidate, 'predict'):
                model = model_candidate
                # pseudo version from mtime
                mtime = int(os.path.getmtime(config.LOCAL_MODEL_PATH))
                model_version = f"local_{mtime}"
        except Exception as e:
            print(f"Local model load failed: {e}")
    if preprocessor is None and os.path.exists(config.LOCAL_PREPROCESSOR_PATH):
        try:
            preprocessor = PreprocessingPipeline.load(config.LOCAL_PREPROCESSOR_PATH)
        except Exception:
            preprocessor = None
    if model is None:
        print("No model loaded (S3 + local fallback failed). API will report model_not_loaded.")


def resolve_threshold() -> tuple[float, str]:
    if config.DECISION_THRESHOLD_ENV is not None:
        try:
            return float(config.DECISION_THRESHOLD_ENV), 'env'
        except ValueError:
            pass
    if champion_meta_threshold is not None:
        try:
            return float(champion_meta_threshold), 'champion_meta'
        except ValueError:
            pass
    return 0.5, 'default'
