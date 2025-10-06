"""Model + preprocessor loading utilities (local + Hugging Face Hub).

S3 support removed as project no longer uses AWS. Loading order:
1. Local baked artifacts (if present)
2. Hugging Face Hub (HF_MODEL_REPO)

Environment variables quick reference:
    HF_MODEL_REPO         remote repo to pull snapshot from (e.g. j2damax/hotel-cancel-model)
    FORCE_HF_LOAD=true    always fetch from HF even if local artifacts exist
    HF_HUB_CACHE=path     optional override for huggingface hub cache directory
    ALLOW_START_WITHOUT_MODEL=true  allow API to start even when model load fails
    DECISION_THRESHOLD    manual override for probability decision threshold
"""
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
    """Idempotent loading routine.

    Order (unless overridden):
      1. Local baked artifacts (if present)
      2. Hugging Face Hub snapshot (HF_MODEL_REPO)

    Env flags:
      FORCE_HF_LOAD=true   -> Skip local artifacts and always pull from HF
      HF_HUB_CACHE=/custom -> Redirect huggingface hub cache (optional; we still store snapshot in models/hf/...)
    """
    global model, preprocessor, model_version, champion_meta_threshold, _last_reload_time

    force_hf = os.getenv('FORCE_HF_LOAD', 'false').lower() == 'true'
    hf_repo = getattr(config, 'HF_MODEL_REPO', None)

    # Optional hub-wide cache redirection for environments with restricted root perms
    hf_hub_cache = os.getenv('HF_HUB_CACHE')
    if hf_hub_cache:
        try:
            os.makedirs(hf_hub_cache, exist_ok=True)
            os.environ['HF_HOME'] = hf_hub_cache  # respected by huggingface_hub
        except Exception as e:
            print(f"Could not create HF_HUB_CACHE directory {hf_hub_cache}: {e}")

    # Local load (unless forcing HF)
    if not force_hf:
        if model is None and os.path.exists(config.LOCAL_MODEL_PATH):
            try:
                model_candidate = joblib.load(config.LOCAL_MODEL_PATH)
                if hasattr(model_candidate, 'predict'):
                    model = model_candidate
                    mtime = int(os.path.getmtime(config.LOCAL_MODEL_PATH))
                    model_version = f"local_{mtime}"
            except Exception as e:
                print(f"Local model load failed: {e}")
        if preprocessor is None and os.path.exists(config.LOCAL_PREPROCESSOR_PATH):
            try:
                preprocessor = PreprocessingPipeline.load(config.LOCAL_PREPROCESSOR_PATH)
            except Exception:
                preprocessor = None
    else:
        print("FORCE_HF_LOAD=true -> Skipping local artifact loading")

    # Hugging Face Hub load if still missing or forced
    need_hf = (model is None or preprocessor is None or force_hf) and hf_repo
    if need_hf:
        try:
            from huggingface_hub import snapshot_download
            repo_id = hf_repo
            cache_dir = os.path.join('models','hf', repo_id.replace('/','__'))
            os.makedirs(cache_dir, exist_ok=True)
            local_dir = snapshot_download(repo_id=repo_id, local_dir=cache_dir, local_dir_use_symlinks=False)
            model_path = os.path.join(local_dir, 'champion_model.pkl')
            preproc_path = os.path.join(local_dir, 'preprocessor.pkl')
            meta_path = os.path.join(local_dir, 'champion_meta.json')
            if (model is None or force_hf) and os.path.exists(model_path):
                try:
                    m_candidate = joblib.load(model_path)
                    if hasattr(m_candidate, 'predict'):
                        model = m_candidate
                        model_version = f"hf_{os.path.getmtime(model_path):.0f}"
                except Exception as e:
                    print(f"HF model load failed: {e}")
            if (preprocessor is None or force_hf) and os.path.exists(preproc_path):
                try:
                    preprocessor = PreprocessingPipeline.load(preproc_path)
                except Exception:
                    preprocessor = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    if 'decision_threshold' in meta:
                        champion_meta_threshold = meta['decision_threshold']
                except Exception:
                    pass
            if model is not None and force_hf:
                print(f"Loaded model (HF FORCE) repo={repo_id} version={model_version}")
            elif model is not None:
                print(f"Loaded model (HF) repo={repo_id} version={model_version}")
        except Exception as e:
            print(f"HF load failed: {e}")

    if model is None:
        print("No model loaded (checked local + HF). API will report model_not_loaded.")


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


def load_model_and_preprocessor():
    """Convenience helper to ensure artifacts are loaded and return them with minimal metadata.

    Returns (model, preprocessor, metadata_dict)
    metadata_dict contains keys: version, threshold, threshold_source
    """
    if model is None or preprocessor is None:
        load_model()
    thr, source = resolve_threshold()
    meta = {
        'version': model_version,
        'threshold': thr,
        'threshold_source': source
    }
    return model, preprocessor, meta
