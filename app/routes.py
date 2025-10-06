"""FastAPI routes (health + predict) using simplified pipeline."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from .schemas import BookingFeatures, PredictionResponse, HealthResponse
import json, os
from . import config
from . import model_loader

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health():
    thr, _src = model_loader.resolve_threshold()
    loaded = model_loader.model is not None
    return HealthResponse(
        status='healthy' if loaded else 'model_not_loaded',
        model_loaded=loaded,
        model_version=model_loader.model_version,
        decision_threshold=thr if loaded else None
    )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal inference-time feature alignment.

    Injects placeholder raw & engineered columns so the persisted preprocessor
    (trained with target encoding on several categorical columns) can operate.

    We intentionally provide conservative defaults for fields not exposed via
    the public API schema. These defaults should be business-plausible and
    neutral (e.g., zeros, most-common style fallbacks) while allowing the
    preprocessor to apply target encodings and scaling without missing-column
    errors.
    """
    df = df.copy()

    # 1. Rename incoming simplified fields to training schema equivalents
    if 'arrival_month' in df.columns:
        df['arrival_date_month'] = df['arrival_month']
    if 'stays_weekend_nights' in df.columns:
        df['stays_in_weekend_nights'] = df['stays_weekend_nights']
    if 'stays_week_nights' in df.columns:
        df['stays_in_week_nights'] = df['stays_week_nights']
    if 'total_of_special_requests' in df.columns:
        df['total_of_special_requests'] = df['total_of_special_requests']  # idempotent clarity

    # 2. Add placeholder raw columns expected by feature contract / preprocessor
    placeholder_defaults = {
        'hotel': 0,
        'arrival_date_year': 2025,
        'arrival_date_week_number': 1,
        'arrival_date_day_of_month': 1,
        'babies': 0,
        'meal': 0,
        'country': 'UNK',
        'market_segment': 0,
        'distribution_channel': 0,
        'previous_bookings_not_canceled': 0,
        'reserved_room_type': 0,
        'assigned_room_type': 0,
        'deposit_type': 0,
        'days_in_waiting_list': 0,
        'customer_type': 0,
    }
    for col, default in placeholder_defaults.items():
        if col not in df.columns:
            df[col] = default

    # 3. Engineered features reproduced (subset)
    if {'stays_in_weekend_nights','stays_in_week_nights'}.issubset(df.columns):
        df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    if {'adults','children','babies'}.issubset(df.columns):
        df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    else:
        df['total_guests'] = df.get('adults', 1)
    # is_family heuristic (children or babies) match training logic closely
    if {'children','babies'}.issubset(df.columns):
        df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    else:
        df['is_family'] = 0
    # guest_type (mirrors feature_engineering logic simplified)
    def _guest_type(row):
        if row.get('babies',0) > 0:
            return 'family_with_babies'
        if row.get('children',0) > 0:
            return 'family_with_children'
        if row.get('adults',0) == 1:
            return 'solo_traveler'
        if row.get('adults',0) == 2:
            return 'couple'
        return 'group'
    if 'guest_type' not in df.columns:
        df['guest_type'] = df.apply(_guest_type, axis=1)

    # 4. Seasonal & temporal flags
    if 'arrival_date_month' in df.columns:
        m = df['arrival_date_month']
        # Normalize numeric months (1-12). If user supplied 0-11 adjust (+1).
        if set(m.unique()).issubset(set(range(0,12))):
            m_norm = m + 1
        else:
            m_norm = m
        season_map = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
        df['arrival_season'] = m_norm.map(season_map)
        df['is_peak_season'] = m_norm.isin([5,6,7,8,9]).astype(int)
        # Quarter flag and additional temporal flags
        def _quarter(x):
            if pd.isna(x):
                return None
            return f"Q{int((int(x)-1)//3)+1}"
        df['arrival_quarter'] = m_norm.apply(_quarter)
        df['is_summer_peak'] = m_norm.isin([7,8]).astype(int)
        df['is_holiday_season'] = m_norm.isin([12,1]).astype(int)
    else:
        for col, default in {
            'arrival_season': 'winter',
            'is_peak_season': 0,
            'arrival_quarter': 'Q1',
            'is_summer_peak': 0,
            'is_holiday_season': 0,
        }.items():
            if col not in df.columns:
                df[col] = default

    # 5. Ensure columns required for target encoding exist (placeholders already above)
    for te_col in ['country','guest_type','arrival_season','arrival_quarter']:
        if te_col not in df.columns:
            df[te_col] = 'UNK'

    return df


@router.post('/predict', response_model=PredictionResponse)
async def predict(booking: BookingFeatures):
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if model_loader.preprocessor is None:
        raise HTTPException(status_code=503, detail='Preprocessor not loaded')
    raw_df = pd.DataFrame([booking.model_dump()])
    prep_df = _prepare(raw_df)
    try:
        processed = model_loader.preprocessor.transform(prep_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Preprocessor transform failed: {e}')
    if hasattr(model_loader.model, 'predict_proba'):
        prob = float(model_loader.model.predict_proba(processed)[0,1])
    else:
        prob = float(model_loader.model.predict(processed)[0])
    thr, src = model_loader.resolve_threshold()
    pred = int(prob >= thr)
    return PredictionResponse(prediction=pred, probability=prob, model_version=model_loader.model_version, applied_threshold=thr, threshold_source=src)


@router.post('/predict/batch', response_model=list[PredictionResponse])
async def predict_batch(bookings: list[BookingFeatures]):
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if model_loader.preprocessor is None:
        raise HTTPException(status_code=503, detail='Preprocessor not loaded')
    raw_df = pd.DataFrame([b.model_dump() for b in bookings])
    prep_df = _prepare(raw_df)
    try:
        processed = model_loader.preprocessor.transform(prep_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Preprocessor transform failed: {e}')
    if hasattr(model_loader.model, 'predict_proba'):
        probs = model_loader.model.predict_proba(processed)[:,1]
    else:
        probs = model_loader.model.predict(processed).astype(float)
    thr, src = model_loader.resolve_threshold()
    preds = (probs >= thr).astype(int)
    return [PredictionResponse(prediction=int(p), probability=float(pr), model_version=model_loader.model_version, applied_threshold=thr, threshold_source=src) for p, pr in zip(preds, probs)]


def startup_load():
    model_loader.load_model()
    if model_loader.model is None and not config.ALLOW_START_WITHOUT_MODEL:
        raise RuntimeError('Model not loaded at startup. Provide S3 env or mount local artifacts.')


@router.get('/model/interpretability')
async def interpretability(top_k: int = 10):
    """Lightweight interpretability stub reading precomputed artifacts if available.

    Returns minimal structure expected by existing tests; if artifacts missing,
    degrades gracefully with empty lists.
    """
    artifacts_dir = config.ARTIFACT_DIR
    feature_importance_path = os.path.join(artifacts_dir, 'feature_importance.json')
    champion_meta_path = os.path.join(artifacts_dir, 'champion_meta.json')
    shap_values_sample_path = os.path.join(artifacts_dir, 'shap_values_sample.json')
    fi = []
    champion_model = None
    decision_threshold = None
    shap_generated = False
    local_examples = []
    if os.path.exists(feature_importance_path):
        try:
            with open(feature_importance_path) as f:
                raw = json.load(f)
            fi = raw[:top_k]
        except Exception:
            fi = []
    if os.path.exists(champion_meta_path):
        try:
            with open(champion_meta_path) as f:
                meta = json.load(f)
            champion_model = meta.get('model_name')
            decision_threshold = meta.get('decision_threshold')
            shap_generated = bool(meta.get('shap_generated'))
        except Exception:
            pass
    if os.path.exists(shap_values_sample_path):
        try:
            with open(shap_values_sample_path) as f:
                raw_local = json.load(f)[:3]
            # Adapt shape: ensure keys top_positive_contributors / top_negative_contributors
            adapted = []
            for rec in raw_local:
                shap_vals = rec.get('shap_values', {})
                positives = sorted([(k,v) for k,v in shap_vals.items() if v > 0], key=lambda x: x[1], reverse=True)[:5]
                negatives = sorted([(k,v) for k,v in shap_vals.items() if v < 0], key=lambda x: x[1])[:5]
                adapted.append({
                    'category': rec.get('category','sample'),
                    'probability': rec.get('probability'),
                    'top_positive_contributors': [{'feature': f, 'shap': v} for f,v in positives],
                    'top_negative_contributors': [{'feature': f, 'shap': v} for f,v in negatives]
                })
            local_examples = adapted
        except Exception:
            local_examples = []
    return {
        'champion_model': champion_model,
        'shap_generated': shap_generated and bool(fi),
        'shap_timestamp': None,
        'decision_threshold': decision_threshold,
        'top_features': fi,
        'local_examples': local_examples,
        'feature_name_map': {},
        'artifacts_available': []
    }
