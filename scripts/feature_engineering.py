"""
Output: A clean feature-enriched DataFrame ready for model training.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Privacy / PII handling
# -----------------------------
PII_COLUMNS = ['name', 'email', 'phone-number', 'credit_card']


def drop_pii(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove personally identifiable columns if present.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe after initial load.
    verbose : bool
        If True log removed columns.
    """
    existing = [c for c in PII_COLUMNS if c in df.columns]
    if existing:
        if verbose:
            logger.info(f"Dropping PII columns: {existing}")
        df = df.drop(columns=existing)
    return df


# -----------------------------
# Core Feature Implementations
# -----------------------------

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Priority 1 core business features.

    Assumptions: Input df already preprocessed (no missing critical columns).
    """
    df = df.copy()

    # 1. total_stay_duration
    if {'stays_in_weekend_nights', 'stays_in_week_nights'}.issubset(df.columns):
        df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # 2. is_family
    if {'children', 'babies'}.issubset(df.columns):
        df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    else:
        df['is_family'] = 0

    # 3. guest_type
    if {'adults', 'children', 'babies'}.issubset(df.columns):
        def _guest_type(row):
            if row['babies'] > 0:
                return 'family_with_babies'
            if row['children'] > 0:
                return 'family_with_children'
            if row['adults'] == 1:
                return 'solo_traveler'
            if row['adults'] == 2:
                return 'couple'
            return 'group'
        df['guest_type'] = df.apply(_guest_type, axis=1)
    else:
        df['guest_type'] = 'unknown'

    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add arrival_season and is_peak_season using month column.

    Supports both numeric (1-12) or string month names.
    """
    df = df.copy()
    col = 'arrival_date_month'
    if col not in df.columns:
        return df

    month_vals = df[col].unique()
    # Determine representation
    month_name_map = {
        'January': 'winter', 'February': 'winter', 'March': 'spring', 'April': 'spring', 'May': 'spring',
        'June': 'summer', 'July': 'summer', 'August': 'summer', 'September': 'autumn', 'October': 'autumn',
        'November': 'autumn', 'December': 'winter'
    }
    if np.issubdtype(df[col].dtype, np.number):
        # Assume 1-12; if 0-based adjust
        if set(month_vals).issubset(set(range(0,12))):  # 0-based
            month_numeric = df[col] + 1
        else:
            month_numeric = df[col]
        season_map_num = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        df['arrival_season'] = month_numeric.map(season_map_num)
        peak_months = {5,6,7,8,9}  # May-Sep per plan
        df['is_peak_season'] = month_numeric.isin(peak_months).astype(int)
    else:
        df['arrival_season'] = df[col].map(month_name_map)
        peak_months = {'May','June','July','August','September'}
        df['is_peak_season'] = df[col].isin(peak_months).astype(int)

    return df


def add_temporal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Priority 2 temporal flags: arrival_quarter, is_summer_peak, is_holiday_season."""
    df = df.copy()
    col = 'arrival_date_month'
    if col not in df.columns:
        return df

    def _month_to_quarter(m):
        if isinstance(m, str):
            mapping = {'January':1,'February':1,'March':1,'April':2,'May':2,'June':2,'July':3,'August':3,'September':3,'October':4,'November':4,'December':4}
            m_num = mapping.get(m, np.nan)
        else:
            # numeric adjust if 0-based
            if m in range(0,12):
                m_num = m+1
            else:
                m_num = m
        if pd.isna(m_num):
            return np.nan
        return f"Q{int((m_num-1)//3)+1}"

    df['arrival_quarter'] = df[col].apply(_month_to_quarter)

    # summer peak / holiday flags (works for numeric or names)
    summer_set_names = {'July','August'}
    holiday_set_names = {'December','January'}

    if np.issubdtype(df[col].dtype, np.number):
        # Normalize numeric
        if set(df[col].unique()).issubset(set(range(0,12))):
            month_norm = df[col] + 1
        else:
            month_norm = df[col]
        df['is_summer_peak'] = month_norm.isin([7,8]).astype(int)
        df['is_holiday_season'] = month_norm.isin([12,1]).astype(int)
    else:
        df['is_summer_peak'] = df[col].isin(summer_set_names).astype(int)
        df['is_holiday_season'] = df[col].isin(holiday_set_names).astype(int)

    return df


# -----------------------------
# Mean Target Encoding
# -----------------------------

def mean_target_encode(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    n_folds: int = 5,
    smoothing: int = 10,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Cross-validated mean target encoding with smoothing to reduce leakage & variance.

    Returns a new DataFrame plus metadata about each encoded column.
    """
    df = df.copy()
    global_mean = df[target_col].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metadata: Dict[str, Dict] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        encoded = pd.Series(index=df.index, dtype=float)
        for train_idx, val_idx in kf.split(df):
            train, valid = df.iloc[train_idx], df.iloc[val_idx]
            stats = train.groupby(col)[target_col].agg(['mean','count'])
            # smoothing formula
            smoothing_weight = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing))
            stats['smoothed'] = global_mean * (1 - smoothing_weight) + stats['mean'] * smoothing_weight
            encoded.iloc[val_idx] = valid[col].map(stats['smoothed']).fillna(global_mean)
        new_col = f"{col}_target_encoded"
        df[new_col] = encoded
        corr = df[new_col].corr(df[target_col])
        metadata[col] = {
            'encoded_column': new_col,
            'correlation_with_target': float(corr),
            'unique_categories': int(df[col].nunique())
        }
    return df, metadata


# -----------------------------
# Orchestrator
# -----------------------------

def build_feature_dataset(
    df: pd.DataFrame,
    target_col: str = 'is_canceled',
    mte_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """End-to-end feature construction applying the simplified plan."""
    log: Dict[str, any] = {'started': datetime.utcnow().isoformat()}

    # Core
    df = add_core_features(df)
    log['core_features_added'] = True

    # Seasonal & temporal basic
    df = add_seasonal_features(df)
    df = add_temporal_flags(df)
    log['temporal_features_added'] = True

    # Mean Target Encoding
    if mte_columns is None:
        # Provide conservative defaults—only include columns present & likely categorical
        candidate_cols = [c for c in ['hotel','market_segment','distribution_channel','reserved_room_type','customer_type'] if c in df.columns]
    else:
        candidate_cols = [c for c in mte_columns if c in df.columns]

    df, mte_meta = mean_target_encode(df, candidate_cols, target_col) if candidate_cols else (df, {})
    log['mean_target_encoding'] = mte_meta

    log['final_shape'] = df.shape
    log['completed'] = datetime.utcnow().isoformat()
    return df, log


def save_features(df: pd.DataFrame, log: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    meta_path = output_path.with_name(output_path.stem + '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(log, f, indent=2)
    logger.info(f"Saved features to {output_path} and metadata to {meta_path}")


def persist_artifacts(df: pd.DataFrame, log: Dict, target_col: str, base_output: Path):
    """Persist inference-time artifacts: MTE mappings, feature contract, schema, rules, distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Feature engineered dataset (includes target column during training phase).
    log : Dict
        Log dict from build_feature_dataset containing mean_target_encoding metadata.
    target_col : str
        Name of target column.
    base_output : Path
        Directory where artifacts/* will be written.
    """
    artifacts_dir = base_output / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. MTE mappings
    mte_meta = log.get('mean_target_encoding', {}) or {}
    if mte_meta:
        encodings = {}
        global_mean = float(df[target_col].mean()) if target_col in df.columns else None
        for base_col, meta in mte_meta.items():
            enc_col = meta['encoded_column']
            if base_col in df.columns and enc_col in df.columns:
                mapping = df.groupby(base_col)[enc_col].mean().to_dict()
            else:
                mapping = {}
            encodings[base_col] = {
                'encoded_column': enc_col,
                'global_mean': global_mean,
                'categories': mapping,
                'unique_categories': meta.get('unique_categories'),
                'correlation_with_target': meta.get('correlation_with_target')
            }
        mte_payload = {
            'created_utc': datetime.utcnow().isoformat(),
            'target': target_col,
            'n_mappings': len(encodings),
            'encodings': encodings
        }
        with open(artifacts_dir / 'mte_mappings.json', 'w') as f:
            json.dump(mte_payload, f, indent=2)
        logger.info("Persisted mte_mappings.json")

    # 2. Feature contract (order & dtypes excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    contract_payload = {
        'created_utc': datetime.utcnow().isoformat(),
        'feature_order': feature_cols,
        'dtypes': {c: str(df[c].dtype) for c in feature_cols}
    }
    with open(artifacts_dir / 'feature_contract.json', 'w') as f:
        json.dump(contract_payload, f, indent=2)
    logger.info("Persisted feature_contract.json")

    # 3. Feature schema (basic constraints for known columns)
    schema_entries = {}
    constraint_hints = {
        'adults': {"min": 1},
        'children': {"min": 0},
        'babies': {"min": 0},
        'stays_in_weekend_nights': {"min": 0},
        'stays_in_week_nights': {"min": 0},
        'total_stay_duration': {"min": 0},
        'is_family': {"values": [0,1]},
        'is_peak_season': {"values": [0,1]},
        'is_summer_peak': {"values": [0,1]},
        'is_holiday_season': {"values": [0,1]}
    }
    for c in df.columns:
        if c == target_col:
            continue
        schema_entries[c] = {
            'dtype': str(df[c].dtype),
            'nullable': bool(df[c].isna().any()),
            'constraints': constraint_hints.get(c, {})
        }
    with open(artifacts_dir / 'feature_schema.json', 'w') as f:
        json.dump({'created_utc': datetime.utcnow().isoformat(), 'schema': schema_entries}, f, indent=2)
    logger.info("Persisted feature_schema.json")

    # 4. Feature rules (deterministic logic specification)
    feature_rules = {
        'guest_type_rule': [
            {'if': 'babies>0', 'return': 'family_with_babies'},
            {'elif': 'children>0', 'return': 'family_with_children'},
            {'elif': 'adults==1', 'return': 'solo_traveler'},
            {'elif': 'adults==2', 'return': 'couple'},
            {'else': True, 'return': 'group'}
        ],
        'season_mapping_numeric': {
            '1':'winter','2':'winter','3':'spring','4':'spring','5':'spring','6':'summer','7':'summer','8':'summer','9':'autumn','10':'autumn','11':'autumn','12':'winter'
        },
        'peak_months_numeric': [5,6,7,8,9],
        'temporal_flags': {
            'arrival_quarter': 'Q{((month-1)//3)+1}',
            'is_summer_peak': '[7,8]',
            'is_holiday_season': '[12,1]'
        }
    }
    with open(artifacts_dir / 'feature_rules.json', 'w') as f:
        json.dump({'created_utc': datetime.utcnow().isoformat(), 'rules': feature_rules}, f, indent=2)
    logger.info("Persisted feature_rules.json")

    # 5. Distribution baselines (value counts / mean per encoded & categorical)
    dist_payload = {'created_utc': datetime.utcnow().isoformat(), 'columns': {}}
    monitor_cols = [c for c in df.columns if c.endswith('_target_encoded') or c in ['hotel','market_segment','distribution_channel','reserved_room_type','customer_type','guest_type','arrival_season']]
    for c in monitor_cols:
        vc = df[c].value_counts(normalize=True).head(50).round(6).to_dict()
        dist_payload['columns'][c] = {'top_value_proportions': vc, 'n_unique': int(df[c].nunique())}
    if target_col in df.columns:
        dist_payload['target_mean'] = float(df[target_col].mean())
    with open(artifacts_dir / 'distribution_baselines.json', 'w') as f:
        json.dump(dist_payload, f, indent=2)
    logger.info("Persisted distribution_baselines.json")

    logger.info("All inference artifacts persisted.")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument('--input', default='data/processed/hotel_booking_preprocessed.csv', help='Input preprocessed CSV path')
    parser.add_argument('--output', default='data/processed/hotel_booking_features.csv', help='Output engineered CSV path')
    parser.add_argument('--keep-pii', action='store_true', help='Retain PII columns (for research only; NOT for production)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    target_col = 'is_canceled'

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded preprocessed data: {df.shape}")

    if not args.keep_pii:
        df = drop_pii(df)
    else:
        logger.warning("Retaining PII columns; ensure this is not used in production training.")

    feature_df, log = build_feature_dataset(df, target_col=target_col)
    logger.info(f"Feature dataset shape: {feature_df.shape}")

    save_features(feature_df, log, output_path)
    persist_artifacts(feature_df, log, target_col, Path('.'))
    print("✅ Feature engineering (simplified) complete")
    print(f"Final shape: {feature_df.shape}")
    print(f" Saved: {output_path}")
    if log.get('mean_target_encoding'):
        print("\n Mean Target Encoded Columns (corr):")
        for k, v in log['mean_target_encoding'].items():
            print(f" - {k} → {v['encoded_column']} (corr={v['correlation_with_target']:.4f})")


if __name__ == '__main__':
    main()