import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data/processed')
ART_DIR = Path('artifacts')
TARGET = 'is_canceled'

REQUIRED_ARTIFACTS = [
    'mte_mappings.json',
    'feature_contract.json',
    'feature_schema.json',
    'feature_rules.json',
    'distribution_baselines.json'
]


def test_artifact_files_exist():
    missing = [f for f in REQUIRED_ARTIFACTS if not (ART_DIR / f).exists()]
    assert not missing, f"Missing artifact files: {missing}"


def test_feature_contract_matches_dataset():
    contract_path = ART_DIR / 'feature_contract.json'
    features_path = DATA_DIR / 'hotel_booking_features.csv'
    assert contract_path.exists(), 'feature_contract.json missing'
    assert features_path.exists(), 'engineered feature dataset missing'

    with open(contract_path) as f:
        contract = json.load(f)
    feature_order = contract['feature_order']

    df = pd.read_csv(features_path, nrows=100)  # sample sufficient for columns
    dataset_cols = [c for c in df.columns if c != TARGET]

    assert set(feature_order) == set(dataset_cols), 'Feature set mismatch between contract and dataset columns'
    # Maintain order check (optional but asserts stability)
    assert feature_order == dataset_cols, 'Column order differs from contract'


def test_mte_global_means_close():
    mte_path = ART_DIR / 'mte_mappings.json'
    features_path = DATA_DIR / 'hotel_booking_features.csv'
    if not mte_path.exists():
        return  # nothing to test
    with open(mte_path) as f:
        mte = json.load(f)
    df = pd.read_csv(features_path, usecols=[TARGET])
    actual_mean = df[TARGET].mean()

    for base_col, meta in mte['encodings'].items():
        gm = meta['global_mean']
        assert gm is not None, f'Global mean missing for {base_col}'
        assert abs(gm - actual_mean) < 1e-6, f'Global mean drift for {base_col}: stored={gm} actual={actual_mean}'
