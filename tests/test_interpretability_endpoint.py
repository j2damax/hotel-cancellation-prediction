import os
import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_interpretability_endpoint_structure():
    # Call endpoint
    resp = client.get("/model/interpretability?top_k=5")
    assert resp.status_code == 200
    data = resp.json()

    # Basic keys presence
    for key in [
        'champion_model', 'shap_generated', 'top_features', 'local_examples',
        'feature_name_map', 'artifacts_available'
    ]:
        assert key in data

    # top_features constraints when present
    if data['top_features']:
        assert len(data['top_features']) <= 5
        assert all('feature' in f and 'mean_abs_shap' in f for f in data['top_features'])

    # local examples shape
    if data['local_examples']:
        ex = data['local_examples'][0]
        for k in ['category', 'top_positive_contributors', 'top_negative_contributors']:
            assert k in ex

    # feature name map should be dict
    assert isinstance(data['feature_name_map'], dict)


def test_interpretability_top_k_param():
    resp = client.get("/model/interpretability?top_k=3")
    assert resp.status_code == 200
    data = resp.json()
    if data['top_features']:
        assert len(data['top_features']) <= 3
