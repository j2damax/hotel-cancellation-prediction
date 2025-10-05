from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data and 'model_loaded' in data
    # status should be healthy if model loaded successfully; allow fallback otherwise
    assert isinstance(data['model_loaded'], bool)
    if data['model_loaded']:
        assert data['status'] == 'healthy'
