from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert 'message' in data and 'version' in data and 'endpoints' in data
    assert isinstance(data['endpoints'], dict)
    assert 'health' in data['endpoints']
