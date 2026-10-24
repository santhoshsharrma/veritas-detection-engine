from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_text_validates_empty_input():
    assert client.post("/v1/analyze/text", json={"text": ""}).status_code == 422
