import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import app
from tests.auth_helpers import auth_headers


@pytest.fixture
def client():
    return TestClient(app, headers=auth_headers())


@pytest.fixture
def anon():
    return TestClient(app)


class TestRouteEndpoint:
    """POST /route - classify without generating"""

    def test_returns_routing_decision(self, client):
        response = client.post("/route", json={"query": "What is 2+2?"})

        assert response.status_code == 200
        data = response.json()

        for key in [
            "predicted_class", "confidence", "selected_tier", "selected_model",
            "fallback_used", "confidence_threshold", "latency_ms"
        ]:
            assert key in data, f"Missing key: {key}"

        assert data["predicted_class"] in ["easy", "medium", "hard"]
        assert data["selected_tier"] in ["easy", "medium", "hard"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["fallback_used"], bool)

    def test_does_not_generate_an_answer(self, client):
        data = client.post("/route", json={"query": "What is 2+2?"}).json()
        assert "response" not in data

    def test_does_not_write_history(self, client):
        before = client.get("/history").json()["total"]
        client.post("/route", json={"query": "What is 2+2?"})
        assert client.get("/history").json()["total"] == before

    def test_is_much_faster_than_chat(self, client):
        # Routing is the cheap half: no LLM call at all
        data = client.post("/route", json={"query": "What is 2+2?"}).json()
        assert data["latency_ms"] < 5000

    def test_agrees_with_chat_for_the_same_query(self, client):
        query = "What is 2+2?"
        routed = client.post("/route", json={"query": query}).json()
        chatted = client.post("/chat", json={"query": query}).json()

        assert routed["predicted_class"] == chatted["predicted_class"]
        assert routed["selected_model"] == chatted["selected_model"]
        assert routed["fallback_used"] == chatted["fallback_used"]
        assert routed["confidence"] == pytest.approx(chatted["confidence"], abs=1e-6)

    def test_fallback_is_consistent_with_the_threshold(self, client):
        data = client.post("/route", json={"query": "What is 2+2?"}).json()

        if data["confidence"] < data["confidence_threshold"]:
            assert data["fallback_used"] is True
            assert data["selected_tier"] == main.policy.config.fallback_tier
        else:
            assert data["fallback_used"] is False
            assert data["selected_tier"] == data["predicted_class"]

    def test_requires_auth(self, anon):
        assert anon.post("/route", json={"query": "hi"}).status_code == 401

    def test_empty_query_rejected(self, client):
        assert client.post("/route", json={"query": ""}).status_code == 400

    def test_whitespace_query_rejected(self, client):
        assert client.post("/route", json={"query": "   "}).status_code == 400

    def test_overlong_query_rejected(self, client):
        assert client.post("/route", json={"query": "x" * 5000}).status_code == 400


class TestThresholdExposed:
    """The UI needs the threshold to draw the fallback tick"""

    def test_health_reports_threshold(self, anon):
        data = anon.get("/health").json()

        assert "confidence_threshold" in data
        assert data["confidence_threshold"] == main.policy.config.confidence_threshold
        assert 0.0 <= data["confidence_threshold"] <= 1.0

    def test_chat_reports_threshold(self, client):
        data = client.post("/chat", json={"query": "What is 2+2?"}).json()
        assert data["confidence_threshold"] == main.policy.config.confidence_threshold
