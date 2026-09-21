import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from tests.auth_helpers import auth_headers


class TestFastAPIBackend:
    """Integration tests for PRD-0001-FR-25, FR-26, FR-27: FastAPI Backend"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app, headers=auth_headers())

    def test_health_endpoint(self):
        """Test GET /health endpoint"""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "router_loaded" in data
        assert "embedding_loaded" in data
        assert "models_loaded" in data

    def test_models_endpoint(self):
        """Test GET /models endpoint"""
        response = self.client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

        # Check all three tiers
        tiers = {m["tier"] for m in data["models"]}
        assert "easy" in tiers
        assert "medium" in tiers
        assert "hard" in tiers

        for model in data["models"]:
            assert "tier" in model
            assert "name" in model
            assert "loaded" in model

    def test_chat_endpoint_valid(self):
        """Test POST /chat with valid query"""
        response = self.client.post("/chat", json={"query": "What is 2+2?"})

        assert response.status_code == 200
        data = response.json()

        required_keys = ["query", "predicted_class", "confidence", "selected_model", "response", "fallback_used", "latency_ms"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        assert data["query"] == "What is 2+2?"
        assert data["predicted_class"] in ["easy", "medium", "hard"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["response"], str)
        assert isinstance(data["fallback_used"], bool)
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0

    def test_chat_endpoint_empty_query(self):
        """Test POST /chat with empty query"""
        response = self.client.post("/chat", json={"query": ""})

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_chat_endpoint_missing_query(self):
        """Test POST /chat with missing query field"""
        response = self.client.post("/chat", json={})

        assert response.status_code in [400, 422]  # Validation error

    def test_chat_endpoint_invalid_json(self):
        """Test POST /chat with invalid JSON"""
        response = self.client.post("/chat", data="not json", headers={"Content-Type": "application/json"})

        assert response.status_code == 422

    def test_chat_endpoint_whitespace_query(self):
        """Test POST /chat with whitespace-only query"""
        response = self.client.post("/chat", json={"query": "   "})

        assert response.status_code == 400

    def test_chat_response_structure(self):
        """Test chat response has correct structure"""
        response = self.client.post("/chat", json={"query": "Explain binary search"})

        assert response.status_code == 200
        data = response.json()

        assert data["predicted_class"] in ["easy", "medium", "hard"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["selected_model"] in ["qwen3:1.7b", "qwen3:4b", "llama3.2:3b"]
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["fallback_used"], bool)
        assert data["latency_ms"] >= 0

    def test_chat_with_various_queries(self):
        """Test chat endpoint with various query types"""
        queries = [
            "What is 2+2?",
            "Explain how a binary search tree works",
            "Write a Python function to sort a list",
            "What is the capital of France?",
        ]

        for query in queries:
            response = self.client.post("/chat", json={"query": query})
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == query
            assert "response" in data
            assert len(data["response"]) > 0


class TestAPIEndpoints:
    """Tests for API endpoint availability"""

    def setup_method(self):
        self.client = TestClient(app, headers=auth_headers())

    def test_endpoints_exist(self):
        """Test all required endpoints exist"""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/models"),
            ("POST", "/chat"),
        ]

        for method, path in endpoints:
            if method == "GET":
                response = self.client.get(path)
            elif method == "POST":
                response = self.client.post(path, json={"query": "test"})

            # Should not be 404
            assert response.status_code != 404, f"Endpoint {method} {path} not found"


if __name__ == "__main__":
    # Quick manual test
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app, headers=auth_headers())

    print("Testing /health...")
    r = client.get("/health")
    print(f"  Status: {r.status_code}")
    print(f"  Data: {r.json()}")

    print("\nTesting /models...")
    r = client.get("/models")
    print(f"  Status: {r.status_code}")
    print(f"  Models: {len(r.json().get('models', []))}")

    print("\nTesting /chat...")
    r = client.post("/chat", json={"query": "What is 2+2?"})
    print(f"  Status: {r.status_code}")
    d = r.json()
    print(f"  Class: {d.get('predicted_class')}")
    print(f"  Confidence: {d.get('confidence'):.4f}")
    print(f"  Model: {d.get('selected_model')}")
    print(f"  Response: {d.get('response')[:50]}...")
    print(f"  Fallback: {d.get('fallback_used')}")
    print(f"  Latency: {d.get('latency_ms')}ms")
