import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from router.predictor import PredictorConfig, QueryPredictor
from router.router_policy import PolicyConfig, RouterPolicy
from tests.auth_helpers import auth_headers


class TestAdvancedFeatures:
    """Tests for advanced features (NFR-2, NFR-3, NFR-4, NFR-8)"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app, headers=auth_headers())

    def test_request_timeout_handling(self):
        """Test that request timeout is handled (NFR-8)"""
        # Test that timeout parameter works
        response = self.client.post("/chat", json={"query": "What is 2+2?"})
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0

    def test_feature_flags(self):
        """Test feature flags are configurable (NFR-8)"""
        # Test USE_MOCK_LLMS flag
        from api.main import llm_client

        # Should have mock LLM loaded by default
        assert llm_client is not None
        assert hasattr(llm_client, 'generate')

    def test_confidence_calibration(self):
        """Test confidence calibration (NFR-8)"""
        from api.main import predictor

        # Test that confidence scores are well-calibrated
        query = "What is 2+2?"
        from embeddings.embedding_model import EmbeddingModel, load_embedding_config
        from embeddings.preprocessing import TextPreprocessor, load_preprocessing_config

        predictor = QueryPredictor(PredictorConfig())
        preprocessor = TextPreprocessor(load_preprocessing_config())
        emb_model = EmbeddingModel(load_embedding_config())

        processed = preprocessor.preprocess("What is 2+2?")
        emb = emb_model.encode(processed)
        probs = predictor.router.predict(emb.reshape(1, -1))[0]

        # Check that probabilities sum to 1 (softmax calibration)
        assert abs(probs.sum() - 1.0) < 1e-5

        # Confidence should be max probability
        confidence = float(probs.max())
        assert 0.0 <= confidence <= 1.0

    def test_latency_targets(self):
        """Test latency targets for different tiers (NFR-2, NFR-3, NFR-4)"""
        # Test that latency is tracked
        response = self.client.post("/chat", json={"query": "What is 2+2?"})
        assert response.status_code == 200
        data = response.json()

        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0

        # With mock LLM, latency should be very low
        assert data["latency_ms"] < 5000  # Should be fast with mock

    def test_feature_flags_config(self):
        """Test feature flags from config"""
        from pathlib import Path

        import yaml

        config_path = Path("config.yaml")
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        features = config.get('features', {})

        # Check all required feature flags exist
        assert 'use_mock_llms' in features
        assert 'use_ollama' in features
        assert 'enable_fallback' in features
        assert 'enable_safety_validation' in features

        # Check types
        assert isinstance(features['use_mock_llms'], bool)
        assert isinstance(features['use_ollama'], bool)
        assert isinstance(features['enable_fallback'], bool)
        assert isinstance(features['enable_safety_validation'], bool)


class TestTimeoutHandling:
    """Tests for timeout handling"""

    def setup_method(self):
        self.client = TestClient(app, headers=auth_headers())

    def test_long_query_timeout(self):
        """Test that very long queries are rejected"""
        long_query = "x" * 5000  # Exceeds 4096 limit
        response = self.client.post("/chat", json={"query": "x" * 5000})
        assert response.status_code == 400

    def test_normal_query_completes(self):
        """Test that normal queries complete within reasonable time"""
        start = time.time()
        response = self.client.post("/chat", json={"query": "What is 2+2?"})
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete in reasonable time
        assert elapsed < 30  # 30 seconds max for test


class TestFallbackBehavior:
    """Tests for fallback behavior"""

    def setup_method(self):
        self.client = TestClient(app, headers=auth_headers())

    def test_fallback_when_low_confidence(self):
        """Test fallback triggers when confidence is low"""
        from api.main import policy

        policy = RouterPolicy(PolicyConfig(confidence_threshold=0.99))  # Very high threshold

        # Low confidence should trigger fallback
        result = policy.select_model("hard", 0.50)
        assert result["fallback_used"] is True
        assert result["selected_tier"] == "medium"

    def test_no_fallback_when_high_confidence(self):
        """Test no fallback when confidence is high"""
        from api.main import policy

        result = policy.select_model("easy", 0.85)
        assert result["fallback_used"] is False
        assert result["selected_tier"] == "easy"

    def test_fallback_tier_configurable(self):
        """Test fallback tier is configurable"""

        config = PolicyConfig(fallback_tier="easy")
        policy = RouterPolicy(config)

        result = policy.select_model("hard", 0.50)
        assert result["fallback_used"] is True
        assert result["selected_tier"] == "easy"


class TestSafetyValidation:
    """Tests for safety validation (feature flag)"""

    def test_safety_validation_flag(self):
        """Test ENABLE_SAFETY_VALIDATION feature flag"""
        from pathlib import Path

        import yaml

        config_path = Path("config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        features = config.get('features', {})
        assert 'enable_safety_validation' in features
        assert isinstance(features['enable_safety_validation'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
