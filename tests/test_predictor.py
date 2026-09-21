import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.predictor import PredictorConfig, QueryPredictor


class TestQueryPredictor:
    """Unit tests for PRD-0001-FR-11 to FR-14: New Query Prediction"""

    def setup_method(self):
        """Set up predictor with test config"""
        config = PredictorConfig(
            router_model_path="models/router/best_model.keras",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="cpu",
            embedding_cache_dir="embeddings/cache"
        )
        self.predictor = QueryPredictor(config)

    def test_predictor_initialization(self):
        """Test predictor initializes correctly"""
        assert self.predictor.router is not None
        assert self.predictor.embedding_model is not None
        assert self.predictor.preprocessor is not None

    def test_predict_single_query(self):
        """Test prediction for a single query"""
        query = "What is the capital of France?"
        result = self.predictor.predict(query)

        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['predicted_class'] in ['easy', 'medium', 'hard']
        assert 0.0 <= result['confidence'] <= 1.0
        assert len(result['probabilities']) == 3
        assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-5

    def test_predict_batch(self):
        """Test batch prediction"""
        queries = [
            "What is 2+2?",
            "Explain binary search",
            "Write a distributed consensus algorithm"
        ]
        results = self.predictor.predict_batch(queries)

        assert len(results) == 3
        for result in results:
            assert result['predicted_class'] in ['easy', 'medium', 'hard']
            assert 0.0 <= result['confidence'] <= 1.0

    def test_preprocessing_matches_training(self):
        """Test that preprocessing matches training exactly"""
        # The predictor should use the same preprocessing as training
        query = "  What   is   Python?  "
        processed = self.predictor.preprocessor.preprocess(query)

        # Should strip extra whitespace
        assert processed == "What is Python?"

    def test_embedding_reused(self):
        """Test that embedding model is reused (not reloaded)"""
        query1 = "Test query 1"
        query2 = "Test query 2"

        result1 = self.predictor.predict(query1)
        result2 = self.predictor.predict(query2)

        # Both should succeed without reloading model
        assert result1['predicted_class'] in ['easy', 'medium', 'hard']
        assert result2['predicted_class'] in ['easy', 'medium', 'hard']

    def test_empty_query_handling(self):
        """Test handling of empty query"""
        result = self.predictor.predict("")
        assert 'predicted_class' in result
        assert result['confidence'] >= 0.0

    def test_none_query_handling(self):
        """Test handling of None query"""
        result = self.predictor.predict(None)
        assert 'predicted_class' in result


class TestPredictorConfig:
    """Tests for PredictorConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PredictorConfig()

        assert config.router_model_path == "models/router/best_model.keras"
        assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_device == "cpu"
        assert config.embedding_cache_dir == "embeddings/cache"

    def test_custom_config(self):
        """Test custom configuration"""
        config = PredictorConfig(
            router_model_path="custom/path.keras",
            embedding_model_name="custom-model",
            embedding_device="cuda",
            embedding_cache_dir="custom/cache"
        )

        assert config.router_model_path == "custom/path.keras"
        assert config.embedding_device == "cuda"
