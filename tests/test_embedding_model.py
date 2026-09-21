import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedding_model import EmbeddingConfig, EmbeddingModel


class TestEmbeddingModel:
    """Unit tests for PRD-0001-FR-4: Generate embeddings with caching"""

    def setup_method(self):
        """Set up embedding model with test config"""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir="embeddings/cache/test",
            batch_size=32
        )
        self.embedding_model = EmbeddingModel(config)

    def test_model_loads(self):
        """Test that embedding model loads correctly"""
        assert self.embedding_model.model is not None
        assert self.embedding_model.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_embedding_dimension(self):
        """Test that embedding dimension is correct"""
        dim = self.embedding_model.get_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2 dimension

    def test_encode_single_query(self):
        """Test encoding a single query"""
        text = "What is the capital of France?"
        embedding = self.embedding_model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_encode_batch(self):
        """Test encoding a batch of queries"""
        texts = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain quantum computing"
        ]
        embeddings = self.embedding_model.encode_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_encode_batch_respects_batch_size(self):
        """Test that batch encoding respects batch_size config"""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir="embeddings/cache/test",
            batch_size=2
        )
        model = EmbeddingModel(config)

        texts = ["Query " + str(i) for i in range(5)]
        embeddings = model.encode_batch(texts)

        assert embeddings.shape == (5, 384)

    def test_cache_save_and_load(self, tmp_path):
        """Test embedding cache save and load"""
        cache_dir = tmp_path / "embeddings_cache"
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir=str(cache_dir),
            batch_size=32
        )
        model = EmbeddingModel(config)

        # Generate embeddings for some texts
        texts = ["Test query 1", "Test query 2"]
        embeddings1 = model.encode_batch(texts)

        # Save cache
        model.save_cache()

        # Create new model instance with same cache dir
        model2 = EmbeddingModel(config)
        embeddings2 = model2.encode_batch(texts)

        # Should be identical (loaded from cache)
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_cache_invalidation_on_model_change(self, tmp_path):
        """Test cache invalidation when embedding model changes"""
        cache_dir = tmp_path / "embeddings_cache"
        config1 = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir=str(cache_dir),
            batch_size=32
        )
        model1 = EmbeddingModel(config1)

        texts = ["Test query"]
        model1.encode_batch(texts)
        model1.save_cache()

        # Create new config with different model name
        config2 = EmbeddingConfig(
            model_name="different-model-name",
            device="cpu",
            cache_dir=str(cache_dir),
            batch_size=32
        )

        # Verify cache metadata includes model name
        # The EmbeddingModel should detect model change via metadata
        assert model1.config.model_name != config2.model_name

        # Load cache metadata and verify it has model_name
        meta_path = cache_dir / "cache_metadata.json"
        assert meta_path.exists()
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta.get('model_name') == config1.model_name

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        embedding = self.embedding_model.encode("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_none_text_handling(self):
        """Test handling of None text"""
        embedding = self.embedding_model.encode(None)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_similar_queries_have_similar_embeddings(self):
        """Test that semantically similar queries have similar embeddings"""
        text1 = "What is machine learning?"
        text2 = "Explain machine learning"
        text3 = "How to cook pasta?"

        emb1 = self.embedding_model.encode(text1)
        emb2 = self.embedding_model.encode(text2)
        emb3 = self.embedding_model.encode(text3)

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        # Similar queries should have higher similarity
        assert sim_12 > sim_13


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EmbeddingConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.batch_size == 32

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'model_name': 'test-model',
            'device': 'cuda',
            'cache_dir': 'custom/cache',
            'batch_size': 64
        }
        config = EmbeddingConfig(**config_dict)
        assert config.model_name == 'test-model'
        assert config.device == 'cuda'
        assert config.cache_dir == 'custom/cache'
        assert config.batch_size == 64
