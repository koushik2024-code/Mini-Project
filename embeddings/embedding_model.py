import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    cache_dir: str = "embeddings/cache"
    batch_size: int = 32


class EmbeddingModel:
    """
    Embedding model with caching support.

    Generates embeddings using Sentence Transformers and caches them
    to disk for reuse. Cache is invalidated when model changes.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

        # Cache metadata
        self.cache_meta_path = self.cache_dir / "cache_metadata.json"
        self.cache_data_path = self.cache_dir / "embeddings_cache.pkl"

        # Load existing cache
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model"""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, text: str | None) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Input text (None or empty string returns zero vector)

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if text is None or text == "":
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Cache it
        self._cache[cache_key] = embedding.copy()
        return embedding

    def encode_batch(self, texts: list[str | None]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.

        Args:
            texts: List of input texts

        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.get_embedding_dimension()), dtype=np.float32)

        # Separate cached and uncached
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text is None or text == "":
                cached_embeddings[i] = np.zeros(self.get_embedding_dimension(), dtype=np.float32)
            else:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_embeddings[i] = self._cache[cache_key].copy()
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)

            # Cache new embeddings
            for text, emb in zip(uncached_texts, new_embeddings, strict=True):
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = emb.copy()

            # Add to cached
            for idx, emb in zip(uncached_indices, new_embeddings, strict=True):
                cached_embeddings[idx] = emb

        # Assemble result in original order
        result = np.zeros((len(texts), self.get_embedding_dimension()), dtype=np.float32)
        for i in range(len(texts)):
            result[i] = cached_embeddings.get(i, np.zeros(self.get_embedding_dimension(), dtype=np.float32))

        return result

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text + model name for cache key
        content = f"{self.config.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk"""
        if not self.cache_meta_path.exists() or not self.cache_data_path.exists():
            return

        try:
            # Check if cache is valid for current model
            with open(self.cache_meta_path) as f:
                meta = json.load(f)

            if meta.get('model_name') != self.config.model_name:
                # Model changed, invalidate cache
                return

            # Load embeddings
            with open(self.cache_data_path, 'rb') as f:
                self._cache = pickle.load(f)
        except (json.JSONDecodeError, pickle.UnpicklingError, KeyError):
            # Corrupted cache, start fresh
            self._cache = {}

    def save_cache(self) -> None:
        """Save cache to disk"""
        try:
            # Save metadata
            meta = {
                'model_name': self.config.model_name,
                'embedding_dim': self.get_embedding_dimension(),
                'cache_size': len(self._cache)
            }
            with open(self.cache_meta_path, 'w') as f:
                json.dump(meta, f)

            # Save embeddings
            with open(self.cache_data_path, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception:
            # Fail silently - caching is optional
            pass

    def clear_cache(self) -> None:
        """Clear cache"""
        self._cache = {}
        if self.cache_meta_path.exists():
            self.cache_meta_path.unlink()
        if self.cache_data_path.exists():
            self.cache_data_path.unlink()


def load_embedding_config(config_path: str = "config.yaml") -> EmbeddingConfig:
    """Load embedding config from YAML file"""
    import yaml

    path = Path(config_path)
    if not path.exists():
        return EmbeddingConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    emb_config = config.get('embedding', {})
    return EmbeddingConfig(
        model_name=emb_config.get('model_name', "sentence-transformers/all-MiniLM-L6-v2"),
        device=emb_config.get('device', "cpu"),
        cache_dir=emb_config.get('cache_dir', "embeddings/cache"),
        batch_size=emb_config.get('batch_size', 32)
    )


if __name__ == "__main__":
    # Quick test
    config = EmbeddingConfig(cache_dir="embeddings/cache/test")
    model = EmbeddingModel(config)

    print(f"Model: {config.model_name}")
    print(f"Embedding dim: {model.get_embedding_dimension()}")

    texts = ["Hello world", "Machine learning is cool", "Hello world"]  # Duplicate to test cache
    embeddings = model.encode_batch(texts)
    print(f"Batch shape: {embeddings.shape}")

    # Test cache
    model.save_cache()
    print("Cache saved")
