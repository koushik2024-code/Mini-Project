import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embeddings.embedding_model import EmbeddingModel, load_embedding_config
from embeddings.preprocessing import (
    TextPreprocessor,
    load_preprocessing_config,
)
from router.keras_mlp import KerasMLPRouter


@dataclass
class PredictorConfig:
    """Configuration for query prediction"""
    router_model_path: str = "models/router/best_model.keras"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_cache_dir: str = "embeddings/cache"
    metadata_path: str = "models/router/metadata.json"


class QueryPredictor:
    """
    Predict query difficulty class for new user queries.
    
    Pipeline:
    1. Preprocess query (same as training)
    2. Generate embedding (using cached embedding model)
    3. Predict class probabilities via Keras MLP
    4. Return predicted class + confidence
    """

    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()

        # Load router model
        self.router = KerasMLPRouter.load(self.config.router_model_path)

        # Load embedding model (with caching)
        emb_config = load_embedding_config()
        emb_config.model_name = self.config.embedding_model_name
        emb_config.device = self.config.embedding_device
        emb_config.cache_dir = self.config.embedding_cache_dir
        self.embedding_model = EmbeddingModel(emb_config)

        # Load preprocessor (same as training)
        prep_config = load_preprocessing_config()
        self.preprocessor = TextPreprocessor(prep_config)

        # Load metadata for label mapping
        self.metadata = self._load_metadata()
        self.label_mapping = self._build_label_mapping()

    def _load_metadata(self) -> dict[str, Any]:
        """Load model metadata"""
        try:
            with open(self.config.metadata_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _build_label_mapping(self) -> dict[int, str]:
        """Build index to difficulty class mapping"""
        # Default mapping: 0=easy, 1=medium, 2=hard
        return {0: "easy", 1: "medium", 2: "hard"}

    def predict(self, query: str | None) -> dict[str, Any]:
        """
        Predict difficulty class for a single query.
        
        Args:
            query: User query string
            
        Returns:
            Dict with predicted_class, confidence, probabilities
        """
        if query is None or query == "":
            query = ""

        # Step 1: Preprocess (same as training)
        processed = self.preprocessor.preprocess(query)

        # Step 2: Generate embedding
        embedding = self.embedding_model.encode(processed)

        # Step 3: Predict probabilities
        probs = self.router.predict(embedding.reshape(1, -1))[0]

        # Step 4: Get predicted class and confidence
        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        predicted_class = self.label_mapping[pred_idx]

        # Step 5: Format probabilities
        class_names = ["easy", "medium", "hard"]
        probabilities = {name: float(probs[i]) for i, name in enumerate(class_names)}

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

    def predict_batch(self, queries: list[str | None]) -> list[dict[str, Any]]:
        """
        Predict difficulty classes for multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of prediction dicts
        """
        if not queries:
            return []

        # Preprocess all queries
        processed = [self.preprocessor.preprocess(q) if q else "" for q in queries]

        # Generate embeddings (batch)
        embeddings = self.embedding_model.encode_batch(processed)

        # Predict probabilities (batch)
        probs_batch = self.router.predict(embeddings)

        # Format results
        class_names = ["easy", "medium", "hard"]
        results = []
        for probs in probs_batch:
            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            predicted_class = self.label_mapping[pred_idx]
            probabilities = {name: float(probs[i]) for i, name in enumerate(class_names)}

            results.append({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities
            })

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get model information"""
        return {
            "router_path": self.config.router_model_path,
            "embedding_model": self.config.embedding_model_name,
            "embedding_dim": self.router.config.input_dim,
            "num_classes": self.router.config.num_classes,
            "metadata": self.metadata
        }


def load_predictor_config(config_path: str = "config.yaml") -> PredictorConfig:
    """Load predictor config from YAML file"""
    import yaml

    path = Path(config_path)
    if not path.exists():
        return PredictorConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    # Use router config for model path
    router_config = config.get('router', {})
    return PredictorConfig(
        router_model_path=router_config.get('model_path', "models/router/best_model.keras"),
        embedding_model_name=config.get('embedding', {}).get('model', "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=config.get('embedding', {}).get('device', "cpu"),
        embedding_cache_dir=config.get('embedding', {}).get('cache_dir', "embeddings/cache")
    )


if __name__ == "__main__":
    # Quick test
    predictor = QueryPredictor()

    test_queries = [
        "What is 2+2?",
        "Explain how a binary search tree works",
        "Write a Python program to implement a distributed consensus algorithm"
    ]

    print("Model Info:", predictor.get_model_info())
    print()

    for q in test_queries:
        result = predictor.predict(q)
        print(f"Query: {q}")
        print(f"  Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
        print(f"  Probs: {result['probabilities']}")
        print()
