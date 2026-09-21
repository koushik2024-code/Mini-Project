import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class RouterConfig:
    """Configuration for Keras MLP Router"""
    input_dim: int = 384
    hidden_dims: list[int] = None
    num_classes: int = 3
    dropout_rate: float = 0.2
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class KerasMLPRouter:
    """
    Keras MLP Router for query difficulty classification.

    Architecture:
    Embedding → Dense(256) → ReLU → Dropout → Dense(128) → ReLU → Dropout → Dense(3) → Softmax

    This implements the exact architecture specified in PRD:
    Embedding → Dense → ReLU → Dense → ReLU → Dense(3) → Softmax
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self) -> keras.Model:
        """Build the MLP architecture"""
        inputs = keras.Input(shape=(self.config.input_dim,), name="embedding_input")

        x = inputs

        # Hidden layers
        for i, dim in enumerate(self.config.hidden_dims):
            x = layers.Dense(
                dim,
                activation='relu',
                name=f"dense_{i+1}"
            )(x)
            x = layers.Dropout(
                self.config.dropout_rate,
                name=f"dropout_{i+1}"
            )(x)

        # Output layer with Softmax
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name="output"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="query_difficulty_router")
        return model

    def _compile_model(self) -> None:
        """Compile model with Adam optimizer and Categorical Crossentropy"""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for embeddings.

        Args:
            embeddings: Array of shape (batch_size, input_dim)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        return self.model.predict(embeddings, verbose=0)

    def predict_classes(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class labels for embeddings.

        Args:
            embeddings: Array of shape (batch_size, input_dim)

        Returns:
            Class indices of shape (batch_size,)
        """
        probs = self.predict(embeddings)
        return np.argmax(probs, axis=1)

    def get_config(self) -> dict[str, Any]:
        """Get model configuration"""
        return asdict(self.config)

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model (e.g., 'models/router/best_model.keras')
        """
        # Save model architecture and weights
        self.model.save(path)

        # Save config separately for easy loading
        config_path = Path(path).with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'KerasMLPRouter':
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded KerasMLPRouter instance
        """
        # Load config
        config_path = Path(path).with_suffix('.json')
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = RouterConfig(**config_dict)
        else:
            config = RouterConfig()

        # Create instance and load weights
        router = cls(config)
        router.model = keras.models.load_model(path)
        return router

    def summary(self) -> str:
        """Get model summary as string"""
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

    def get_embedding_dimension(self) -> int:
        """Get input embedding dimension"""
        return self.config.input_dim

    def get_num_classes(self) -> int:
        """Get number of output classes"""
        return self.config.num_classes


def load_router_config(config_path: str = "config.yaml") -> RouterConfig:
    """Load router config from YAML file"""
    import yaml

    path = Path(config_path)
    if not path.exists():
        return RouterConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    router_config = config.get('router', {})
    return RouterConfig(
        input_dim=router_config.get('embedding_dim', 384),
        hidden_dims=router_config.get('hidden_dims', [256, 128]),
        num_classes=3,
        dropout_rate=router_config.get('dropout_rate', 0.2),
        learning_rate=router_config.get('learning_rate', 0.001)
    )


if __name__ == "__main__":
    # Quick test
    config = RouterConfig()
    router = KerasMLPRouter(config)

    print("Model Summary:")
    print(router.summary())

    print("\nConfig:")
    print(router.get_config())

    # Test forward pass
    dummy = np.random.randn(2, 384).astype(np.float32)
    output = router.predict(dummy)
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output sum: {output.sum(axis=1)}")
    print(f"Test output:\n{output}")
