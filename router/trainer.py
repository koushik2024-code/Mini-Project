import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from router.keras_mlp import KerasMLPRouter, RouterConfig


@dataclass
class TrainingConfig:
    """Configuration for router training"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    class_weights: bool = True
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    random_seed: int = 42


class RouterTrainer:
    """
    Trainer for the Keras MLP Router.

    Handles training with:
    - Adam optimizer
    - Categorical Crossentropy loss
    - Early stopping
    - Model checkpointing
    - Learning rate reduction on plateau
    - Class weights for imbalanced data
    - Fixed random seed for reproducible results
    """

    def __init__(self, config: TrainingConfig | None = None, router_config: RouterConfig | None = None):
        self.config = config or TrainingConfig()
        self.router_config = router_config or RouterConfig()

        # Seed all RNGs before model construction so weight init, dropout,
        # and batch shuffling are reproducible run-to-run.
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)

        self.model = KerasMLPRouter(self.router_config).model
        self.history = None

    def get_callbacks(self, model_path: str) -> list[keras.callbacks.Callback]:
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            )
        ]
        return callbacks

    def compute_class_weights(self, y: np.ndarray) -> dict[int, float]:
        """Compute class weights for imbalanced datasets"""
        if not self.config.class_weights:
            return None

        # y is one-hot encoded, convert to class indices
        y_indices = np.argmax(y, axis=1)
        classes, counts = np.unique(y_indices, return_counts=True)

        total = len(y_indices)
        n_classes = len(classes)

        weights = {}
        for cls, count in zip(classes, counts, strict=True):
            weights[cls] = total / (n_classes * count)

        return weights

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        model_path: str = "",
    ) -> keras.callbacks.History:
        """
        Train the router model.

        Args:
            X_train: Training embeddings (n_samples, input_dim)
            y_train: Training labels one-hot (n_samples, num_classes)
            X_val: Validation embeddings
            y_val: Validation labels
            model_path: Path to save best model. Required - there is no
                default, so a caller can never silently overwrite the
                production model by forgetting to pass this.

        Returns:
            Training history
        """
        if not model_path:
            raise ValueError(
                "model_path is required - pass an explicit path (e.g. a "
                "tmp_path in tests) so training can never silently overwrite "
                "the production model at models/router/best_model.keras"
            )
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # Get callbacks
        callbacks = self.get_callbacks(model_path)

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test embeddings
            y_test: Test labels one-hot

        Returns:
            Dictionary with loss and accuracy
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': float(results[0]),
            'accuracy': float(results[1])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict(X, verbose=0)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def get_training_history(self) -> dict[str, list]:
        """Get training history as dictionary"""
        if self.history is None:
            return {}
        return {k: [float(v) for v in vals] for k, vals in self.history.history.items()}

    def save_history(self, path: str) -> None:
        """Save training history to JSON"""
        history_dict = self.get_training_history()
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    def save_model_and_metadata(
        self,
        model_path: str = "models/router/best_model.keras",
        metadata_path: str = "models/router/metadata.json",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        label_mapping: dict | None = None
    ) -> None:
        """
        Save model with metadata (embedding model name, label mapping, training config).

        Args:
            model_path: Path to save Keras model
            metadata_path: Path to save metadata JSON
            embedding_model_name: Name of embedding model used
            label_mapping: Label mapping dictionary
        """
        # Save model (already saved by ModelCheckpoint, but ensure it exists)
        if not Path(model_path).exists():
            self.model.save(model_path)

        # Save metadata
        # Convert tuple keys to strings for JSON serialization
        label_mapping_serializable = {}
        if label_mapping:
            for k, v in label_mapping.items():
                if isinstance(k, tuple):
                    label_mapping_serializable[str(k)] = v
                else:
                    label_mapping_serializable[k] = v

        metadata = {
            'embedding_model': embedding_model_name,
            'label_mapping': label_mapping_serializable,
            'training_config': asdict(self.config),
            'router_config': self.router_config.get_config() if hasattr(self.router_config, 'get_config') else asdict(self.router_config),
            'training_history': self.get_training_history()
        }

        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_training_config(config_path: str = "config.yaml") -> TrainingConfig:
    """Load training config from YAML file"""
    import yaml

    path = Path(config_path)
    if not path.exists():
        return TrainingConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    training_config = config.get('training', {})
    return TrainingConfig(
        epochs=training_config.get('epochs', 100),
        batch_size=training_config.get('batch_size', 32),
        learning_rate=training_config.get('learning_rate', 0.001),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        validation_split=training_config.get('validation_split', 0.1),
        class_weights=training_config.get('class_weights', True),
        random_seed=training_config.get('random_seed', 42)
    )


def main():
    """Quick test"""
    from data.ml_dataset_prep import MLDatasetPrep
    from embeddings.embedding_model import EmbeddingModel, load_embedding_config

    # Load data and prepare splits
    prep = MLDatasetPrep()
    data = prep.loader.load_dataset()
    train_idx, val_idx, test_idx = prep.split_data(data)

    # Generate embeddings
    emb_config = load_embedding_config()
    emb_model = EmbeddingModel(emb_config)

    # Get texts for training
    texts = [data[i]['prompt'] for i in train_idx]
    val_texts = [data[i]['prompt'] for i in val_idx]
    test_texts = [data[i]['prompt'] for i in test_idx]

    print("Generating embeddings...")
    X_train = emb_model.encode_batch(texts)
    X_val = emb_model.encode_batch(val_texts)
    X_test = emb_model.encode_batch(test_texts)

    # Prepare labels
    train_data = prep.prepare_training_data(data, None)
    y_train = train_data['y_train']
    y_val = train_data['y_val']
    y_test = train_data['y_test']

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Train
    trainer = RouterTrainer()
    print("Training...")
    _ = trainer.train(X_train, y_train, X_val, y_val, "models/router/best_model.keras")

    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate(X_test, y_test)
    print(f"Test results: {results}")

    # Save
    trainer.save_model_and_metadata(
        label_mapping=prep.label_mapping.rules
    )
    print("Model saved!")


if __name__ == "__main__":
    main()
