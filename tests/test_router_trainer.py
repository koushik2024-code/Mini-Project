import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.trainer import RouterTrainer, TrainingConfig


class TestRouterTrainer:
    """Unit tests for PRD-0001-FR-7, FR-8: Router training and evaluation"""

    def setup_method(self):
        """Set up trainer with test config"""
        config = TrainingConfig(
            epochs=2,  # Small for testing
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5,
            validation_split=0.1,
            class_weights=False
        )
        self.trainer = RouterTrainer(config)

        # Create dummy data
        self.n_samples = 100
        self.input_dim = 384
        self.X = np.random.randn(self.n_samples, self.input_dim).astype(np.float32)
        self.y = np.random.randint(0, 3, (self.n_samples, 3))
        # Make one-hot
        y_onehot = np.zeros((self.n_samples, 3), dtype=np.float32)
        for i in range(self.n_samples):
            y_onehot[i, self.y[i]] = 1.0
        self.y = y_onehot

    def test_trainer_initialization(self):
        """Test trainer initializes correctly"""
        assert self.trainer.config.epochs == 2
        assert self.trainer.config.batch_size == 32
        assert self.trainer.model is not None

    def test_train_with_validation(self, tmp_path):
        """Test training with validation data"""
        # Split data
        split = int(0.8 * self.n_samples)
        X_train, X_val = self.X[:split], self.X[split:]
        y_train, y_val = self.y[:split], self.y[split:]

        history = self.trainer.train(X_train, y_train, X_val, y_val, str(tmp_path / "test_model.keras"))

        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 2  # 2 epochs

    def test_early_stopping_callback(self):
        """Test early stopping is configured"""
        callbacks = self.trainer.get_callbacks("test_model.keras")

        # Check EarlyStopping callback exists
        early_stop = [cb for cb in callbacks if cb.__class__.__name__ == 'EarlyStopping']
        assert len(early_stop) == 1
        assert early_stop[0].patience == 5

    def test_model_checkpoint_callback(self):
        """Test model checkpoint is configured"""
        callbacks = self.trainer.get_callbacks("test_model.keras")

        # Check ModelCheckpoint callback exists
        checkpoint = [cb for cb in callbacks if cb.__class__.__name__ == 'ModelCheckpoint']
        assert len(checkpoint) == 1
        assert checkpoint[0].save_best_only is True

    def test_evaluate_model(self, tmp_path):
        """Test model evaluation"""
        # Split data
        split = int(0.8 * self.n_samples)
        X_train, X_test = self.X[:split], self.X[split:]
        y_train, y_test = self.y[:split], self.y[split:]

        # Quick train
        self.trainer.train(X_train, y_train, X_train[:10], y_train[:10], str(tmp_path / "test_model.keras"))

        # Evaluate
        results = self.trainer.evaluate(X_test, y_test)

        assert 'loss' in results
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1

    def test_predict_with_trained_model(self, tmp_path):
        """Test prediction after training"""
        split = int(0.8 * self.n_samples)
        X_train, y_train = self.X[:split], self.y[:split]

        self.trainer.train(X_train, y_train, model_path=str(tmp_path / "test_model.keras"))

        # Predict
        preds = self.trainer.predict(X_train[:5])

        assert preds.shape == (5, 3)
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_classes(self, tmp_path):
        """Test class prediction"""
        split = int(0.8 * self.n_samples)
        X_train, y_train = self.X[:split], self.y[:split]

        self.trainer.train(X_train, y_train, model_path=str(tmp_path / "test_model.keras"))

        classes = self.trainer.predict_classes(X_train[:5])

        assert classes.shape == (5,)
        assert np.all((classes >= 0) & (classes < 3))

    def test_class_weights_computation(self):
        """Test class weights are computed for imbalanced data"""
        # Create imbalanced data
        y_imbalanced = np.zeros((100, 3), dtype=np.float32)
        y_imbalanced[:70, 0] = 1  # 70% class 0
        y_imbalanced[70:90, 1] = 1  # 20% class 1
        y_imbalanced[90:, 2] = 1   # 10% class 2

        # Use config with class_weights=True
        config = TrainingConfig(class_weights=True)
        trainer = RouterTrainer(config)

        weights = trainer.compute_class_weights(y_imbalanced)

        assert weights is not None
        assert len(weights) == 3
        # Class 2 (minority) should have higher weight
        assert weights[2] > weights[0]


class TestTrainingConfig:
    """Tests for TrainingConfig"""

    def test_default_config(self):
        """Test default training configuration"""
        config = TrainingConfig()

        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.early_stopping_patience == 10
        assert config.validation_split == 0.1
        assert config.class_weights is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.0001,
            early_stopping_patience=5,
            class_weights=False
        )

        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001
        assert config.class_weights is False
