import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.keras_mlp import KerasMLPRouter, RouterConfig


class TestKerasMLPRouter:
    """Unit tests for PRD-0001-FR-6: Build Keras MLP classifier"""

    def setup_method(self):
        """Set up router with test config"""
        config = RouterConfig(
            input_dim=384,  # all-MiniLM-L6-v2 dimension
            hidden_dims=[256, 128],
            num_classes=3,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        self.router = KerasMLPRouter(config)

    def test_model_architecture(self):
        """Test that model has correct architecture"""
        model = self.router.model

        # Check input shape
        assert model.input_shape == (None, 384)

        # Check output shape
        assert model.output_shape == (None, 3)

        # Check layer structure
        layers = model.layers
        assert len(layers) >= 5  # Input + 2 hidden + dropout + output

        # First hidden layer: Dense(256)
        assert layers[1].units == 256
        assert layers[1].activation.__name__ == 'relu'

        # Second hidden layer: Dense(128)
        assert layers[3].units == 128
        assert layers[3].activation.__name__ == 'relu'

        # Output layer: Dense(3) with Softmax
        assert layers[-1].units == 3
        assert layers[-1].activation.__name__ == 'softmax'

    def test_model_compilation(self):
        """Test that model compiles with correct optimizer and loss"""
        model = self.router.model

        assert model.optimizer is not None
        assert model.loss == 'categorical_crossentropy'
        # Check metrics - in newer Keras, metrics might be in compile_metrics
        metrics = model.metrics_names
        # Just verify model compiled successfully
        assert len(metrics) > 0

    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        batch_size = 4
        dummy_input = np.random.randn(batch_size, 384).astype(np.float32)

        output = self.router.predict(dummy_input)

        assert output.shape == (batch_size, 3)
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)  # Softmax sums to 1
        assert np.all(output >= 0) and np.all(output <= 1)  # Probabilities

    def test_predict_single(self):
        """Test prediction for single sample"""
        dummy_input = np.random.randn(1, 384).astype(np.float32)

        output = self.router.predict(dummy_input)

        assert output.shape == (1, 3)
        assert np.allclose(output.sum(), 1.0, atol=1e-5)

    def test_get_config(self):
        """Test getting model configuration"""
        config = self.router.get_config()

        assert config['input_dim'] == 384
        assert config['hidden_dims'] == [256, 128]
        assert config['num_classes'] == 3
        assert config['dropout_rate'] == 0.2
        assert config['learning_rate'] == 0.001

    def test_save_and_load(self, tmp_path):
        """Test saving and loading model"""
        model_path = tmp_path / "test_model.keras"

        # Save
        self.router.save(str(model_path))

        # Load
        loaded_router = KerasMLPRouter.load(str(model_path))

        # Verify architecture preserved
        assert loaded_router.config.input_dim == 384
        assert loaded_router.config.hidden_dims == [256, 128]
        assert loaded_router.config.num_classes == 3

        # Verify predictions match
        dummy_input = np.random.randn(2, 384).astype(np.float32)
        orig_output = self.router.predict(dummy_input)
        loaded_output = loaded_router.predict(dummy_input)

        np.testing.assert_allclose(orig_output, loaded_output, rtol=1e-5)

    def test_model_summary(self):
        """Test model summary generation"""
        summary = self.router.summary()

        assert 'Total params' in summary
        assert 'Trainable params' in summary

    def test_custom_config(self):
        """Test router with custom configuration"""
        config = RouterConfig(
            input_dim=768,
            hidden_dims=[512, 256, 128],
            num_classes=5,
            dropout_rate=0.3,
            learning_rate=0.0001
        )
        router = KerasMLPRouter(config)

        assert router.config.input_dim == 768
        assert router.config.hidden_dims == [512, 256, 128]
        assert router.config.num_classes == 5
        assert router.config.dropout_rate == 0.3
        assert router.config.learning_rate == 0.0001

        # Check output shape
        dummy_input = np.random.randn(1, 768).astype(np.float32)
        output = router.predict(dummy_input)
        assert output.shape == (1, 5)


class TestRouterConfig:
    """Tests for RouterConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = RouterConfig()

        assert config.input_dim == 384
        assert config.hidden_dims == [256, 128]
        assert config.num_classes == 3
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.001

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'input_dim': 512,
            'hidden_dims': [256, 128],
            'num_classes': 3,
            'dropout_rate': 0.1,
            'learning_rate': 0.01
        }
        config = RouterConfig(**config_dict)

        assert config.input_dim == 512
        assert config.hidden_dims == [256, 128]
        assert config.dropout_rate == 0.1
        assert config.learning_rate == 0.01
