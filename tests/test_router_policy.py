import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.router_policy import PolicyConfig, RouterPolicy


class TestRouterPolicy:
    """Unit tests for PRD-0001-FR-15, FR-16, FR-21, FR-22, FR-23: Router Policy Engine"""

    def setup_method(self):
        """Set up policy with test config"""
        config = PolicyConfig(
            confidence_threshold=0.70,
            fallback_tier="medium",
            model_registry={
                "easy": "qwen3:1.7b",
                "medium": "qwen3:4b",
                "hard": "llama3.2:3b"
            }
        )
        self.policy = RouterPolicy(config)

    def test_policy_initialization(self):
        """Test policy initializes correctly"""
        assert self.policy.config.confidence_threshold == 0.70
        assert self.policy.config.fallback_tier == "medium"
        assert "easy" in self.policy.config.model_registry
        assert "medium" in self.policy.config.model_registry
        assert "hard" in self.policy.config.model_registry

    def test_high_confidence_uses_predicted(self):
        """Test high confidence uses predicted tier"""
        result = self.policy.select_model("medium", 0.85)

        assert result['selected_tier'] == "medium"
        assert result['selected_model'] == "qwen3:4b"
        assert result['fallback_used'] is False
        assert result['confidence'] == 0.85

    def test_low_confidence_uses_fallback(self):
        """Test low confidence uses fallback tier"""
        result = self.policy.select_model("hard", 0.50)

        assert result['selected_tier'] == "medium"  # fallback
        assert result['selected_model'] == "qwen3:4b"
        assert result['fallback_used'] is True
        assert result['confidence'] == 0.50

    def test_exact_threshold_uses_predicted(self):
        """Test exact threshold uses predicted tier"""
        result = self.policy.select_model("easy", 0.70)

        assert result['selected_tier'] == "easy"
        assert result['selected_model'] == "qwen3:1.7b"
        assert result['fallback_used'] is False

    def test_all_tiers_mapping(self):
        """Test all tier to model mappings"""
        test_cases = [
            ("easy", 0.80, "easy", "qwen3:1.7b"),
            ("medium", 0.80, "medium", "qwen3:4b"),
            ("hard", 0.80, "hard", "llama3.2:3b"),
        ]

        for tier, conf, expected_tier, expected_model in test_cases:
            result = self.policy.select_model(tier, conf)
            assert result['selected_tier'] == expected_tier
            assert result['selected_model'] == expected_model
            assert result['fallback_used'] is False

    def test_fallback_tier_configurable(self):
        """Test fallback tier is configurable"""
        config = self.policy.config
        config.fallback_tier = "easy"
        policy = RouterPolicy(config)

        result = policy.select_model("hard", 0.50)
        assert result['selected_tier'] == "easy"
        assert result['selected_model'] == "qwen3:1.7b"

    def test_confidence_threshold_configurable(self):
        """Test confidence threshold is configurable"""
        config = self.policy.config
        config.confidence_threshold = 0.50
        policy = RouterPolicy(config)

        result = policy.select_model("hard", 0.60)
        assert result['selected_tier'] == "hard"
        assert result['fallback_used'] is False

    def test_invalid_tier_uses_fallback(self):
        """Test invalid tier defaults to fallback"""
        result = self.policy.select_model("invalid_tier", 0.80)

        assert result['selected_tier'] == "medium"
        assert result['fallback_used'] is True

    def test_response_structure(self):
        """Test response has correct structure"""
        result = self.policy.select_model("medium", 0.85)

        required_keys = ['selected_tier', 'selected_model', 'fallback_used', 'confidence']
        for key in required_keys:
            assert key in result

    def test_model_registry_separate_from_nn(self):
        """Test Router Policy is SEPARATE from neural network"""
        # The policy should only use the predicted class, not the raw logits
        # This is a design test - the policy takes class + confidence, not logits
        result = self.policy.select_model("medium", 0.85)
        assert 'selected_model' in result
        # No logits or raw probabilities in output


class TestPolicyConfig:
    """Tests for PolicyConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PolicyConfig()

        assert config.confidence_threshold == 0.70
        assert config.fallback_tier == "medium"
        assert config.model_registry == {
            "easy": "qwen3:1.7b",
            "medium": "qwen3:4b",
            "hard": "llama3.2:3b"
        }

    def test_custom_config(self):
        """Test custom configuration"""
        config = PolicyConfig(
            confidence_threshold=0.60,
            fallback_tier="easy",
            model_registry={"easy": "model-a", "medium": "model-b", "hard": "model-c"}
        )

        assert config.confidence_threshold == 0.60
        assert config.fallback_tier == "easy"
