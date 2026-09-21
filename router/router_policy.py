from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyConfig:
    """Configuration for Router Policy"""
    confidence_threshold: float = 0.70
    fallback_tier: str = "medium"
    model_registry: dict[str, str] = None

    def __post_init__(self):
        if self.model_registry is None:
            self.model_registry = {
                "easy": "qwen3:1.7b",
                "medium": "qwen3:4b",
                "hard": "llama3.2:3b"
            }


class RouterPolicy:
    """
    Router Policy Engine.
    
    Separates the neural network prediction from LLM selection.
    
    The neural network (Keras MLP) ONLY predicts:
    - Difficulty class (easy/medium/hard)
    - Confidence (max probability)
    
    The Router Policy uses this information to:
    1. Check confidence against threshold
    2. Select appropriate LLM from model registry
    3. Apply fallback if confidence is low
    
    This separation ensures:
    - NN does NOT directly call LLMs
    - NN does NOT generate answers
    - Policy is configurable without retraining
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def select_model(self, predicted_class: str, confidence: float) -> dict[str, Any]:
        """
        Select LLM based on predicted class and confidence.
        
        Args:
            predicted_class: Difficulty class from neural network (easy/medium/hard)
            confidence: Confidence score (max probability from softmax)
            
        Returns:
            Dict with selected_tier, selected_model, fallback_used, confidence
        """
        # Validate predicted class
        valid_tiers = ["easy", "medium", "hard"]
        if predicted_class not in valid_tiers:
            # Invalid tier -> use fallback
            return self._fallback_response(confidence, f"invalid_tier:{predicted_class}")

        # Check confidence threshold
        if confidence >= self.config.confidence_threshold:
            # High confidence -> use predicted tier
            return {
                "selected_tier": predicted_class,
                "selected_model": self.config.model_registry[predicted_class],
                "fallback_used": False,
                "confidence": confidence,
                "reason": "high_confidence"
            }
        else:
            # Low confidence -> use fallback tier
            return self._fallback_response(confidence, "low_confidence")

    def _fallback_response(self, confidence: float, reason: str) -> dict[str, Any]:
        """Generate fallback response"""
        fallback_tier = self.config.fallback_tier
        if fallback_tier not in self.config.model_registry:
            fallback_tier = "medium"  # ultimate fallback

        return {
            "selected_tier": fallback_tier,
            "selected_model": self.config.model_registry[fallback_tier],
            "fallback_used": True,
            "confidence": confidence,
            "reason": reason
        }

    def get_model_for_tier(self, tier: str) -> str | None:
        """Get model name for a given tier"""
        return self.config.model_registry.get(tier)

    def get_all_models(self) -> dict[str, str]:
        """Get all available models"""
        return self.config.model_registry.copy()


def load_policy_config(config_path: str = "config.yaml") -> PolicyConfig:
    """Load policy config from YAML file"""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        return PolicyConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    router_config = config.get('router', {})
    features = config.get('features', {})

    return PolicyConfig(
        confidence_threshold=router_config.get('confidence_threshold', 0.70),
        fallback_tier=router_config.get('fallback_tier', "medium"),
        model_registry={
            "easy": config.get('ollama', {}).get('models', {}).get('easy', "qwen3:1.7b"),
            "medium": config.get('ollama', {}).get('models', {}).get('medium', "qwen3:4b"),
            "hard": config.get('ollama', {}).get('models', {}).get('hard', "llama3.2:3b")
        }
    )


if __name__ == "__main__":
    # Quick test
    policy = RouterPolicy()

    test_cases = [
        ("easy", 0.85),
        ("medium", 0.85),
        ("hard", 0.85),
        ("hard", 0.50),  # Low confidence
        ("invalid", 0.90),
    ]

    print("Router Policy Tests:")
    for tier, conf in test_cases:
        result = policy.select_model(tier, conf)
        print(f"  {tier} @ {conf:.2f} -> {result['selected_tier']} ({result['selected_model']}) fallback={result['fallback_used']}")
