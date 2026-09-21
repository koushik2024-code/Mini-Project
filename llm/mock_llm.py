import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockLLMConfig:
    """Configuration for Mock LLM"""
    responses: dict[str, str] = field(default_factory=lambda: {
        "easy": "This is a simple, direct answer from the Easy model (Qwen 1.7B equivalent).",
        "medium": "This is a detailed, well-structured answer from the Medium model (Qwen 4B equivalent) with examples and explanations.",
        "hard": "This is a comprehensive, in-depth answer from the Hard model (Llama 3.2 3B equivalent) with thorough analysis, edge cases, and advanced concepts."
    })
    default_response: str = "Mock response for testing."
    delay_ms: int = 0


class MockLLM:
    """
    Mock LLM for testing the complete pipeline before real model integration.
    
    Simulates the three tiers:
    - Easy -> Qwen 1.7B equivalent
    - Medium -> Qwen 4B equivalent  
    - Hard -> Llama 3.2 3B equivalent
    
    The mock returns predefined responses for each tier to test the complete
    pipeline: Query -> Embedding -> Router -> Policy -> Mock LLM -> Response
    """

    def __init__(self, config: MockLLMConfig | None = None):
        self.config = config or MockLLMConfig()

    def generate(self, tier: str, query: str) -> dict[str, Any]:
        """
        Generate a mock response for the given tier and query.
        
        Args:
            tier: Difficulty tier (easy/medium/hard)
            query: Original user query (NOT embedding, NOT class name)
            
        Returns:
            Dict with response, tier, latency_ms
        """
        start_time = time.time()

        # Simulate delay if configured
        if self.config.delay_ms > 0:
            time.sleep(self.config.delay_ms / 1000.0)

        # Get response for tier
        response = self.config.responses.get(tier, self.config.default_response)

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "response": response,
            "tier": tier,
            "latency_ms": latency_ms
        }

    def generate_batch(self, requests: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Generate responses for multiple requests"""
        return [self.generate(req["tier"], req["query"]) for req in requests]

    def get_available_tiers(self) -> list[str]:
        """Get list of available tiers"""
        return list(self.config.responses.keys())


def create_mock_llm_from_config(config_path: str = "config.yaml") -> MockLLM:
    """Create MockLLM from config file"""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        return MockLLM()

    with open(path) as f:
        config = yaml.safe_load(f)

    # Check if mock LLM is enabled
    features = config.get('features', {})
    if not features.get('use_mock_llms', True):
        return None

    return MockLLM()


if __name__ == "__main__":
    # Quick test
    mock = MockLLM()

    test_cases = [
        ("easy", "What is 2+2?"),
        ("medium", "Explain binary search"),
        ("hard", "Write a distributed consensus algorithm"),
    ]

    print("Mock LLM Tests:")
    for tier, query in test_cases:
        result = mock.generate(tier, query)
        print(f"  {tier}: {result['response'][:50]}... (latency: {result['latency_ms']}ms)")
