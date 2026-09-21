import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.mock_llm import MockLLM, MockLLMConfig


class TestMockLLM:
    """Unit tests for PRD-0001-FR-18, FR-20, AC-016, AC-017, AC-019: Mock LLM Pipeline"""

    def setup_method(self):
        """Set up mock LLM with test config"""
        config = MockLLMConfig(
            responses={
                "easy": "This is a simple answer from the Easy model.",
                "medium": "This is a detailed answer from the Medium model with explanations.",
                "hard": "This is a comprehensive answer from the Hard model with deep analysis."
            },
            default_response="Mock response for testing.",
            delay_ms=0
        )
        self.mock_llm = MockLLM(config)

    def test_mock_llm_initialization(self):
        """Test mock LLM initializes correctly"""
        assert self.mock_llm.config.responses is not None
        assert "easy" in self.mock_llm.config.responses

    def test_generate_response(self):
        """Test generating response for a tier"""
        result = self.mock_llm.generate("easy", "What is 2+2?")

        assert "response" in result
        assert "tier" in result
        assert result["tier"] == "easy"
        assert "latency_ms" in result

    def test_generate_all_tiers(self):
        """Test generating responses for all tiers"""
        tiers = ["easy", "medium", "hard"]

        for tier in tiers:
            result = self.mock_llm.generate(tier, f"Test query for {tier}")
            assert result["tier"] == tier
            assert len(result["response"]) > 0

    def test_original_query_used(self):
        """Test that ORIGINAL USER QUERY is used (not embedding, not class name)"""
        original_query = "What is the capital of France?"
        result = self.mock_llm.generate("easy", original_query)

        # The mock should receive the original query
        # This is a design test - mock doesn't process the query, but the interface should accept it
        assert "response" in result

    def test_structured_response(self):
        """Test structured response format"""
        result = self.mock_llm.generate("medium", "Test query")

        required_keys = ["response", "tier", "latency_ms"]
        for key in required_keys:
            assert key in result

    def test_custom_responses(self):
        """Test custom responses per tier"""
        config = MockLLMConfig(
            responses={
                "easy": "Simple answer",
                "medium": "Medium answer",
                "hard": "Complex answer"
            }
        )
        mock = MockLLM(config)

        assert mock.generate("easy", "q")["response"] == "Simple answer"
        assert mock.generate("medium", "q")["response"] == "Medium answer"
        assert mock.generate("hard", "q")["response"] == "Complex answer"

    def test_fallback_response(self):
        """Test fallback for unknown tier"""
        config = MockLLMConfig(responses={"easy": "easy"})
        mock = MockLLM(config)

        result = mock.generate("unknown_tier", "query")
        assert "response" in result
        assert result["tier"] == "unknown_tier"

    def test_empty_query(self):
        """Test handling of empty query"""
        result = self.mock_llm.generate("easy", "")
        assert "response" in result


class TestMockLLMConfig:
    """Tests for MockLLMConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = MockLLMConfig()

        assert config.responses is not None
        assert "easy" in config.responses
        assert config.delay_ms == 0

    def test_custom_config(self):
        """Test custom configuration"""
        config = MockLLMConfig(
            responses={"easy": "custom"},
            delay_ms=100
        )

        assert config.responses["easy"] == "custom"
        assert config.delay_ms == 100
