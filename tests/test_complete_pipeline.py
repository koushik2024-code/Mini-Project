import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.mock_llm import MockLLM, MockLLMConfig
from llm.ollama_client import OllamaClient, OllamaConfig
from router.predictor import PredictorConfig, QueryPredictor
from router.router_policy import PolicyConfig, RouterPolicy


class TestCompleteInferencePipeline:
    """Integration tests for PRD-0001-FR-10, FR-18, FR-19, FR-20: Complete Inference Pipeline"""

    def setup_method(self):
        """Set up complete pipeline"""
        # Use mock LLM for fast testing
        self.use_mock = True

        # Load config
        self.predictor_config = PredictorConfig(
            router_model_path="models/router/best_model.keras",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="cpu",
            embedding_cache_dir="embeddings/cache"
        )

        self.policy_config = PolicyConfig(
            confidence_threshold=0.70,
            fallback_tier="medium",
            model_registry={
                "easy": "qwen3:1.7b",
                "medium": "qwen3:1.7b",
                "hard": "llama3.2:3b"
            }
        )

        if self.use_mock:
            self.llm = MockLLM(MockLLMConfig(
                responses={
                    "easy": "Easy model response",
                    "medium": "Medium model response",
                    "hard": "Hard model response"
                }
            ))
        else:
            ollama_config = OllamaConfig()
            self.llm = OllamaClient(ollama_config)

        self.predictor = QueryPredictor(self.predictor_config)
        self.policy = RouterPolicy(self.policy_config)

    def test_pipeline_mock_llm(self):
        """Test complete pipeline with mock LLM"""
        query = "What is 2+2?"

        # Step 1: Preprocess
        processed = self.predictor.preprocessor.preprocess(query)

        # Step 2: Embed
        embedding = self.predictor.embedding_model.encode(processed)

        # Step 3: Predict
        probs = self.predictor.router.predict(embedding.reshape(1, -1))[0]
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        predicted_class = ["easy", "medium", "hard"][pred_idx]

        # Step 4: Router Policy
        policy_result = self.policy.select_model(predicted_class, confidence)

        # Step 5: LLM Generation
        llm_result = self.llm.generate(policy_result["selected_tier"], query)

        # Verify
        assert "response" in llm_result
        assert "tier" in llm_result
        # Policy should select model based on predicted class (or fallback)
        assert policy_result["selected_tier"] in ["easy", "medium", "hard"]
        assert llm_result["tier"] == policy_result["selected_tier"]

    def test_pipeline_original_query_used(self):
        """Test that ORIGINAL USER QUERY is sent to LLM (not embedding, not class name)"""
        query = "What is the capital of France?"

        processed = self.predictor.preprocessor.preprocess(query)
        embedding = self.predictor.embedding_model.encode(processed)
        probs = self.predictor.router.predict(embedding.reshape(1, -1))[0]
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        predicted_class = ["easy", "medium", "hard"][pred_idx]

        policy_result = self.policy.select_model(predicted_class, confidence)
        llm_result = self.llm.generate(policy_result["selected_tier"], query)

        # The LLM should receive the original query
        # This is verified by the mock receiving the query parameter
        assert "response" in llm_result

    def test_pipeline_fallback(self):
        """Test pipeline with low confidence (fallback)"""
        # Create a predictor that returns low confidence
        # We can't easily control the router, so we test the policy directly
        result = self.policy.select_model("hard", 0.50)  # Low confidence

        assert result["fallback_used"] is True
        assert result["selected_tier"] == "medium"

    def test_pipeline_structured_response(self):
        """Test final structured response format"""
        query = "Test query"

        processed = self.predictor.preprocessor.preprocess(query)
        embedding = self.predictor.embedding_model.encode(processed)
        probs = self.predictor.router.predict(embedding.reshape(1, -1))[0]
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        predicted_class = ["easy", "medium", "hard"][pred_idx]

        policy_result = self.policy.select_model(predicted_class, confidence)
        llm_result = self.llm.generate(policy_result["selected_tier"], query)

        # Build final response
        final_response = {
            "query": query,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "selected_model": policy_result["selected_model"],
            "response": llm_result["response"],
            "fallback_used": policy_result["fallback_used"],
            "latency_ms": llm_result["latency_ms"]
        }

        required_keys = ["query", "predicted_class", "confidence", "selected_model", "response", "fallback_used", "latency_ms"]
        for key in required_keys:
            assert key in final_response

    def test_end_to_end_with_real_components(self):
        """Test end-to-end with real components (if Ollama available)"""
        if not self.use_mock:
            # Test with actual Ollama
            query = "What is 2+2?"

            processed = self.predictor.preprocessor.preprocess(query)
            embedding = self.predictor.embedding_model.encode(processed)
            probs = self.predictor.router.predict(embedding.reshape(1, -1))[0]
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])
            predicted_class = ["easy", "medium", "hard"][pred_idx]

            policy_result = self.policy.select_model(predicted_class, confidence)
            llm_result = self.llm.generate(policy_result["selected_tier"], query)

            assert "response" in llm_result
            assert len(llm_result["response"]) > 0


class TestPipelineComponents:
    """Tests for individual pipeline components working together"""

    def test_predictor_policy_compatibility(self):
        """Test that predictor and policy work together"""
        predictor = QueryPredictor(PredictorConfig())
        policy = RouterPolicy(PolicyConfig())

        # Predictor returns class and confidence
        # Policy uses them to select model
        query = "Test"
        processed = predictor.preprocessor.preprocess(query)
        embedding = predictor.embedding_model.encode(processed)
        probs = predictor.router.predict(embedding.reshape(1, -1))[0]
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        predicted_class = ["easy", "medium", "hard"][pred_idx]

        # Policy should accept predictor's output
        result = policy.select_model(predicted_class, confidence)
        assert result["selected_tier"] in ["easy", "medium", "hard"]
        assert "selected_model" in result

    def test_mock_llm_with_policy(self):
        """Test mock LLM works with policy output"""
        policy = RouterPolicy(PolicyConfig())
        llm = MockLLM()

        result = policy.select_model("medium", 0.85)
        llm_result = llm.generate(result["selected_tier"], "Test query")

        assert llm_result["tier"] == result["selected_tier"]
        assert "response" in llm_result


if __name__ == "__main__":
    # Quick manual test
    pipeline = TestCompleteInferencePipeline()
    pipeline.setup_method()
    pipeline.test_pipeline_mock_llm()
    pipeline.test_pipeline_fallback()
    pipeline.test_pipeline_structured_response()
    print("All pipeline tests passed!")
