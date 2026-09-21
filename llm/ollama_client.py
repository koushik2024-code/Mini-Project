import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class OllamaConfig:
    """Configuration for Ollama client"""
    host: str = "http://localhost:11434"
    timeout: int = 180
    models: dict[str, str] = field(default_factory=lambda: {
        "easy": "qwen3:1.7b",
        "medium": "qwen3:1.7b",  # Use 1.7b for medium too (4b too slow on CPU)
        "hard": "llama3.2:3b"
    })


class OllamaClient:
    """
    Client for Ollama local LLM inference.
    
    Connects to Ollama running locally and provides chat completion
    for the three tiers:
    - Easy -> qwen3:1.7b
    - Medium -> qwen3:4b
    - Hard -> llama3.2:3b
    
    All models are pre-downloaded via Ollama and run locally.
    """

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self.base_url = self.config.host.rstrip("/")
        self.timeout = httpx.Timeout(self.config.timeout)

    def _get_model_name(self, tier: str) -> str:
        """Get model name for tier"""
        return self.config.models.get(tier, self.config.models.get("medium"))

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models in Ollama"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            return [{"error": str(e)}]

    def health_check(self) -> bool:
        """Check if Ollama is healthy and reachable"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        tier: str,
        query: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9
    ) -> dict[str, Any]:
        """
        Generate chat completion using the specified tier model.
        
        Args:
            tier: Difficulty tier (easy/medium/hard)
            query: Original user query (NOT embedding, NOT class name)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Returns:
            Dict with response, tier, latency_ms, and optional error
        """
        start_time = time.time()

        model_name = self._get_model_name(tier)

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p
            },
            "think": False,
            "stream": False
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            # Thinking models (e.g. qwen3) can still return their reasoning
            # in a separate 'thinking' field with an empty 'content' if the
            # response was truncated before the final answer - fall back to
            # it rather than showing nothing.
            response_text = message.get("content", "") or message.get("thinking", "")

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "response": response_text,
                "tier": tier,
                "model": model_name,
                "latency_ms": latency_ms
            }

        except httpx.TimeoutException:
            return {
                "response": "",
                "tier": tier,
                "model": model_name,
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": "Request timeout"
            }
        except httpx.ConnectError:
            return {
                "response": "",
                "tier": tier,
                "model": model_name,
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": "Cannot connect to Ollama. Is it running?"
            }
        except Exception as e:
            return {
                "response": "",
                "tier": tier,
                "model": model_name,
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }

    def get_model_info(self, tier: str) -> dict[str, Any]:
        """Get information about a specific model"""
        model_name = self._get_model_name(tier)

        try:
            response = httpx.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_available_tiers(self) -> list[str]:
        """Get list of available tiers"""
        return list(self.config.models.keys())


def load_ollama_config(config_path: str = "config.yaml") -> OllamaConfig:
    """Load Ollama config from YAML file"""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        return OllamaConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    ollama_config = config.get('ollama', {})
    return OllamaConfig(
        host=ollama_config.get('host', "http://localhost:11434"),
        timeout=ollama_config.get('timeout', 30),
        models=ollama_config.get('models', {
            "easy": "qwen3:1.7b",
            "medium": "qwen3:4b",
            "hard": "llama3.2:3b"
        })
    )


if __name__ == "__main__":
    # Quick test
    client = OllamaClient()

    print("Ollama Health:", client.health_check())

    models = client.list_models()
    print("Available Models:")
    for m in models[:5]:
        print(f"  {m.get('name', 'unknown')}")

    # Test chat
    if client.health_check():
        result = client.chat("easy", "What is 2+2?")
        print(f"\nEasy tier response: {result['response'][:100]}...")
        print(f"Latency: {result['latency_ms']}ms")
