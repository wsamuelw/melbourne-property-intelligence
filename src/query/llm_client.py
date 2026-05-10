"""LLM client for text generation.

Supports Claude API (Anthropic) and local Ollama as backends.
Falls back to a simple template-based response if neither is available.
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a text response from a prompt."""
        ...


class ClaudeClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaClient(LLMClient):
    """Local Ollama client."""

    def __init__(self, base_url: str | None = None, model: str = "llama3"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


class FallbackClient(LLMClient):
    """Template-based fallback when no LLM is available."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return (
            "This is a template response. To get AI-powered insights, "
            "set ANTHROPIC_API_KEY or install and run Ollama locally."
        )


def get_llm_client() -> LLMClient:
    """Get the best available LLM client.

    Priority: Claude API > Ollama > Fallback.
    """
    import os

    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Using Claude API")
        return ClaudeClient()

    try:
        client = OllamaClient()
        # Quick health check
        import httpx

        httpx.get(f"{client.base_url}/api/tags", timeout=5)
        logger.info("Using Ollama at %s", client.base_url)
        return client
    except Exception:
        logger.info("No LLM available — using fallback client")
        return FallbackClient()
