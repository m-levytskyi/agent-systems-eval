from __future__ import annotations

import os
from typing import Optional

from llm.base import LLMClient
from llm.ollama import OllamaClient


def create_llm_client(*, provider: Optional[str] = None) -> LLMClient:
    """Create an LLM client based on environment configuration.

    Env:
      - LLM_PROVIDER: 'ollama' (local) or 'gemini'
      - OLLAMA_BASE_URL, OLLAMA_MODEL
      - GEMINI_API_KEY / GOOGLE_API_KEY, GEMINI_MODEL
      - MAX_OUTPUT_TOKENS or GEMINI_MAX_OUTPUT_TOKENS
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if provider == "ollama":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        )

    if provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER={provider!r} (expected 'ollama' or 'gemini')")

    # Gemini
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini provider")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    client = genai.Client(api_key=api_key)

    from llm.gemini import GeminiClient

    return GeminiClient(client=client, model=model)


def get_max_output_tokens(default: int = 4000) -> int:
    raw = os.getenv("MAX_OUTPUT_TOKENS") or os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default
