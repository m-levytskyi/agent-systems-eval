from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm.types import LLMResult


@dataclass
class GeminiClient:
    """Thin wrapper around google-genai to match the LLMClient interface."""

    client: "object"  # genai.Client
    model: str

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> LLMResult:
        # Local import so Ollama-only setups don't choke at import-time.
        from google.genai import types

        full_prompt = f"{system}\n\n{user}" if system.strip() else user

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if usage:
            prompt_tokens = int(getattr(usage, "input_tokens", getattr(usage, "prompt_token_count", 0)) or 0)
            completion_tokens = int(getattr(usage, "output_tokens", getattr(usage, "candidates_token_count", 0)) or 0)
            total_tokens = int(getattr(usage, "total_tokens", getattr(usage, "total_token_count", prompt_tokens + completion_tokens)) or 0)

        return LLMResult(
            text=(response.text or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=response,
        )
