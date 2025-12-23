from __future__ import annotations

from typing import Protocol

from llm.types import LLMResult


class LLMClient(Protocol):
    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> LLMResult:
        ...
