from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from llm.types import LLMResult


class OllamaClient:
    """Minimal Ollama chat client using the local HTTP API.

    Uses POST /api/chat (non-streaming).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> LLMResult:
        url = f"{self.base_url}/api/chat"
        messages: List[Dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        message = (raw.get("message") or {})
        text = (message.get("content") or "")

        prompt_tokens = int(raw.get("prompt_eval_count") or 0)
        completion_tokens = int(raw.get("eval_count") or 0)
        # Note: Ollama doesn't always return a total token count; we approximate.
        total_tokens = prompt_tokens + completion_tokens

        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=raw,
        )
