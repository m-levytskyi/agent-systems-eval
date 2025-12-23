"""
Monolithic Agent: Single LLM approach for document synthesis.

This module implements a straightforward single-agent system that uses one LLM
to read source documents and synthesize them according to task requirements.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rate_limits import RequestRateLimiter

from llm.base import LLMClient
from llm.factory import create_llm_client, get_max_output_tokens

load_dotenv()


class MonolithicAgent:
    """A single LLM agent that performs document synthesis."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
        llm_client: Optional[LLMClient] = None,
        rate_limiter: Optional[RequestRateLimiter] = None,
    ):
        """
        Initialize the monolithic agent.
        
        Args:
            model: Gemini model to use (defaults to env GEMINI_MODEL or gemini-2.5-pro)
            api_key: Gemini API key (defaults to env GEMINI_API_KEY or GOOGLE_API_KEY)
        """
        # Backwards-compatible args: (model, api_key, client) were Gemini-specific.
        # New path: pass llm_client or set LLM_PROVIDER=ollama.
        self.llm_client: LLMClient
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # If a legacy Gemini client is passed, wrap it.
            if client is not None:
                from llm.gemini import GeminiClient

                self.llm_client = GeminiClient(client=client, model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
            else:
                # If caller passes an api_key, prefer it for Gemini.
                provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
                if provider == "gemini" and api_key:
                    from google import genai
                    from llm.gemini import GeminiClient

                    gemini_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
                    self.llm_client = GeminiClient(client=genai.Client(api_key=api_key), model=gemini_model)
                else:
                    # Factory reads env for provider + model.
                    self.llm_client = create_llm_client(provider=provider)

        # Keep `model` attribute for logging.
        self.model = model or os.getenv("OLLAMA_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.rate_limiter = rate_limiter
        self.max_output_tokens = get_max_output_tokens(default=4000)
        self.metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_seconds": 0.0,
            "num_api_calls": 0
        }
    
    def synthesize(self, source_documents: List[str], task_description: str) -> Dict[str, Any]:
        """
        Synthesize source documents according to task description.
        
        Args:
            source_documents: List of source document contents
            task_description: Description of the synthesis task
            
        Returns:
            Dictionary containing the synthesized output and metadata
        """
        start_time = time.time()
        
        # Build the prompt
        documents_text = "\n\n".join([
            f"DOCUMENT {i+1}:\n{doc}" 
            for i, doc in enumerate(source_documents)
        ])
        
        system_prompt = """You are an expert document synthesizer. Your task is to read multiple source documents and create a comprehensive, well-structured synthesis that addresses the given task requirements.

Guidelines:
- Integrate information from all provided documents
- Maintain accuracy and cite key points appropriately
- Create a coherent narrative that flows logically
- Ensure the output directly addresses the task description
- Be concise yet thorough"""

        user_prompt = f"""Task: {task_description}

    Source Documents:
    {documents_text}

    Please synthesize the above documents to complete the task. Provide a well-structured, comprehensive response."""

        # Make LLM call with a lightweight retry to reduce empty responses
        output_text = self._generate_with_retry(system_prompt=system_prompt, user_prompt=user_prompt)
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["latency_seconds"] = end_time - start_time
        
        return {
            "output": output_text,
            "metrics": self.metrics.copy(),
            "model": self.model
        }

    def _generate_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempts: int = 2,
        backoff: float = 1.0,
    ) -> str:
        """Generate content with a short retry loop to mitigate empty responses."""
        last_text = ""
        for attempt in range(1, attempts + 1):
            if self.rate_limiter:
                self.rate_limiter.acquire()

            llm_result = self.llm_client.chat(
                system=system_prompt,
                user=user_prompt,
                temperature=0.7,
                max_tokens=self.max_output_tokens,
            )
            # Count this attempt
            self.metrics["num_api_calls"] += 1

            self.metrics["prompt_tokens"] += int(llm_result.prompt_tokens or 0)
            self.metrics["completion_tokens"] += int(llm_result.completion_tokens or 0)
            self.metrics["total_tokens"] += int(llm_result.total_tokens or 0)

            text = llm_result.text or ""
            if text.strip():
                return text

            print(f"⚠️  Empty response (attempt {attempt}/{attempts})")

            last_text = text
            if attempt < attempts:
                time.sleep(backoff)
        return last_text
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    agent = MonolithicAgent()
    
    # Load sample documents
    doc_dir = os.path.join(os.path.dirname(__file__), "data", "source_documents")
    documents = []
    for filename in sorted(os.listdir(doc_dir)):
        if filename.endswith(".txt") or filename.endswith(".pdf"):
            filepath = os.path.join(doc_dir, filename)
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            # PDF loading will be handled by evaluate.py
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    print("Synthesized Output:")
    print(result["output"])
    print("\nMetrics:")
    print(result["metrics"])
