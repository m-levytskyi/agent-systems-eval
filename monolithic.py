"""
Monolithic Agent: Single LLM approach for document synthesis.

This module implements a straightforward single-agent system that uses one LLM
to read source documents and synthesize them according to task requirements.
"""

import os
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rate_limits import RequestRateLimiter

load_dotenv()


class MonolithicAgent:
    """A single LLM agent that performs document synthesis."""
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        client: Optional[genai.Client] = None,
        rate_limiter: Optional[RequestRateLimiter] = None,
    ):
        """
        Initialize the monolithic agent.
        
        Args:
            model: Gemini model to use (defaults to env GEMINI_MODEL or gemini-2.5-pro)
            api_key: Gemini API key (defaults to env GEMINI_API_KEY or GOOGLE_API_KEY)
        """
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = client or genai.Client(api_key=api_key)
        self.rate_limiter = rate_limiter
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4000"))
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

        full_prompt = f"""{system_prompt}

Task: {task_description}

Source Documents:
{documents_text}

Please synthesize the above documents to complete the task. Provide a well-structured, comprehensive response."""

        # Make API call with a lightweight retry to reduce empty responses
        response, output_text = self._generate_with_retry(full_prompt)

        output_text = response.text or ""
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["latency_seconds"] = end_time - start_time
        
        # Gemini token usage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_token_count", 0)) or 0
            completion_tokens = getattr(usage, "output_tokens", getattr(usage, "candidates_token_count", 0)) or 0
            total_tokens = getattr(usage, "total_tokens", getattr(usage, "total_token_count", prompt_tokens + completion_tokens)) or 0
            self.metrics["prompt_tokens"] += prompt_tokens
            self.metrics["completion_tokens"] += completion_tokens
            self.metrics["total_tokens"] += total_tokens
        
        return {
            "output": output_text,
            "metrics": self.metrics.copy(),
            "model": self.model
        }

    def _generate_with_retry(self, contents: str, attempts: int = 2, backoff: float = 1.0):
        """Generate content with a short retry loop to mitigate empty responses."""
        last_response = None
        for attempt in range(1, attempts + 1):
            if self.rate_limiter:
                self.rate_limiter.acquire()

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=self.max_output_tokens
                )
            )
            # Count this attempt
            self.metrics["num_api_calls"] += 1

            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_token_count", 0)) or 0
                completion_tokens = getattr(usage, "output_tokens", getattr(usage, "candidates_token_count", 0)) or 0
                total_tokens = getattr(usage, "total_tokens", getattr(usage, "total_token_count", prompt_tokens + completion_tokens)) or 0
                self.metrics["prompt_tokens"] += prompt_tokens
                self.metrics["completion_tokens"] += completion_tokens
                self.metrics["total_tokens"] += total_tokens

            text = response.text or ""
            if text.strip():
                return response, text

            finish_reason = None
            if getattr(response, "candidates", None):
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
            feedback = getattr(response, "prompt_feedback", None)
            print(f"⚠️  Empty response (attempt {attempt}/{attempts}); finish_reason={finish_reason}, feedback={feedback}")

            last_response = response
            if attempt < attempts:
                time.sleep(backoff)
        return last_response or response, text
    
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
                with open(filepath, "r") as f:
                    documents.append(f.read())
            # PDF loading will be handled by evaluate.py
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    print("Synthesized Output:")
    print(result["output"])
    print("\nMetrics:")
    print(result["metrics"])
