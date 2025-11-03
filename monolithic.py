"""
Monolithic Agent: Single LLM approach for document synthesis.

This module implements a straightforward single-agent system that uses one LLM
to read source documents and synthesize them according to task requirements.
"""

import os
import time
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class MonolithicAgent:
    """A single LLM agent that performs document synthesis."""
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the monolithic agent.
        
        Args:
            model: Gemini model to use (defaults to env GEMINI_MODEL or gemini-2.0-flash-exp)
            api_key: Google API key (defaults to env GOOGLE_API_KEY)
        """
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)
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

        # Make API call
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000
            )
        )
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["latency_seconds"] = end_time - start_time
        self.metrics["num_api_calls"] += 1
        
        # Gemini token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.metrics["prompt_tokens"] += response.usage_metadata.prompt_token_count
            self.metrics["completion_tokens"] += response.usage_metadata.candidates_token_count
            self.metrics["total_tokens"] += response.usage_metadata.total_token_count
        
        return {
            "output": response.text,
            "metrics": self.metrics.copy(),
            "model": self.model
        }
    
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
