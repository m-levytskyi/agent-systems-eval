"""
Monolithic Agent: Single LLM approach for document synthesis.

This module implements a straightforward single-agent system that uses one LLM
to read source documents and synthesize them according to task requirements.
"""

import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MonolithicAgent:
    """A single LLM agent that performs document synthesis."""
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the monolithic agent.
        
        Args:
            model: OpenAI model to use (defaults to env OPENAI_MODEL or gpt-4)
            api_key: OpenAI API key (defaults to env OPENAI_API_KEY)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
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

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["latency_seconds"] = end_time - start_time
        self.metrics["num_api_calls"] += 1
        if response.usage:
            self.metrics["prompt_tokens"] += response.usage.prompt_tokens
            self.metrics["completion_tokens"] += response.usage.completion_tokens
            self.metrics["total_tokens"] += response.usage.total_tokens
        
        return {
            "output": response.choices[0].message.content,
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
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), "r") as f:
                documents.append(f.read())
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    print("Synthesized Output:")
    print(result["output"])
    print("\nMetrics:")
    print(result["metrics"])
