"""
Ensemble Agent: Multi-agent system for document synthesis.

This module implements a three-agent ensemble with distinct roles:
- Archivist: Extracts and organizes key information from source documents
- Drafter: Creates initial synthesis based on archivist's organization
- Critic: Reviews and refines the draft for quality and completeness
"""

import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EnsembleAgent:
    """Multi-agent ensemble for document synthesis with specialized roles."""
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the ensemble agent.
        
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
            "num_api_calls": 0,
            "archivist_tokens": 0,
            "drafter_tokens": 0,
            "critic_tokens": 0
        }
    
    def _call_llm(self, system_prompt: str, user_prompt: str, role: str) -> str:
        """
        Make an LLM API call and track metrics.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            role: Name of the agent role (for metric tracking)
            
        Returns:
            LLM response content
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        self.metrics["num_api_calls"] += 1
        if response.usage:
            tokens = response.usage.total_tokens
            self.metrics["prompt_tokens"] += response.usage.prompt_tokens
            self.metrics["completion_tokens"] += response.usage.completion_tokens
            self.metrics["total_tokens"] += tokens
            self.metrics[f"{role}_tokens"] += tokens
        
        return response.choices[0].message.content
    
    def archivist(self, source_documents: List[str], task_description: str) -> str:
        """
        Archivist agent: Extract and organize key information from documents.
        
        Args:
            source_documents: List of source document contents
            task_description: Description of the synthesis task
            
        Returns:
            Structured summary of key information
        """
        documents_text = "\n\n".join([
            f"DOCUMENT {i+1}:\n{doc}" 
            for i, doc in enumerate(source_documents)
        ])
        
        system_prompt = """You are an expert archivist. Your role is to read multiple documents and extract, categorize, and organize the most relevant information for a given task.

Your responsibilities:
- Identify key concepts, facts, and themes from each document
- Organize information by topic and relevance to the task
- Note any contradictions or complementary information across documents
- Create a structured knowledge base for the drafter to use

Output Format:
Provide a clear, organized summary with sections for:
1. Key Topics and Themes
2. Important Facts and Details
3. Cross-document Connections
4. Relevant Context for the Task"""

        user_prompt = f"""Task: {task_description}

Source Documents:
{documents_text}

Please extract and organize the key information from these documents that is relevant to the task."""

        return self._call_llm(system_prompt, user_prompt, "archivist")
    
    def drafter(self, archived_info: str, task_description: str) -> str:
        """
        Drafter agent: Create initial synthesis from organized information.
        
        Args:
            archived_info: Organized information from the archivist
            task_description: Description of the synthesis task
            
        Returns:
            Initial draft of the synthesis
        """
        system_prompt = """You are an expert writer and synthesizer. Your role is to create a comprehensive, well-structured document based on organized information provided by the archivist.

Your responsibilities:
- Transform the archived information into a coherent narrative
- Ensure logical flow and structure
- Address all aspects of the task requirements
- Write clearly and professionally
- Integrate information naturally without simply listing facts

Focus on creating a draft that is complete and well-organized, ready for review."""

        user_prompt = f"""Task: {task_description}

Organized Information from Archivist:
{archived_info}

Please create a comprehensive synthesis that addresses the task using the organized information above."""

        return self._call_llm(system_prompt, user_prompt, "drafter")
    
    def critic(self, draft: str, task_description: str, archived_info: str) -> str:
        """
        Critic agent: Review and refine the draft for quality and completeness.
        
        Args:
            draft: Initial draft from the drafter
            task_description: Description of the synthesis task
            archived_info: Original archived information for reference
            
        Returns:
            Final refined synthesis
        """
        system_prompt = """You are an expert editor and critic. Your role is to review and refine drafts to ensure they meet the highest quality standards.

Your responsibilities:
- Check for completeness against task requirements
- Ensure logical flow and coherence
- Improve clarity and conciseness
- Verify accuracy against source information
- Enhance overall quality and professionalism
- Fix any errors or weaknesses

Provide a refined, polished version of the document."""

        user_prompt = f"""Task: {task_description}

Draft to Review:
{draft}

Original Archived Information (for reference):
{archived_info}

Please review and refine the draft to ensure it fully addresses the task with high quality and completeness."""

        return self._call_llm(system_prompt, user_prompt, "critic")
    
    def synthesize(self, source_documents: List[str], task_description: str) -> Dict[str, Any]:
        """
        Synthesize source documents using the three-agent ensemble.
        
        Args:
            source_documents: List of source document contents
            task_description: Description of the synthesis task
            
        Returns:
            Dictionary containing the final synthesis and metadata
        """
        start_time = time.time()
        
        # Step 1: Archivist extracts and organizes information
        archived_info = self.archivist(source_documents, task_description)
        
        # Step 2: Drafter creates initial synthesis
        draft = self.drafter(archived_info, task_description)
        
        # Step 3: Critic reviews and refines
        final_output = self.critic(draft, task_description, archived_info)
        
        end_time = time.time()
        self.metrics["latency_seconds"] = end_time - start_time
        
        return {
            "output": final_output,
            "intermediate_outputs": {
                "archived_info": archived_info,
                "draft": draft
            },
            "metrics": self.metrics.copy(),
            "model": self.model
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    agent = EnsembleAgent()
    
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
    print("Final Synthesized Output:")
    print(result["output"])
    print("\nMetrics:")
    print(result["metrics"])
