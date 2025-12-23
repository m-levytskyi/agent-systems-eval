"""
Example usage of the monolithic and ensemble agents.

This script demonstrates how to use both agents with your own documents and tasks.
Note: Defaults to local Ollama. Set LLM_PROVIDER=gemini to use Gemini.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from monolithic import MonolithicAgent
from ensemble import EnsembleAgent

# Load environment variables
load_dotenv()


def load_source_documents(doc_dir: str, pattern: str = "doc*.pdf") -> List[str]:
    """Load source documents (PDF or text) from the specified directory.
    
    Args:
        doc_dir: Directory containing source documents
        pattern: Glob pattern for filtering files (default: "doc*.pdf" for examples)
    """
    documents = []
    doc_path = Path(doc_dir)
    
    # Load PDF files matching the pattern
    for filepath in sorted(doc_path.glob(pattern)):
        if filepath.suffix.lower() == '.pdf':
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append(text.strip())
        elif filepath.suffix.lower() == '.txt':
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())
    
    return documents


def load_tasks(task_file: str) -> List[Dict[str, Any]]:
    """Load synthesis tasks from JSON file."""
    with open(task_file, "r", encoding="utf-8") as f:
        return json.load(f)


def example_monolithic(documents: List[str], task: str):
    """Example usage of the monolithic agent.
    
    Args:
        documents: List of source documents to synthesize
        task: Task description for synthesis
    """
    print("="*60)
    print("MONOLITHIC AGENT EXAMPLE")
    print("="*60)
    
    # Initialize agent
    agent = MonolithicAgent()
    
    # Run synthesis
    print(f"\nSynthesizing {len(documents)} documents...")
    print(f"Task: {task[:80]}..." if len(task) > 80 else f"Task: {task}")
    result = agent.synthesize(documents, task)
    
    # Display results
    print("\n" + "-"*60)
    print("OUTPUT:")
    print("-"*60)
    print(result["output"])
    print("\n" + "-"*60)
    print("METRICS:")
    print("-"*60)
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")


def example_ensemble(documents: List[str], task: str):
    """Example usage of the ensemble agent.
    
    Args:
        documents: List of source documents to synthesize
        task: Task description for synthesis
    """
    print("\n" + "="*60)
    print("ENSEMBLE AGENT EXAMPLE")
    print("="*60)
    
    # Initialize agent
    agent = EnsembleAgent()
    
    # Run synthesis
    print(f"\nSynthesizing {len(documents)} documents with ensemble (3 agents)...")
    print(f"Task: {task[:80]}..." if len(task) > 80 else f"Task: {task}")
    result = agent.synthesize(documents, task)
    
    # Display results
    print("\n" + "-"*60)
    print("ARCHIVIST OUTPUT:")
    print("-"*60)
    print(result["intermediate_outputs"]["archived_info"][:300] + "...")
    
    print("\n" + "-"*60)
    print("DRAFTER OUTPUT:")
    print("-"*60)
    print(result["intermediate_outputs"]["draft"][:300] + "...")
    
    print("\n" + "-"*60)
    print("CRITIC (FINAL) OUTPUT:")
    print("-"*60)
    print(result["output"])
    
    print("\n" + "-"*60)
    print("METRICS:")
    print("-"*60)
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")


def check_api_key():
    """Validate required configuration for the selected provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider != "gemini":
        return True

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("⚠️  WARNING: Gemini API key not configured!")
        print("Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file")
        return False
    return True


def main():
    """Run example demonstrations."""
    print("="*60)
    print("Agent Systems Evaluation - Usage Examples")
    print("="*60)
    print()
    
    if not check_api_key():
        print("\nSkipping examples (missing required configuration)")
        return
    
    # Configuration for test/example runs
    doc_dir = "data/source_documents"
    task_file = "data/tasks/example_tasks.json"
    doc_pattern = "doc*.pdf"  # Use doc*.pdf for examples/testing
    
    try:
        # Load documents and tasks
        print("\nLoading example documents and tasks...")
        documents = load_source_documents(doc_dir, pattern=doc_pattern)
        tasks = load_tasks(task_file)
        
        print(f"Loaded {len(documents)} example documents")
        print(f"Loaded {len(tasks)} example tasks")
        
        # Run examples with first task
        if tasks:
            first_task = tasks[0]
            task_description = first_task["task_description"]
            
            # Run monolithic example
            example_monolithic(documents, task_description)
            
            # Run ensemble example
            example_ensemble(documents, task_description)
        else:
            print("\n⚠️  No tasks found in example_tasks.json")
            return
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("  1. Valid configuration in .env (or using Ollama locally)")
        print("  2. Installed all requirements: pip install -r requirements.txt")
        print("  3. Example documents (doc*.pdf) in data/source_documents/")
        print("  4. Example tasks in data/tasks/example_tasks.json")


if __name__ == "__main__":
    main()
