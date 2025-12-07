"""
Example usage of the monolithic and ensemble agents.

This script demonstrates how to use both agents with your own documents and tasks.
Note: Requires a valid Gemini API key in .env file.
"""

import os
from dotenv import load_dotenv
from monolithic import MonolithicAgent
from ensemble import EnsembleAgent

# Load environment variables
load_dotenv()


def example_monolithic():
    """Example usage of the monolithic agent."""
    print("="*60)
    print("MONOLITHIC AGENT EXAMPLE")
    print("="*60)
    
    # Initialize agent
    agent = MonolithicAgent()
    
    # Sample documents
    documents = [
        """Machine learning is a subset of AI that enables systems to learn from data.
        It includes supervised learning, unsupervised learning, and reinforcement learning.""",
        
        """Deep learning uses neural networks with multiple layers to process complex patterns.
        It has revolutionized computer vision and natural language processing."""
    ]
    
    # Synthesis task
    task = "Write a brief summary of machine learning and deep learning."
    
    # Run synthesis
    print("\nSynthesizing documents...")
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


def example_ensemble():
    """Example usage of the ensemble agent."""
    print("\n" + "="*60)
    print("ENSEMBLE AGENT EXAMPLE")
    print("="*60)
    
    # Initialize agent
    agent = EnsembleAgent()
    
    # Sample documents
    documents = [
        """Machine learning is a subset of AI that enables systems to learn from data.
        It includes supervised learning, unsupervised learning, and reinforcement learning.""",
        
        """Deep learning uses neural networks with multiple layers to process complex patterns.
        It has revolutionized computer vision and natural language processing."""
    ]
    
    # Synthesis task
    task = "Write a brief summary of machine learning and deep learning."
    
    # Run synthesis
    print("\nSynthesizing documents with ensemble (3 agents)...")
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
    """Check if Google API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("⚠️  WARNING: Google API key not configured!")
        print("Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file")
        print("Example:")
        print("  1. Copy .env.example to .env")
        print("  2. Edit .env and add your API key")
        return False
    return True


def main():
    """Run example demonstrations."""
    print("="*60)
    print("Agent Systems Evaluation - Usage Examples")
    print("="*60)
    print()
    
    if not check_api_key():
        print("\nSkipping examples (no API key configured)")
        return
    
    try:
        # Run monolithic example
        example_monolithic()
        
        # Run ensemble example
        example_ensemble()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("  1. Valid Google Gemini API key in .env")
        print("  2. Installed all requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
