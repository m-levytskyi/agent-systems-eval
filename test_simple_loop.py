#!/usr/bin/env python3
"""Test the new simple loop implementation."""

import os

# Configure CrewAI to use Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key for Ollama

from ensemble import EnsembleAgent
from utils import setup_logging

logger = setup_logging("test_simple_loop")

def main():
    logger.info("="*60)
    logger.info("Testing Simple Loop Implementation (No CrewAI Flow)")
    logger.info("="*60)
    
    agent = EnsembleAgent()
    
    # Use minimal test documents
    test_documents = [
        "Document 1: AI is transforming healthcare with better diagnostics.",
        "Document 2: Machine learning models can detect diseases early.",
        "Document 3: AI systems are being used in radiology and pathology.",
    ]
    
    task = "Summarize the role of AI in healthcare based on the provided documents."
    
    logger.info(f"\nTask: {task}")
    logger.info(f"Documents: {len(test_documents)}")
    
    try:
        result = agent.synthesize(test_documents, task)
        
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULT")
        logger.info("="*60)
        logger.info(f"\nSynthesis:\n{result['output']}")
        logger.info(f"\nIterations: {result['metrics']['num_iterations']}")
        logger.info(f"Total Tokens: {result['metrics']['total_tokens']}")
        logger.info(f"Latency: {result['metrics']['latency_seconds']:.2f}s")
        
        # Print iteration history
        logger.info("\n" + "="*60)
        logger.info("ITERATION HISTORY")
        logger.info("="*60)
        for iteration in result['intermediate_outputs']['iteration_history']:
            logger.info(f"\nIteration {iteration['iteration']}:")
            logger.info(f"  Production Ready: {iteration['decision']['is_production_ready']}")
            logger.info(f"  Reason: {iteration['decision']['reason']}")
            improvements = iteration['decision']['actionable_improvements']
            if improvements:
                logger.info(f"  Improvements: {len(improvements)} items")
        
        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
