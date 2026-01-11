#!/usr/bin/env python3
"""Minimal test - direct synthesis without file loading."""

import os
import time
from ensemble import EnsembleAgent
from utils import setup_logging

# Configure for Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

logger = setup_logging("minimal_test")

def main():
    logger.info("="*80)
    logger.info("MINIMAL TEST - Simple Loop Implementation")
    logger.info("="*80)
    
    # Minimal test documents
    documents = [
        "AI is transforming healthcare through improved diagnostics and personalized treatment.",
        "Machine learning models can detect diseases earlier than traditional methods.",
        "AI-powered medical imaging helps radiologists identify anomalies more accurately."
    ]
    
    task = "Summarize how AI is being used in healthcare"
    
    logger.info(f"\nDocuments: {len(documents)}")
    logger.info(f"Task: {task}\n")
    
    # Create agent
    agent = EnsembleAgent(
        model="openai/qwen2.5:7b",
        max_iterations=3,  # Lower for faster test
        timeout_seconds=600  # 10 minutes
    )
    
    logger.info("Starting synthesis...\n")
    start_time = time.time()
    
    try:
        result = agent.synthesize(documents, task)
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("✓ SUCCESS - Simple Loop Completed!")
        logger.info("="*80)
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Iterations: {result['metrics']['num_iterations']}")
        logger.info(f"API Calls: {result['metrics']['num_api_calls']}")
        
        # Show iteration details
        logger.info("\n" + "="*80)
        logger.info("ITERATION DETAILS")
        logger.info("="*80)
        for i, iter_data in enumerate(result['intermediate_outputs']['iteration_history'], 1):
            dec = iter_data['decision']
            logger.info(f"\nIteration {i}:")
            logger.info(f"  Production Ready: {dec['is_production_ready']}")
            logger.info(f"  Reason: {dec['reason']}")
            if dec.get('actionable_improvements'):
                logger.info(f"  Improvements: {len(dec['actionable_improvements'])}")
        
        logger.info(f"\n" + "="*80)
        logger.info(f"Final Output ({len(result['output'])} chars):")
        logger.info("="*80)
        logger.info(result['output'][:300] + "...")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
