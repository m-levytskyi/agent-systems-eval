#!/usr/bin/env python3
"""Quick test of ensemble agent with simple loop - no monolithic comparison."""

import os
import time
from ensemble import EnsembleAgent
from utils import setup_logging, load_source_documents

# Configure for Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

logger = setup_logging("test_ensemble")

def main():
    logger.info("="*80)
    logger.info("ENSEMBLE AGENT TEST - Simple Loop Implementation")
    logger.info("="*80)
    
    # Load source documents
    doc_dir = "data/source_documents"
    logger.info(f"\nLoading documents from {doc_dir}...")
    source_documents = load_source_documents(doc_dir, pattern="paper_*.pdf")
    logger.info(f"Loaded {len(source_documents)} documents")
    
    # Task
    task = "Define the academic scope, terminology, and technological context of the provided corpus"
    logger.info(f"\nTask: {task}")
    
    # Create ensemble agent
    logger.info("\nInitializing Ensemble Agent...")
    agent = EnsembleAgent(
        model="openai/qwen2.5:7b",
        max_iterations=5,
        timeout_seconds=1800  # 30 minutes
    )
    
    # Run synthesis
    logger.info("\n" + "="*80)
    logger.info("STARTING SYNTHESIS")
    logger.info("="*80)
    start_time = time.time()
    
    try:
        result = agent.synthesize(source_documents, task)
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("SYNTHESIS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nTotal Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        logger.info(f"Iterations: {result['metrics']['num_iterations']}")
        logger.info(f"API Calls: {result['metrics']['num_api_calls']}")
        logger.info(f"Total Tokens: {result['metrics']['total_tokens']:,}")
        
        # Print iteration history
        logger.info("\n" + "="*80)
        logger.info("ITERATION HISTORY")
        logger.info("="*80)
        for i, iteration in enumerate(result['intermediate_outputs']['iteration_history'], 1):
            decision = iteration['decision']
            logger.info(f"\nIteration {i}:")
            logger.info(f"  Production Ready: {decision['is_production_ready']}")
            logger.info(f"  Reason: {decision['reason']}")
            improvements = decision.get('actionable_improvements', [])
            if improvements:
                logger.info(f"  Improvements Requested: {len(improvements)}")
                for j, imp in enumerate(improvements[:3], 1):  # Show first 3
                    logger.info(f"    {j}. {imp[:100]}...")
        
        # Print final output summary
        logger.info("\n" + "="*80)
        logger.info("FINAL OUTPUT (first 500 chars)")
        logger.info("="*80)
        logger.info(result['output'][:500] + "...")
        
        logger.info("\n" + "="*80)
        logger.info("✓ TEST COMPLETE - Simple Loop Implementation Working!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
