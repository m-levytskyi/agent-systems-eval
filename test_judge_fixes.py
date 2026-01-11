#!/usr/bin/env python3
"""Quick test to verify judge scoring fixes work correctly.

Tests with minimal data (3 short docs, 1 simple task) to quickly validate:
1. Judges return non-zero scores
2. BERTScore CPU fallback works
3. ROUGE reference/hypothesis are correct
4. Debug logs show proper data flow
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble import EnsembleAgent
from utils import setup_logging

load_dotenv()

logger = setup_logging("test_judge_fixes")

# Minimal test documents
TEST_DOCUMENTS = [
    "AGDebugger is a tool for debugging multi-agent AI systems. It allows developers to modify agent messages interactively. The system helps identify issues in agent conversations.",
    
    "Companion chatbots can help reduce loneliness. Seven user profiles were identified ranging from well-adjusted users to those seeking emotional support. Ethical considerations are important.",
    
    "CHOIR is a document management system built with Slack Bolt and GitHub API. It provides real-time assistance for document updates and enhances collaboration among team members."
]

# Simple test task
TEST_TASK = "Summarize the key technologies and tools mentioned across the documents. List the main tool names and their purposes in 2-3 sentences."

def main():
    logger.info("="*60)
    logger.info("JUDGE SCORING FIX TEST")
    logger.info("="*60)
    
    # Initialize ensemble agent
    logger.info("\nInitializing ensemble agent...")
    agent = EnsembleAgent(
        max_iterations=2,  # Keep it short for testing
        timeout_seconds=300  # 5 minutes max
    )
    
    # Run synthesis
    logger.info(f"\nRunning synthesis with {len(TEST_DOCUMENTS)} documents...")
    logger.info(f"Task: {TEST_TASK}")
    
    result = agent.synthesize(TEST_DOCUMENTS, TEST_TASK)
    
    # Check result
    output = result.get("output", "")
    metrics = result.get("metrics", {})
    
    logger.info("\n" + "="*60)
    logger.info("SYNTHESIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Output length: {len(output)} chars")
    logger.info(f"Iterations: {metrics.get('num_iterations', 'N/A')}")
    logger.info(f"Production ready: {result.get('production_ready', 'N/A')}")
    logger.info(f"\nOutput preview:\n{output[:500]}...")
    
    # Now test judge evaluation
    logger.info("\n" + "="*60)
    logger.info("TESTING JUDGE EVALUATION")
    logger.info("="*60)
    
    from evaluate import evaluate_with_mlflow_judges
    
    # Prepare context from intermediate outputs
    context_text = ""
    if "intermediate_outputs" in result and "archivist_output" in result["intermediate_outputs"]:
        context_text = result["intermediate_outputs"]["archivist_output"]
        logger.info(f"Using archivist output as context ({len(context_text)} chars)")
    else:
        context_text = "\n\n".join(TEST_DOCUMENTS)
        logger.info(f"Using concatenated documents as context ({len(context_text)} chars)")
    
    # Run judges with Gemini (MLflow format: provider:/model-name)
    judge_result = evaluate_with_mlflow_judges(
        task_description=TEST_TASK,
        synthesis=output,
        context=context_text,
        judge_model="gemini:/gemini-2.5-flash-lite"
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("JUDGE RESULTS")
    logger.info("="*60)
    
    scores = judge_result.get("scores", {})
    feedback = judge_result.get("detailed", {})
    
    for judge_name, score in scores.items():
        logger.info(f"\n{judge_name.upper()}: {score}")
        if judge_name in feedback:
            logger.info(f"Feedback: {str(feedback[judge_name])[:200]}...")
    
    # Validate results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION")
    logger.info("="*60)
    
    all_zeros = all(score == 0.0 for score in scores.values())
    if all_zeros:
        logger.error("❌ FAILED: All judge scores are 0.0 - issue NOT fixed!")
        return 1
    else:
        logger.info(f"✅ SUCCESS: At least one non-zero score found")
    
    # Test NLP metrics
    logger.info("\n" + "="*60)
    logger.info("TESTING NLP METRICS (BERTScore, ROUGE)")
    logger.info("="*60)
    
    from evaluate import compute_nlp_metrics
    
    nlp_metrics = compute_nlp_metrics(reference=context_text, hypothesis=output)
    
    for metric_name, metric_value in nlp_metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    if nlp_metrics.get('bertscore_f1', 0.0) > 0:
        logger.info("✅ BERTScore computed successfully")
    else:
        logger.warning("⚠️  BERTScore returned 0 - check CPU fallback")
    
    if nlp_metrics.get('rouge1_f1', 0.0) > 0:
        logger.info("✅ ROUGE computed successfully")
    else:
        logger.warning("⚠️  ROUGE returned 0 - check reference/hypothesis")
    
    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
