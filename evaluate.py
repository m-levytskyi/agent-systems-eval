"""Evaluate Monolithic vs Ensemble agents for document synthesis.

Runs both agent approaches on a set of tasks and source documents, tracking metrics
in MLflow and using MLflow GenAI LLM-judge scorers for quality evaluation.
"""

import os
import json
import sys
from typing import List, Dict, Any
from pathlib import Path

import mlflow
import pandas as pd
from packaging.version import Version
from dotenv import load_dotenv
from rate_limits import RequestRateLimiter

from monolithic import MonolithicAgent
from ensemble import EnsembleAgent
from utils import setup_logging, load_source_documents

load_dotenv()

logger = setup_logging("evaluate")

def load_tasks(task_file: str) -> List[Dict[str, Any]]:
    """Load synthesis tasks from JSON file."""
    with open(task_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_value_to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"yes", "true", "y"}:
        return 1.0
    if text in {"no", "false", "n"}:
        return 0.0
    if text in {"fully"}:
        return 1.0
    if text in {"mostly"}:
        return 0.75
    if text in {"partially"}:
        return 0.5
    if text in {"not"}:
        return 0.0

    try:
        return float(text)
    except ValueError:
        return 0.0


def evaluate_with_mlflow_judges(
    *,
    task_description: str,
    synthesis: str,
    reference_text: str,
    judge_model: str,
) -> Dict[str, Any]:
    """Evaluate synthesis quality using concurrent execution with fallback to mlflow.genai.evaluate.
    
    Uses parallel execution for efficiency and weighted scoring for accuracy.
    Grounding failures act as a kill-switch capping the overall score.
    """
    try:
        from mlflow.genai.scorers import Guidelines
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception as exc:
        raise RuntimeError(
            "MLflow GenAI scorers are not available. Install a recent MLflow version with GenAI support. "
            "(e.g., `pip install -U mlflow`)."
        ) from exc

    # Define scorers with proper Guidelines-based metrics
    scorers = [
        Guidelines(
            name="instruction_adherence",
            model=judge_model,
            guidelines=(
                "The synthesis must strictly follow the task instructions provided in inputs.query. "
                "It should address all requirements and constraints specified in the task."
            ),
        ),
        Guidelines(
            name="completeness",
            model=judge_model,
            guidelines=(
                "The synthesis must fully address the task requirements. "
                "It should cover all key aspects implied by the task prompt."
            ),
        ),
        Guidelines(
            name="coherence",
            model=judge_model,
            guidelines=(
                "The synthesis must be clear, logically structured, and coherent. "
                "It should read as a single integrated document."
            ),
        ),
        Guidelines(
            name="grounded_in_sources",
            model=judge_model,
            guidelines=(
                "All factual claims must be supported by the provided context in inputs.context. "
                "Do not introduce unsupported facts. This is a critical requirement."
            ),
        ),
        Guidelines(
            name="professional_quality",
            model=judge_model,
            guidelines=(
                "The synthesis must be professional quality: accurate, appropriately detailed, and well written."
            ),
        ),
    ]

    # Use concurrent execution for parallel scoring
    inputs = {"query": task_description, "context": reference_text}
    scores: Dict[str, float] = {}
    feedback: Dict[str, Any] = {}
    
    def evaluate_scorer(scorer):
        """Helper to evaluate a single scorer."""
        name = getattr(scorer, "name", scorer.__class__.__name__)
        fb = scorer(inputs=inputs, outputs=synthesis)
        value = getattr(fb, "value", fb)
        rationale = getattr(fb, "rationale", None)
        return name, _score_value_to_float(value), {"value": value, "rationale": rationale}
    
    # Run scorers concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_scorer, scorer): scorer for scorer in scorers}
        for future in as_completed(futures):
            name, score, fb = future.result()
            scores[name] = score
            feedback[name] = fb
    
    # Compute weighted overall score with grounding kill-switch
    grounding_score = scores.get("grounded_in_sources", 0.0)
    instruction_adherence = scores.get("instruction_adherence", 0.0)
    completeness = scores.get("completeness", 0.0)
    coherence = scores.get("coherence", 0.0)
    quality = scores.get("professional_quality", 0.0)
    
    # Weighted average: grounding and instruction adherence are critical
    # Weights: grounding=0.3, instruction_adherence=0.25, completeness=0.2, coherence=0.15, quality=0.1
    weighted_overall = (
        grounding_score * 0.30 +
        instruction_adherence * 0.25 +
        completeness * 0.20 +
        coherence * 0.15 +
        quality * 0.10
    )
    
    # Kill-switch: if grounding fails badly (< 0.5), cap overall at 0.5
    if grounding_score < 0.5:
        weighted_overall = min(weighted_overall, 0.5)
    
    scores["weighted_overall"] = weighted_overall
    
    return {"scores": scores, "detailed": feedback, "all_metrics": scores}


def compute_nlp_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute NLP metrics (BERTScore, ROUGE) between reference and hypothesis.
    
    Args:
        reference: Reference text (ground truth or source documents)
        hypothesis: Generated text (synthesis output)
        
    Returns:
        Dictionary of NLP metric scores
    """
    metrics = {}

    # Guard against missing outputs
    reference = reference or ""
    hypothesis = hypothesis or ""
    if not reference or not hypothesis:
        metrics['bertscore_precision'] = 0.0
        metrics['bertscore_recall'] = 0.0
        metrics['bertscore_f1'] = 0.0
        metrics['rouge1_f1'] = 0.0
        metrics['rougeL_f1'] = 0.0
        return metrics
    
    try:
        # BERTScore
        from bert_score import score as bert_score
        P, R, F1 = bert_score([hypothesis], [reference], lang='en', verbose=False)
        metrics['bertscore_precision'] = float(P[0])
        metrics['bertscore_recall'] = float(R[0])
        metrics['bertscore_f1'] = float(F1[0])
    except Exception as e:
        logger.warning(f"Warning: BERTScore computation failed: {e}")
        metrics['bertscore_precision'] = 0.0
        metrics['bertscore_recall'] = 0.0
        metrics['bertscore_f1'] = 0.0
    
    try:
        # ROUGE scores
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        metrics['rouge1_f1'] = scores['rouge1'].fmeasure
        metrics['rougeL_f1'] = scores['rougeL'].fmeasure
    except Exception as e:
        logger.warning(f"Warning: ROUGE computation failed: {e}")
        metrics['rouge1_f1'] = 0.0
        metrics['rougeL_f1'] = 0.0
    
    return metrics


def run_experiment(
    agent_type: str,
    agent,
    source_documents: List[str],
    tasks: List[Dict[str, Any]],
    judge_model: str,
) -> None:
    """
    Run an experiment for a specific agent type.
    
    Args:
        agent_type: "monolithic" or "ensemble"
        agent: The agent instance
        source_documents: List of source documents
        tasks: List of synthesis tasks
        judge_model: Model to use for LLM-as-a-judge evaluation
    """
    experiment_name = f"document_synthesis_{agent_type}"
    
    mlflow.set_experiment(experiment_name)
    
    for task in tasks:
        task_id = task["task_id"]
        task_description = task["task_description"]
        
        with mlflow.start_run(run_name=f"{agent_type}_{task_id}"):
            # Log parameters
            mlflow.log_param("agent_type", agent_type)
            mlflow.log_param("task_id", task_id)
            mlflow.log_param("task_description", task_description)
            mlflow.log_param("num_source_documents", len(source_documents))
            mlflow.log_param("model", agent.model)
            
            # Run synthesis
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {agent_type} on {task_id}")
            logger.info(f"{'='*60}")
            
            result = agent.synthesize(source_documents, task_description)
            output = result.get("output") or ""
            metrics = result["metrics"]

            if not output.strip():
                logger.warning("⚠️  Model returned empty output; judge and NLP metrics will be zero.")
            
            # Log process metrics
            mlflow.log_metric("latency_seconds", metrics["latency_seconds"])
            mlflow.log_metric("total_tokens", metrics["total_tokens"])
            mlflow.log_metric("prompt_tokens", metrics["prompt_tokens"])
            mlflow.log_metric("completion_tokens", metrics["completion_tokens"])
            mlflow.log_metric("num_api_calls", metrics["num_api_calls"])
            
            # Log agent-specific metrics
            if agent_type == "ensemble":
                mlflow.log_metric("archivist_tokens", metrics.get("archivist_tokens", 0))
                mlflow.log_metric("drafter_tokens", metrics.get("drafter_tokens", 0))
                mlflow.log_metric("critic_tokens", metrics.get("critic_tokens", 0))
                mlflow.log_metric("orchestrator_tokens", metrics.get("orchestrator_tokens", 0))
                mlflow.log_metric("num_iterations", metrics.get("num_iterations", 0))
            
            # LLM-as-a-judge evaluation (using concurrent futures for parallel execution)
            logger.info("Evaluating output quality with LLM judge (parallel execution)...")
            reference_text = "\n\n".join(source_documents)
            judge_result = (
                evaluate_with_mlflow_judges(
                    task_description=task_description,
                    synthesis=output,
                    reference_text=reference_text,
                    judge_model=judge_model,
                )
                if output
                else {"scores": {
                "completeness": 0.0,
                "coherence": 0.0,
                "accuracy": 0.0,
                "quality": 0.0,
                "overall": 0.0,
            }, "detailed": {}, "all_metrics": {}}
            )
            judge_scores = judge_result["scores"]
            all_metrics = judge_result.get("all_metrics", {})
            detailed_feedback = judge_result.get("detailed", {})
            
            # Log quality scores with weighted overall
            for criterion, score in judge_scores.items():
                mlflow.log_metric(f"judge_{criterion}_score", score)
            
            # Log all detailed metrics
            for metric_name, score in all_metrics.items():
                if f"judge_{metric_name}_score" not in [f"judge_{k}_score" for k in judge_scores.keys()]:
                    mlflow.log_metric(f"judge_{metric_name}_score", score)

            # Store rationales as an artifact for later inspection
            if detailed_feedback:
                judge_file = f"{agent_type}_{task_id}_judge_feedback.json"
                with open(judge_file, "w", encoding="utf-8") as f:
                    json.dump(detailed_feedback, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(judge_file)
                os.remove(judge_file)
            
            # Compute NLP metrics (using concatenated source documents as reference)
            logger.info("Computing NLP metrics (BERTScore, ROUGE)...")
            nlp_metrics = compute_nlp_metrics(reference_text, output)
            
            # Log NLP metrics
            for metric_name, metric_value in nlp_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts
            output_file = f"{agent_type}_{task_id}_output.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            mlflow.log_artifact(output_file)
            os.remove(output_file)
            
            # Log monolithic document summaries
            if agent_type == "monolithic" and "intermediate_outputs" in result:
                intermediate = result["intermediate_outputs"]
                
                if "document_summaries" in intermediate:
                    for idx, summary in enumerate(intermediate["document_summaries"], start=1):
                        summary_file = f"{agent_type}_{task_id}_doc_{idx}_summary.txt"
                        with open(summary_file, "w", encoding="utf-8") as f:
                            f.write(summary)
                        mlflow.log_artifact(summary_file)
                        os.remove(summary_file)
                
                # Log summary metadata as JSON
                if "summary_metadata" in intermediate:
                    metadata_file = f"{agent_type}_{task_id}_summary_metadata.json"
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(intermediate["summary_metadata"], f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(metadata_file)
                    os.remove(metadata_file)
                
                # Log summarization metrics
                if "document_summaries_tokens" in metrics:
                    mlflow.log_metric("document_summaries_tokens", metrics["document_summaries_tokens"])
                if "num_documents_summarized" in metrics:
                    mlflow.log_metric("num_documents_summarized", metrics["num_documents_summarized"])
            
            # Log intermediate outputs for ensemble
            if agent_type == "ensemble" and "intermediate_outputs" in result:
                intermediate = result["intermediate_outputs"]
                
                # Log document summaries from ensemble archivist
                if "document_summaries" in intermediate:
                    for idx, summary in enumerate(intermediate["document_summaries"], start=1):
                        summary_file = f"{agent_type}_{task_id}_doc_{idx}_summary.txt"
                        with open(summary_file, "w", encoding="utf-8") as f:
                            f.write(summary)
                        mlflow.log_artifact(summary_file)
                        os.remove(summary_file)
                
                # Log summary metadata as JSON
                if "summary_metadata" in intermediate:
                    metadata_file = f"{agent_type}_{task_id}_summary_metadata.json"
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(intermediate["summary_metadata"], f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(metadata_file)
                    os.remove(metadata_file)
                
                # Log summarization metrics
                if "document_summaries_tokens" in metrics:
                    mlflow.log_metric("document_summaries_tokens", metrics["document_summaries_tokens"])
                if "num_documents_summarized" in metrics:
                    mlflow.log_metric("num_documents_summarized", metrics["num_documents_summarized"])
                
                # Log archived_info and draft
                for stage in ["archived_info", "draft"]:
                    if stage in intermediate:
                        stage_file = f"{agent_type}_{task_id}_{stage}.txt"
                        with open(stage_file, "w", encoding="utf-8") as f:
                            f.write(str(intermediate[stage]))
                        mlflow.log_artifact(stage_file)
                        os.remove(stage_file)
                
                # Log iteration history as JSON
                if "iteration_history" in intermediate and intermediate["iteration_history"]:
                    history_file = f"{agent_type}_{task_id}_iteration_history.json"
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(intermediate["iteration_history"], f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(history_file)
                    os.remove(history_file)
                    
                    # Also log per-iteration drafts and critiques
                    for iteration_data in intermediate["iteration_history"]:
                        iter_num = iteration_data["iteration"]
                        
                        draft_file = f"{agent_type}_{task_id}_iteration_{iter_num}_draft.txt"
                        with open(draft_file, "w", encoding="utf-8") as f:
                            f.write(iteration_data.get("draft", ""))
                        mlflow.log_artifact(draft_file)
                        os.remove(draft_file)
                        
                        critique_file = f"{agent_type}_{task_id}_iteration_{iter_num}_critique.txt"
                        with open(critique_file, "w", encoding="utf-8") as f:
                            f.write(iteration_data.get("critique", ""))
                        mlflow.log_artifact(critique_file)
                        os.remove(critique_file)
                        
            logger.info("\nMetrics Summary:")
            logger.info(f"  Latency: {metrics['latency_seconds']:.2f}s")
            logger.info(f"  Total Tokens: {metrics['total_tokens']}")
            logger.info(f"  API Calls: {metrics['num_api_calls']}")
            if agent_type == "ensemble":
                logger.info(f"  Iterations: {metrics.get('num_iterations', 0)}")
            logger.info("\nJudge Scores (0-1 scale, weighted with grounding kill-switch):")
            for criterion, score in judge_scores.items():
                # MLflow scorers return 0-1; we keep the native scale to avoid the misleading "/5" label.
                logger.info(f"  {criterion.capitalize()}: {score:.2f}")
            
            logger.info("\nNLP Metrics:")
            for metric_name, metric_value in nlp_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")


def main():
    """Main evaluation function."""
    
    # Check for test mode
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    logger.info("="*60)
    if test_mode:
        logger.info("TEST MODE: Single Paper Evaluation")
    else:
        logger.info("Document Synthesis Evaluation: Monolithic vs Ensemble")
    logger.info("="*60)
    
    # Configuration
    doc_dir = "data/source_documents"
    task_file = "data/tasks/synthesis_tasks.json"
    doc_pattern = "paper_1.pdf" if test_mode else "paper_*.pdf"  # Single paper for test mode
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    judge_model = os.getenv("JUDGE_MODEL", f"openai:/{model}")
    crewai_model = os.getenv("CREWAI_MODEL", f"openai/{model}")

    # Optional rate limiting for remote providers
    rpm_limit = int(os.getenv("MAX_RPM", "0"))
    rpd_limit = int(os.getenv("MAX_RPD", "0"))
    
    # Load data
    logger.info("\nLoading source documents and tasks...")
    source_documents = load_source_documents(doc_dir, pattern=doc_pattern)
    tasks = load_tasks(task_file)
    
    if test_mode:
        logger.info(f"\n⚠️  TEST MODE: Using only first document and first task")
        source_documents = source_documents[:1]
        tasks = tasks[:1]
    
    logger.info(f"Loaded {len(source_documents)} source documents")
    logger.info(f"Loaded {len(tasks)} synthesis tasks")
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    assert Version(mlflow.__version__) >= Version("3.1.0"), "Please upgrade mlflow to >= 3.1.0"

    rate_limiter = None
    if rpm_limit > 0 and rpd_limit > 0:
        rate_limiter = RequestRateLimiter(max_per_minute=rpm_limit, max_per_day=rpd_limit)
    
    # Run monolithic agent experiments
    logger.info("\n" + "="*60)
    logger.info("MONOLITHIC AGENT EVALUATION")
    logger.info("="*60)
    monolithic_agent = MonolithicAgent(model=model, rate_limiter=rate_limiter)
    run_experiment("monolithic", monolithic_agent, source_documents, tasks, judge_model)
    
    # Run ensemble agent experiments (CrewAI Flow-based with recursive orchestration)
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE AGENT EVALUATION (CrewAI Flows with Orchestrator)")
    logger.info("="*60)
    ensemble_agent = EnsembleAgent(model=crewai_model, rate_limiter=rate_limiter)
    run_experiment("ensemble", ensemble_agent, source_documents, tasks, judge_model)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info("\nResults logged to MLflow. To view:")
    logger.info("  mlflow ui")
    logger.info("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
