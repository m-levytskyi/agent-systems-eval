"""
Evaluation Script: Compare Monolithic vs Ensemble agents for document synthesis.

This script runs both agent approaches on a set of tasks and source documents,
tracking all metrics in MLflow and using LLM-as-a-judge for quality evaluation.
"""

import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path

import mlflow
from packaging.version import Version
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google import genai
from google.genai import types
from rate_limits import RequestRateLimiter

from monolithic import MonolithicAgent
from ensemble import EnsembleAgent

load_dotenv()


def load_source_documents(doc_dir: str) -> List[str]:
    """Load all source documents (PDF or text) from the specified directory."""
    documents = []
    doc_path = Path(doc_dir)
    
    # Load PDF files
    for filepath in sorted(doc_path.glob("*.pdf")):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        documents.append(text.strip())
    
    # Load text files (if any)
    for filepath in sorted(doc_path.glob("*.txt")):
        with open(filepath, "r") as f:
            documents.append(f.read())
    
    return documents


def load_tasks(task_file: str) -> List[Dict[str, Any]]:
    """Load synthesis tasks from JSON file."""
    with open(task_file, "r") as f:
        return json.load(f)


def estimate_cost(metrics: Dict[str, Any], model: str) -> float:
    """
    Estimate API cost based on token usage and model pricing.
    
    Pricing (as of 2024):
    - Gemini Flash: Free tier up to 15 RPM, 1M TPM, 1500 RPD
    - For paid: ~$0.000075/1K input tokens, ~$0.0003/1K output tokens
    """
    # Gemini pricing (very low cost)
    if "gemini" in model.lower():
        pricing = {"prompt": 0.000075, "completion": 0.0003}
    else:
        # Default fallback pricing
        pricing = {"prompt": 0.001, "completion": 0.002}
    
    prompt_cost = (metrics["prompt_tokens"] / 1000) * pricing["prompt"]
    completion_cost = (metrics["completion_tokens"] / 1000) * pricing["completion"]
    
    return prompt_cost + completion_cost


def create_llm_judge_prompt(task_description: str, synthesis: str) -> str:
    """Create a prompt for LLM-as-a-judge evaluation."""
    return f"""Evaluate the following document synthesis on a scale of 1-5 for each criterion:

Task: {task_description}

Synthesized Document:
{synthesis}

Please rate the following aspects (1=Poor, 5=Excellent):
1. Completeness: Does it fully address the task requirements?
2. Coherence: Is the writing clear, logical, and well-structured?
3. Accuracy: Is the information accurate and well-integrated?
4. Quality: Overall professional quality of the synthesis

Provide your ratings in this format:
Completeness: X/5
Coherence: X/5
Accuracy: X/5
Quality: X/5
Overall: X/5

Briefly explain your ratings."""


def evaluate_with_llm_judge(
    task_description: str,
    synthesis: str,
    model: str,
    client: genai.Client,
    rate_limiter: RequestRateLimiter,
) -> Dict[str, float]:
    """
    Evaluate synthesis quality using LLM-as-a-judge.
    
    Args:
        task_description: The synthesis task
        synthesis: The synthesized output
        model: Model to use for judging
        
    Returns:
        Dictionary of scores
    """
    judge_prompt = create_llm_judge_prompt(task_description, synthesis)

    if rate_limiter:
        rate_limiter.acquire()

    response = client.models.generate_content(
        model=model,
        contents=judge_prompt,
        config=types.GenerateContentConfig(temperature=0.3)
    )

    content = response.text or ""
    scores: Dict[str, float] = {}

    for line in content.split('\n'):
        if ':' in line:
            for criterion in ['Completeness', 'Coherence', 'Accuracy', 'Quality', 'Overall']:
                if criterion.lower() in line.lower():
                    try:
                        score_part = line.split(':')[1].strip().split('/')[0].strip()
                        scores[criterion.lower()] = float(score_part)
                    except (IndexError, ValueError):
                        pass

    # Ensure all criteria present
    for criterion in ['completeness', 'coherence', 'accuracy', 'quality', 'overall']:
        scores.setdefault(criterion, 0.0)

    return scores


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
        metrics['rouge2_f1'] = 0.0
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
        print(f"Warning: BERTScore computation failed: {e}")
        metrics['bertscore_precision'] = 0.0
        metrics['bertscore_recall'] = 0.0
        metrics['bertscore_f1'] = 0.0
    
    try:
        # ROUGE scores
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        metrics['rouge1_f1'] = scores['rouge1'].fmeasure
        metrics['rouge2_f1'] = scores['rouge2'].fmeasure
        metrics['rougeL_f1'] = scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Warning: ROUGE computation failed: {e}")
        metrics['rouge1_f1'] = 0.0
        metrics['rouge2_f1'] = 0.0
        metrics['rougeL_f1'] = 0.0
    
    return metrics


def run_experiment(
    agent_type: str,
    agent,
    source_documents: List[str],
    tasks: List[Dict[str, Any]],
    judge_model: str,
    client: genai.Client,
    rate_limiter: RequestRateLimiter,
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
            print(f"\n{'='*60}")
            print(f"Running {agent_type} on {task_id}")
            print(f"{'='*60}")
            
            result = agent.synthesize(source_documents, task_description)
            output = result.get("output") or ""
            metrics = result["metrics"]

            if not output.strip():
                print("⚠️  Model returned empty output; judge and NLP metrics will be zero.")
            
            # Calculate cost
            estimated_cost = estimate_cost(metrics, agent.model)
            
            # Log process metrics
            mlflow.log_metric("latency_seconds", metrics["latency_seconds"])
            mlflow.log_metric("total_tokens", metrics["total_tokens"])
            mlflow.log_metric("prompt_tokens", metrics["prompt_tokens"])
            mlflow.log_metric("completion_tokens", metrics["completion_tokens"])
            mlflow.log_metric("num_api_calls", metrics["num_api_calls"])
            mlflow.log_metric("estimated_cost_usd", estimated_cost)
            
            # Log agent-specific metrics
            if agent_type == "ensemble":
                mlflow.log_metric("archivist_tokens", metrics.get("archivist_tokens", 0))
                mlflow.log_metric("drafter_tokens", metrics.get("drafter_tokens", 0))
                mlflow.log_metric("critic_tokens", metrics.get("critic_tokens", 0))
            
            # LLM-as-a-judge evaluation
            print(f"Evaluating output quality with LLM judge...")
            judge_scores = evaluate_with_llm_judge(task_description, output, judge_model, client, rate_limiter) if output else {
                "completeness": 0.0,
                "coherence": 0.0,
                "accuracy": 0.0,
                "quality": 0.0,
                "overall": 0.0,
            }
            
            # Log quality scores
            for criterion, score in judge_scores.items():
                mlflow.log_metric(f"judge_{criterion}_score", score)
            
            # Compute NLP metrics (using concatenated source documents as reference)
            print(f"Computing NLP metrics (BERTScore, ROUGE)...")
            reference_text = "\n\n".join(source_documents)
            nlp_metrics = compute_nlp_metrics(reference_text, output)
            
            # Log NLP metrics
            for metric_name, metric_value in nlp_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts
            output_file = f"{agent_type}_{task_id}_output.txt"
            with open(output_file, "w") as f:
                f.write(output)
            mlflow.log_artifact(output_file)
            os.remove(output_file)
            
            # Log intermediate outputs for ensemble
            if agent_type == "ensemble" and "intermediate_outputs" in result:
                for stage, content in result["intermediate_outputs"].items():
                    stage_file = f"{agent_type}_{task_id}_{stage}.txt"
                    with open(stage_file, "w") as f:
                        f.write(content)
                    mlflow.log_artifact(stage_file)
                    os.remove(stage_file)
            
            print(f"\nMetrics Summary:")
            print(f"  Latency: {metrics['latency_seconds']:.2f}s")
            print(f"  Total Tokens: {metrics['total_tokens']}")
            print(f"  API Calls: {metrics['num_api_calls']}")
            print(f"  Estimated Cost: ${estimated_cost:.4f}")
            print(f"\nJudge Scores:")
            for criterion, score in judge_scores.items():
                print(f"  {criterion.capitalize()}: {score}/5")
            print(f"\nNLP Metrics:")
            for metric_name, metric_value in nlp_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("Document Synthesis Evaluation: Monolithic vs Ensemble")
    print("="*60)
    
    # Configuration
    doc_dir = "data/source_documents"
    task_file = "data/tasks/synthesis_tasks.json"
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    judge_model = os.getenv("GEMINI_JUDGE_MODEL", model)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    rpm_limit = int(os.getenv("GEMINI_MAX_RPM", "4"))
    rpd_limit = int(os.getenv("GEMINI_MAX_RPD", "15"))
    
    # Load data
    print("\nLoading source documents and tasks...")
    source_documents = load_source_documents(doc_dir)
    tasks = load_tasks(task_file)
    
    print(f"Loaded {len(source_documents)} source documents")
    print(f"Loaded {len(tasks)} synthesis tasks")
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    assert Version(mlflow.__version__) >= Version("2.18.0"), (
        "This feature requires MLflow version 2.18.0 or newer. "
        "Please upgrade mlflow to enable Gemini trace logging."
    )
    mlflow.gemini.autolog()

    # Shared Gemini client and rate limiter across all calls
    gemini_client = genai.Client(api_key=api_key)
    rate_limiter = RequestRateLimiter(max_per_minute=rpm_limit, max_per_day=rpd_limit)
    
    # Run monolithic agent experiments
    print("\n" + "="*60)
    print("MONOLITHIC AGENT EVALUATION")
    print("="*60)
    monolithic_agent = MonolithicAgent(model=model, client=gemini_client, rate_limiter=rate_limiter)
    run_experiment("monolithic", monolithic_agent, source_documents, tasks, judge_model, gemini_client, rate_limiter)
    
    # Run ensemble agent experiments
    print("\n" + "="*60)
    print("ENSEMBLE AGENT EVALUATION")
    print("="*60)
    ensemble_agent = EnsembleAgent(model=model, client=gemini_client, rate_limiter=rate_limiter)
    run_experiment("ensemble", ensemble_agent, source_documents, tasks, judge_model, gemini_client, rate_limiter)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults logged to MLflow. To view:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
