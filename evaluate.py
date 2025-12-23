"""Evaluate Monolithic vs Ensemble agents for document synthesis.

Runs both agent approaches on a set of tasks and source documents, tracking metrics
in MLflow and using MLflow GenAI LLM-judge scorers for quality evaluation.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path

import mlflow
from packaging.version import Version
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from rate_limits import RequestRateLimiter

from monolithic import MonolithicAgent
from ensemble import EnsembleAgent

load_dotenv()


def load_source_documents(doc_dir: str, pattern: str = "*.pdf") -> List[str]:
    """Load all source documents (PDF or text) from the specified directory.
    
    Args:
        doc_dir: Directory containing source documents
        pattern: Glob pattern for filtering files (default: "*.pdf")
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


def estimate_cost(metrics: Dict[str, Any], model: str) -> float:
    """
    Estimate API cost based on token usage and model pricing.
    
    Pricing (as of 2024):
    - Gemini Flash: Free tier up to 15 RPM, 1M TPM, 1500 RPD
    - For paid: ~$0.000075/1K input tokens, ~$0.0003/1K output tokens
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return 0.0

    # Gemini pricing (very low cost)
    if "gemini" in model.lower():
        pricing = {"prompt": 0.000075, "completion": 0.0003}
    else:
        # Default fallback pricing
        pricing = {"prompt": 0.001, "completion": 0.002}
    
    prompt_cost = (metrics["prompt_tokens"] / 1000) * pricing["prompt"]
    completion_cost = (metrics["completion_tokens"] / 1000) * pricing["completion"]
    
    return prompt_cost + completion_cost


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
    """Evaluate synthesis quality using MLflow GenAI built-in LLM-judge scorers."""
    try:
        from mlflow.genai.scorers import RelevanceToQuery, Guidelines
    except Exception as exc:
        raise RuntimeError(
            "MLflow GenAI scorers are not available. Install a recent MLflow version with GenAI support. "
            "(e.g., `pip install -U mlflow`)."
        ) from exc

    inputs = {"query": task_description, "context": reference_text}

    scorers = [
        RelevanceToQuery(name="relevance_to_task", model=judge_model),
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
                "Do not introduce unsupported facts."
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

    scores: Dict[str, float] = {}
    feedback: Dict[str, Any] = {}
    for scorer in scorers:
        name = getattr(scorer, "name", scorer.__class__.__name__)
        fb = scorer(inputs=inputs, outputs=synthesis)
        value = getattr(fb, "value", fb)
        rationale = getattr(fb, "rationale", None)
        scores[name] = _score_value_to_float(value)
        feedback[name] = {"value": value, "rationale": rationale}

    # For backward-compat with older metric names.
    legacy = {
        "completeness": scores.get("completeness", 0.0),
        "coherence": scores.get("coherence", 0.0),
        "accuracy": scores.get("grounded_in_sources", 0.0),
        "quality": scores.get("professional_quality", 0.0),
        "overall": (
            scores.get("relevance_to_task", 0.0)
            + scores.get("completeness", 0.0)
            + scores.get("coherence", 0.0)
            + scores.get("grounded_in_sources", 0.0)
            + scores.get("professional_quality", 0.0)
        )
        / 5.0,
    }

    return {"scores": legacy, "detailed": feedback}


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
                mlflow.log_metric("orchestrator_tokens", metrics.get("orchestrator_tokens", 0))
                mlflow.log_metric("num_iterations", metrics.get("num_iterations", 0))
            
            # LLM-as-a-judge evaluation
            print("Evaluating output quality with LLM judge...")
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
            }, "detailed": {}}
            )
            judge_scores = judge_result["scores"]
            
            # Log quality scores
            for criterion, score in judge_scores.items():
                mlflow.log_metric(f"judge_{criterion}_score", score)

            # Store rationales as an artifact for later inspection
            if judge_result.get("detailed"):
                judge_file = f"{agent_type}_{task_id}_judge_feedback.json"
                with open(judge_file, "w", encoding="utf-8") as f:
                    json.dump(judge_result["detailed"], f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(judge_file)
                os.remove(judge_file)
            
            # Compute NLP metrics (using concatenated source documents as reference)
            print("Computing NLP metrics (BERTScore, ROUGE)...")
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
            
            # Log intermediate outputs for ensemble
            if agent_type == "ensemble" and "intermediate_outputs" in result:
                intermediate = result["intermediate_outputs"]
                
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
            
            print("\nMetrics Summary:")
            print(f"  Latency: {metrics['latency_seconds']:.2f}s")
            print(f"  Total Tokens: {metrics['total_tokens']}")
            print(f"  API Calls: {metrics['num_api_calls']}")
            print(f"  Estimated Cost: ${estimated_cost:.4f}")
            if agent_type == "ensemble":
                print(f"  Iterations: {metrics.get('num_iterations', 0)}")
            print("\nJudge Scores:")
            for criterion, score in judge_scores.items():
                print(f"  {criterion.capitalize()}: {score}/5")
            print("\nNLP Metrics:")
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
    doc_pattern = "paper_*.pdf"  # Use paper_*.pdf for evaluation
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    judge_model = os.getenv("JUDGE_MODEL", f"openai:/{model}")
    crewai_model = os.getenv("CREWAI_MODEL", f"openai/{model}")

    # Optional rate limiting for remote providers
    rpm_limit = int(os.getenv("MAX_RPM", "0"))
    rpd_limit = int(os.getenv("MAX_RPD", "0"))
    
    # Load data
    print("\nLoading source documents and tasks...")
    source_documents = load_source_documents(doc_dir, pattern=doc_pattern)
    tasks = load_tasks(task_file)
    
    print(f"Loaded {len(source_documents)} source documents")
    print(f"Loaded {len(tasks)} synthesis tasks")
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    assert Version(mlflow.__version__) >= Version("3.1.0"), "Please upgrade mlflow to >= 3.1.0"

    rate_limiter = None
    if rpm_limit > 0 and rpd_limit > 0:
        rate_limiter = RequestRateLimiter(max_per_minute=rpm_limit, max_per_day=rpd_limit)
    
    # Run monolithic agent experiments
    print("\n" + "="*60)
    print("MONOLITHIC AGENT EVALUATION")
    print("="*60)
    monolithic_agent = MonolithicAgent(model=model, rate_limiter=rate_limiter)
    run_experiment("monolithic", monolithic_agent, source_documents, tasks, judge_model)
    
    # Run ensemble agent experiments (CrewAI Flow-based with recursive orchestration)
    print("\n" + "="*60)
    print("ENSEMBLE AGENT EVALUATION (CrewAI Flows with Orchestrator)")
    print("="*60)
    ensemble_agent = EnsembleAgent(model=crewai_model, rate_limiter=rate_limiter)
    run_experiment("ensemble", ensemble_agent, source_documents, tasks, judge_model)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults logged to MLflow. To view:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
