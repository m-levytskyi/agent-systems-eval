"""Evaluate Monolithic vs Ensemble agents for document synthesis.

Runs both agent approaches on a set of tasks and source documents, tracking metrics
in MLflow and using MLflow GenAI LLM-judge scorers for quality evaluation.
"""

import os
import json
import sys
import argparse
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
    context: str,
    judge_model: str,
) -> Dict[str, Any]:
    """Evaluate synthesis quality using mlflow.genai.evaluate with specific judges.
    
    Uses groundedness, instruction_adherence and completeness judges.
    """
    try:
        from mlflow.genai.judges import make_judge
    except ImportError as exc:
        raise RuntimeError(
            "MLflow GenAI judges are not available. Install a recent MLflow version with GenAI support."
        ) from exc
    
    # 1. Define Judges
    # Note: MLflow variables must be exactly {{ inputs }}, {{ outputs }}, etc. No dot notation allowed.
    # We will pass a formatted string containing both task and context into inputs.
    groundedness_judge = make_judge(
        name="groundedness",
        model=judge_model,
        instructions=(
            "You are an expert academic reviewer evaluating **Groundedness**.\n\n"
            "**Task**: Verify all claims in the Agent Output can be traced to the Context.\n\n"
            "**Evaluation Criteria:**\n"
            "1. **Direct Statements**: Facts, statistics, and specific findings must be in the Context\n"
            "2. **Reasonable Synthesis**: Allow paraphrasing and combining related information from Context\n"
            "3. **Hallucination Detection**: Flag ONLY information that cannot be traced to Context at all\n\n"
            "**Grading Scale (0-5):**\n"
            "- 0: Contains fabricated information not traceable to Context\n"
            "- 1: Significant hallucinations or unsupported claims\n"
            "- 2: Some claims lack clear support in Context\n"
            "- 3: Mostly grounded with minor paraphrasing issues\n"
            "- 4: Well-grounded, reasonable synthesis\n"
            "- 5: Perfectly grounded, all claims directly traceable\n\n"
            "**Input Data (Task + Context):**\n{{ inputs }}\n\n"
            "**Agent Output:**\n{{ outputs }}\n\n"
            "Consider: Synthesis requires combining information. Paraphrasing is acceptable if the core meaning is preserved."
        ),
    )

    instruction_adherence_judge = make_judge(
        name="instruction_adherence",
        model=judge_model,
        instructions=(
            "You are a strict Logic Evaluator. Assess **Instruction Adherence** with rigorous precision.\n\n"
            "**Input Data (containing Task and Context):**\n"
            "{{ inputs }}\n\n"
            "**Strict Evaluation Criteria:**\n"
            "1. **Expected Elements Coverage:** Check if ALL expected elements from the task (if specified) are explicitly addressed. Missing even ONE element should reduce the score.\n"
            "2. **Format Compliance:** If the task requested specific format (JSON, list, headers, sections), verify EXACT adherence. Generic prose when structure was requested = major penalty.\n"
            "3. **Task Fidelity:** Did the output perform ONLY the specific task requested without scope drift? Check:\n"
            "   - Task 1 (scope/terminology) should NOT include detailed findings from individual papers\n"
            "   - Task 2 (structured extraction) should cover EACH paper individually, not synthesize across them\n"
            "   - Task 3 (comparative meta-analysis) should identify patterns ACROSS papers, not just list them\n"
            "4. **Negative Constraints:** If task specified exclusions or what NOT to do, verify strict compliance.\n"
            "5. **Specificity Requirements:** If task asks for 'explicit research questions' or 'key statistical findings', vague summaries are insufficient.\n\n"
            "**Grading Scale (0-5) - BE HARSH:**\n"
            "- 0-1: Wrong task entirely, or ignored critical format requirements\n"
            "- 2: Correct general direction but missing 2+ expected elements OR significant format violations\n"
            "- 3: Addressed most requirements but missing 1 expected element OR minor format issues OR insufficient specificity\n"
            "- 4: All expected elements present with good structure, but minor deviations in depth or formatting\n"
            "- 5: Perfect compliance - all elements, correct format, appropriate scope, proper specificity\n\n"
            "**Agent Output:**\n"
            "{{ outputs }}\n\n"
            "**Scoring Philosophy:** Default to 2-3 range. Only award 4-5 for truly excellent adherence. Penalize heavily for missing expected elements or wrong task interpretation."
        ),
    )

    completeness_judge = make_judge(
        name="completeness",
        model=judge_model,
        instructions=(
            "You are evaluating **Completeness** of a synthesis output.\n\n"
            "**Evaluation Criteria:**\n"
            "1. **Task Requirements**: Check if Output addresses all elements requested in the TASK\n"
            "2. **Coverage**: If task mentions specific topics/papers, verify they're included\n"
            "3. **Depth**: Sections should have substance, not empty placeholders\n"
            "4. **Scope Accuracy**: Output should ONLY include information from the provided Context\n\n"
            "**Grading Scale (1-5):**\n"
            "- 1: Misses most required elements\n"
            "- 2: Partial coverage, significant gaps\n"
            "- 3: Adequate coverage with some omissions\n"
            "- 4: Comprehensive, minor elements missing\n"
            "- 5: Complete coverage of all task requirements\n\n"
            "**Input Data (Task + Context):**\n{{ inputs }}\n\n"
            "**Agent Output:**\n{{ outputs }}\n\n"
            "Note: First verify the Output only uses information from Context, then assess completeness."
        ),
    )

    # 2. Prepare Data
    formatted_inputs = f"TASK:\n{task_description}\n\nCONTEXT (Summaries):\n{context}"
    inputs_payload = {"inputs": formatted_inputs}
    outputs_payload = {"outputs": synthesis}

    # 3. Runs Judges Manually
    # Note: Call judges directly to avoid MLflow type check issues (InstructionsJudge vs EvaluationMetric)
    import time
    
    scores = {}
    feedback = {}
    
    # Define metric names mapping
    judge_map = {
        "groundedness": groundedness_judge,
        "instruction_adherence": instruction_adherence_judge,
        "completeness": completeness_judge
    }
    
    # Debug: Log what's being evaluated
    logger.info(f"\n{'='*60}")
    logger.info("DEBUG: Evaluating synthesis with judges")
    logger.info(f"Synthesis length: {len(synthesis)} chars")
    logger.info(f"Synthesis preview (first 200 chars): {synthesis[:200]}...")
    logger.info(f"Context length: {len(context)} chars")
    logger.info(f"{'='*60}\n")
    
    for name, judge in judge_map.items():
        try:
            logger.info(f"Running judge: {name}")
            # Judge call returns an Assessment/EvaluationResult object
            # Pass inputs/outputs as dictionaries as expected by InstructionsJudge
            result = judge(inputs=inputs_payload, outputs=outputs_payload)
            
            # Debug: Log raw judge result
            logger.info(f"DEBUG: Judge '{name}' raw result type: {type(result)}")
            logger.info(f"DEBUG: Judge '{name}' result attributes: {dir(result)}")
            
            # Extract score and justification
            score_val = getattr(result, "score", None)
            if score_val is None:
                score_val = getattr(result, "value", 0.0)
            
            logger.info(f"DEBUG: Judge '{name}' extracted score_val: {score_val} (type: {type(score_val)})")
                
            justification = getattr(result, "justification", None)
            if justification is None:
                justification = getattr(result, "rationale", "No rationale.")
            
            logger.info(f"DEBUG: Judge '{name}' justification preview: {str(justification)[:200]}...")
            
            scores[name] = _score_value_to_float(score_val)
            feedback[name] = justification
            
            logger.info(f"DEBUG: Judge '{name}' final float score: {scores[name]}")
            
            # Rate limit sleep (10 RPM = 6s per request)
            time.sleep(6)
            
        except Exception as e:
            logger.error(f"Judge {name} failed: {e}")
            scores[name] = 0.0
            feedback[name] = f"Error: {e}"

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
        # BERTScore with distilbert (3x faster than roberta-large) and CPU fallback for CUDA errors
        from bert_score import score as bert_score
        try:
            P, R, F1 = bert_score([hypothesis], [reference], 
                                 model_type='distilbert-base-uncased',
                                 lang='en', verbose=False)
            metrics['bertscore_precision'] = float(P[0])
            metrics['bertscore_recall'] = float(R[0])
            metrics['bertscore_f1'] = float(F1[0])
            logger.info(f"BERTScore computed successfully (GPU, distilbert)")
        except RuntimeError as cuda_err:
            if "CUDA" in str(cuda_err) or "kernel" in str(cuda_err).lower():
                logger.warning(f"CUDA error in BERTScore, falling back to CPU: {cuda_err}")
                P, R, F1 = bert_score([hypothesis], [reference], 
                                     model_type='distilbert-base-uncased',
                                     lang='en', verbose=False, device='cpu')
                metrics['bertscore_precision'] = float(P[0])
                metrics['bertscore_recall'] = float(R[0])
                metrics['bertscore_f1'] = float(F1[0])
                logger.info(f"BERTScore computed successfully (CPU fallback, distilbert)")
            else:
                raise
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
    agent_model_type: str = "ollama",
) -> None:
    """
    Run an experiment for a specific agent type.
    
    Args:
        agent_type: "monolithic" or "ensemble"
        agent: The agent instance
        source_documents: List of source documents
        tasks: List of synthesis tasks
        judge_model: Model to use for LLM-as-a-judge evaluation
        agent_model_type: "ollama" or "gemini" - appends _gemini suffix if gemini
    """
    experiment_name = f"document_synthesis_{agent_type}{'_gemini' if agent_model_type == 'gemini' else ''}"
    
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
            
            # LLM-as-a-judge evaluation
            logger.info("\n" + "="*60)
            logger.info("Evaluating output quality with LLM judge...")
            logger.info(f"Output to evaluate - length: {len(output)} chars")
            logger.info(f"Output preview (first 300 chars): {output[:300]}...")
            logger.info("="*60 + "\n")
            
            # Prepare context from summaries if available, else raw documents
            context_text = ""
            if "intermediate_outputs" in result and "document_summaries" in result["intermediate_outputs"]:
                summaries = result["intermediate_outputs"]["document_summaries"]
                for i, s in enumerate(summaries, 1):
                    context_text += f"Summary Paper {i}: {s}\n\n"
            
            if not context_text.strip():
                # Fallback to full source documents if summaries missing
                logger.info("No summaries found in output; using full source documents for judge context.")
                context_text = "\n\n".join(source_documents)
            
            # Prepare reference text for NLP metrics: use summaries if available (more appropriate comparison)
            # Comparing draft synthesis to summaries is more meaningful than to full source docs
            reference_text_for_nlp = context_text if context_text.strip() else "\n\n".join(source_documents)
            
            # Debug: Log what's being used for NLP metrics
            logger.info(f"DEBUG: NLP metrics reference source: {'summaries' if context_text.strip() else 'full documents'}")
            logger.info(f"DEBUG: Reference length: {len(reference_text_for_nlp)} chars")
            logger.info(f"DEBUG: Hypothesis (output) length: {len(output)} chars")

            judge_result = (
                evaluate_with_mlflow_judges(
                    task_description=task_description,
                    synthesis=output,
                    context=context_text,
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
            nlp_metrics = compute_nlp_metrics(reference_text_for_nlp, output)
            
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
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate Monolithic vs Ensemble agents for document synthesis"
    )
    parser.add_argument(
        "--agents-model",
        choices=["ollama", "gemini"],
        default="ollama",
        help="Model to use for agents: 'ollama' (default) or 'gemini'. Judges always use Gemini."
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Run in test mode (single paper, single task)"
    )
    args = parser.parse_args()
    
    test_mode = args.test
    agents_model = args.agents_model
    
    logger.info("="*60)
    if test_mode:
        logger.info("TEST MODE: Single Paper Evaluation")
    else:
        logger.info("Document Synthesis Evaluation: Monolithic vs Ensemble")
    logger.info(f"Agents Model: {agents_model.upper()}")
    logger.info("="*60)
    
    # Validate Gemini API key if using Gemini for agents
    if agents_model == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("ERROR: GEMINI_API_KEY environment variable is not set.")
            logger.error("Please set GEMINI_API_KEY in your .env file to use --agents-model=gemini")
            sys.exit(1)
    
    # Configuration
    doc_dir = "data/source_documents"
    task_file = "data/tasks/synthesis_tasks.json"
    doc_pattern = "paper_1.pdf" if test_mode else "paper_*.pdf"  # Single paper for test mode
    
    # Set model strings based on agents_model selection
    if agents_model == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        crewai_model = os.getenv("CREWAI_MODEL", f"openai/{model}")
    else:  # gemini
        model = "gemini-2.5-flash-lite"
        crewai_model = "gemini/gemini-2.5-flash-lite"
    
    # Judge model always uses Gemini
    judge_model = os.getenv("JUDGE_MODEL", "gemini:/gemini-2.5-flash-lite")

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
    run_experiment("monolithic", monolithic_agent, source_documents, tasks, judge_model, agents_model)
    
    # Run ensemble agent experiments (CrewAI Flow-based with recursive orchestration)
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE AGENT EVALUATION (CrewAI Flows with Orchestrator)")
    logger.info("="*60)
    ensemble_agent = EnsembleAgent(model=crewai_model, rate_limiter=rate_limiter)
    run_experiment("ensemble", ensemble_agent, source_documents, tasks, judge_model, agents_model)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info("\nResults logged to MLflow. To view:")
    logger.info("  mlflow ui")
    logger.info("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
