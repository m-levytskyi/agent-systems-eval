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
import mlflow.metrics.genai as genai_metrics
from dotenv import load_dotenv

from monolithic import MonolithicAgent
from ensemble import EnsembleAgent

load_dotenv()


def load_source_documents(doc_dir: str) -> List[str]:
    """Load all source documents from the specified directory."""
    documents = []
    doc_path = Path(doc_dir)
    
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
    
    Pricing (as of 2024, approximate):
    - GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
    - GPT-3.5-turbo: $0.001/1K prompt tokens, $0.002/1K completion tokens
    """
    pricing = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    }
    
    # Default to GPT-4 pricing if model not found
    model_key = "gpt-4"
    for key in pricing.keys():
        if key in model.lower():
            model_key = key
            break
    
    rates = pricing[model_key]
    prompt_cost = (metrics["prompt_tokens"] / 1000) * rates["prompt"]
    completion_cost = (metrics["completion_tokens"] / 1000) * rates["completion"]
    
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


def evaluate_with_llm_judge(task_description: str, synthesis: str, model: str) -> Dict[str, float]:
    """
    Evaluate synthesis quality using LLM-as-a-judge.
    
    Args:
        task_description: The synthesis task
        synthesis: The synthesized output
        model: Model to use for judging
        
    Returns:
        Dictionary of scores
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    judge_prompt = create_llm_judge_prompt(task_description, synthesis)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert evaluator of document synthesis quality."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.3
    )
    
    # Parse scores from response
    content = response.choices[0].message.content
    scores = {}
    
    for line in content.split('\n'):
        if ':' in line:
            for criterion in ['Completeness', 'Coherence', 'Accuracy', 'Quality', 'Overall']:
                if criterion.lower() in line.lower():
                    try:
                        # Extract the numeric score (X from "X/5")
                        score_part = line.split(':')[1].strip().split('/')[0].strip()
                        scores[criterion.lower()] = float(score_part)
                    except (IndexError, ValueError):
                        pass
    
    return scores


def run_experiment(
    agent_type: str,
    agent,
    source_documents: List[str],
    tasks: List[Dict[str, Any]],
    judge_model: str
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
            output = result["output"]
            metrics = result["metrics"]
            
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
            judge_scores = evaluate_with_llm_judge(task_description, output, judge_model)
            
            # Log quality scores
            for criterion, score in judge_scores.items():
                mlflow.log_metric(f"judge_{criterion}_score", score)
            
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


def main():
    """Main evaluation function."""
    print("="*60)
    print("Document Synthesis Evaluation: Monolithic vs Ensemble")
    print("="*60)
    
    # Configuration
    doc_dir = "data/source_documents"
    task_file = "data/tasks/synthesis_tasks.json"
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    judge_model = os.getenv("OPENAI_JUDGE_MODEL", model)
    
    # Load data
    print("\nLoading source documents and tasks...")
    source_documents = load_source_documents(doc_dir)
    tasks = load_tasks(task_file)
    
    print(f"Loaded {len(source_documents)} source documents")
    print(f"Loaded {len(tasks)} synthesis tasks")
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Run monolithic agent experiments
    print("\n" + "="*60)
    print("MONOLITHIC AGENT EVALUATION")
    print("="*60)
    monolithic_agent = MonolithicAgent(model=model)
    run_experiment("monolithic", monolithic_agent, source_documents, tasks, judge_model)
    
    # Run ensemble agent experiments
    print("\n" + "="*60)
    print("ENSEMBLE AGENT EVALUATION")
    print("="*60)
    ensemble_agent = EnsembleAgent(model=model)
    run_experiment("ensemble", ensemble_agent, source_documents, tasks, judge_model)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults logged to MLflow. To view:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
