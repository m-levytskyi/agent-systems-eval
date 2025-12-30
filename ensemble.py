"""
Ensemble Agent: CrewAI-based multi-agent system with recursive orchestration and Map-Reduce.

This module implements a four-agent ensemble using CrewAI Flows:
- Archivist: Extracts and organizes key information from source documents (runs once with map-reduce)
- Drafter: Creates synthesis based on archivist's organization (iterative)
- Critic: Reviews and provides feedback on the draft (iterative)
- Orchestrator: Decides whether to iterate or finalize (recursive control)

The workflow uses CrewAI Flows API for recursive orchestration:
Archivist (map-reduce) → [Drafter → Critic → Orchestrator] → (loop or finalize)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from rate_limits import RequestRateLimiter
from utils import setup_logging, sanitize_document, chunk_document, estimate_tokens

logger = setup_logging("ensemble_agent")

class EnsembleAgent:
    """CrewAI Flow-based ensemble with recursive orchestration via Orchestrator agent.
    
    This implementation uses CrewAI Flows to enable iterative refinement where:
    1. Archivist runs once to organize source material
    2. Drafter creates a synthesis
    3. Critic provides feedback
    4. Orchestrator decides: continue iterating or finalize
    5. Loop continues until production-ready or limits reached (max 5 iterations, 30min timeout)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        rate_limiter: Optional[RequestRateLimiter] = None,
        max_iterations: int = 5,
        timeout_seconds: float = 1800.0,  # 30 minutes
    ) -> None:
        """
        Initialize the ensemble agent.
        
        Args:
            model: CrewAI model identifier (defaults to env CREWAI_MODEL or openai/qwen2.5:7b)
            rate_limiter: Optional rate limiter for API calls
            max_iterations: Maximum number of draft-critique-orchestrator iterations
            timeout_seconds: Maximum total time for synthesis (default 30 minutes)
        """
        self.model = model or os.getenv("CREWAI_MODEL", "openai/qwen2.5:7b")
        self.rate_limiter = rate_limiter
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        self.metrics: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_seconds": 0.0,
            "num_api_calls": 0,
            "archivist_tokens": 0,
            "drafter_tokens": 0,
            "critic_tokens": 0,
            "orchestrator_tokens": 0,
            "num_iterations": 0,
            "document_summaries_tokens": 0,
            "num_documents_summarized": 0,
        }

    def _preprocess_documents_for_archivist(
        self,
        source_documents: List[str],
        llm,
        cache_dir: str = "data/cache/ensemble_summaries",
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Map phase for Archivist: Summarize each document individually.
        Uses caching to allow restart on interruption.
        
        Args:
            source_documents: List of raw source documents
            llm: CrewAI LLM instance
            cache_dir: Directory to store cached summaries
            
        Returns:
            Tuple of (summaries, summary_metadata)
        """
        from crewai import Agent, Crew, Process, Task
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        summaries = []
        summary_metadata = []
        
        logger.info(f"{'='*60}")
        logger.info(f"ARCHIVIST MAP PHASE: Summarizing {len(source_documents)} documents")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"{'='*60}")
        
        # Create a temporary agent for summarization
        summarizer_agent = Agent(
            role="Document Summarizer",
            goal="Create comprehensive summaries of academic documents preserving all critical information.",
            backstory="You are an expert at extracting and preserving key information from academic papers.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )
        
        for doc_idx, doc in enumerate(source_documents, start=1):
            # Check cache first
            cache_file = cache_path / f"doc_{doc_idx}_summary.json"
            
            if cache_file.exists():
                logger.info(f"Document {doc_idx}/{len(source_documents)}: Loading from cache...")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    summaries.append(cached['summary'])
                    summary_metadata.append(cached['metadata'])
                    # Update metrics from cache
                    self.metrics["num_api_calls"] += cached['metadata'].get('num_api_calls', 0)
                    self.metrics["document_summaries_tokens"] += cached['metadata']['tokens_used']
                    
                    # Also update global token counts if available in metadata or estimate
                    tokens_used = cached['metadata']['tokens_used']
                    self.metrics["total_tokens"] += tokens_used
                    # Assume mostly prompt tokens for summarization (reading large doc)
                    # This is an approximation if exact split isn't saved
                    self.metrics["prompt_tokens"] += int(tokens_used * 0.8)
                    self.metrics["completion_tokens"] += int(tokens_used * 0.2)
                continue
            
            logger.info(f"Processing Document {doc_idx}/{len(source_documents)}...")
            
            # Sanitize
            sanitized_doc = sanitize_document(doc)
            original_tokens = estimate_tokens(doc)
            sanitized_tokens = estimate_tokens(sanitized_doc)
            tokens_saved = original_tokens - sanitized_tokens
            
            logger.info(f"  Original: ~{original_tokens:,} tokens")
            logger.info(f"  Sanitized: ~{sanitized_tokens:,} tokens (saved ~{tokens_saved:,})")
            
            # Chunk if necessary
            chunks = chunk_document(sanitized_doc, max_tokens=16000)
            logger.info(f"  Chunks: {len(chunks)}")
            
            # Summarize each chunk
            chunk_summaries = []
            chunk_metrics = []
            
            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_tokens = estimate_tokens(chunk)
                logger.info(f"    Chunk {chunk_idx}/{len(chunks)}: ~{chunk_tokens:,} tokens")
                
                if self.rate_limiter:
                    self.rate_limiter.acquire()
                
                if len(chunks) > 1:
                    description = f"""This is CHUNK {chunk_idx} of {len(chunks)} from DOCUMENT {doc_idx}.

Document Chunk:
{chunk}

Provide a comprehensive summary preserving all critical information, research questions, methodology, findings, and technical details."""
                else:
                    description = f"""This is DOCUMENT {doc_idx}.

Document:
{chunk}

Provide a comprehensive summary preserving all critical information, research questions, methodology, findings, and technical details."""
                
                task = Task(
                    description=description,
                    agent=summarizer_agent,
                    expected_output="Comprehensive summary with all critical information preserved.",
                )
                
                crew = Crew(
                    agents=[summarizer_agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                
                result = crew.kickoff()
                summary = self._extract_output(result)
                
                # Track metrics
                usage = getattr(result, "usage_metrics", None)
                metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if usage:
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if hasattr(usage, key):
                            metrics[key] = int(getattr(usage, key) or 0)
                
                chunk_summaries.append(summary)
                chunk_metrics.append(metrics)
                
                # Update global metrics
                self.metrics["num_api_calls"] += 1
                self.metrics["prompt_tokens"] += metrics["prompt_tokens"]
                self.metrics["completion_tokens"] += metrics["completion_tokens"]
                self.metrics["total_tokens"] += metrics["total_tokens"]
                self.metrics["document_summaries_tokens"] += metrics["total_tokens"]
            
            # Combine chunks
            if len(chunks) > 1:
                combined_summary = f"DOCUMENT {doc_idx} (multi-part summary):\n\n" + "\n\n---\n\n".join(chunk_summaries)
            else:
                combined_summary = f"DOCUMENT {doc_idx}:\n\n{chunk_summaries[0]}"
            
            summaries.append(combined_summary)
            metadata = {
                "doc_index": doc_idx,
                "original_length": len(doc),
                "sanitized_length": len(sanitized_doc),
                "num_chunks": len(chunks),
                "summary_length": len(combined_summary),
                "tokens_used": sum(m["total_tokens"] for m in chunk_metrics),
                "num_api_calls": len(chunks),
            }
            summary_metadata.append(metadata)
            
            # Save checkpoint
            with open(cache_file, 'w') as f:
                json.dump({
                    'summary': combined_summary,
                    'metadata': metadata
                }, f, indent=2)
            
            logger.info(f"  Summary: {len(combined_summary)} chars, {sum(m['total_tokens'] for m in chunk_metrics)} tokens")
            logger.info(f"  ✓ Checkpoint saved to {cache_file}")
        
        self.metrics["num_documents_summarized"] = len(source_documents)
        
        return summaries, summary_metadata
    
    def _extract_output(self, crew_result) -> str:
        """Extract text output from CrewAI result."""
        return (
            getattr(crew_result, "raw", None)
            or getattr(crew_result, "final_output", None)
            or str(crew_result)
        )

    def synthesize(self, source_documents: List[str], task_description: str) -> Dict[str, Any]:
        """
        Synthesize source documents using CrewAI Flow-based recursive orchestration.
        
        Args:
            source_documents: List of source document contents
            task_description: Description of the synthesis task
            
        Returns:
            Dictionary containing the final synthesis, iteration history, and metadata
        """
        # Lazy import so the rest of the repo runs without CrewAI installed.
        from crewai import Agent, Crew, LLM, Process, Task
        from crewai.flow.flow import Flow, listen, router, start

        start_time = time.time()
        
        # Configure CrewAI LLM
        llm = LLM(model=self.model)
        
        # Preprocess documents with map-reduce
        document_summaries, summary_metadata = self._preprocess_documents_for_archivist(source_documents, llm)
        documents_text = "\n\n".join(document_summaries)

        # Define agents
        archivist_agent = Agent(
            role="Archivist",
            goal="Extract and organize relevant information from the provided documents for the given task.",
            backstory="You are an expert archivist who creates structured knowledge bases from raw documents.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )

        drafter_agent = Agent(
            role="Drafter",
            goal="Write a comprehensive synthesis using the archivist's organized notes and any feedback from previous iterations.",
            backstory="You are an expert technical writer focused on clarity and structure. You incorporate feedback to improve your drafts.",
            allow_delegation=False,
            verbose=True,
            cache=False,
            llm=llm,
        )

        critic_agent = Agent(
            role="Critic",
            goal="Provide detailed, actionable feedback on the draft to improve completeness, coherence, and quality.",
            backstory="You are a meticulous editor who identifies specific improvements needed in drafts. Focus on actionable suggestions.",
            allow_delegation=False,
            verbose=True,
            cache=False,
            llm=llm,
        )

        orchestrator_agent = Agent(
            role="Orchestrator",
            goal="Evaluate critic feedback and decide if the draft is production-ready or needs another iteration.",
            backstory=(
                "You are an expert quality coordinator who makes strategic decisions about document readiness. "
                "You consider whether the critic's feedback identifies substantial issues that would significantly "
                "improve the output if addressed. Minor suggestions don't warrant iteration. "
                "Examples of production-ready: minor wording tweaks, optional enhancements, already comprehensive. "
                "Examples of needs-revision: missing key information, structural problems, factual errors, unclear sections."
            ),
            allow_delegation=False,
            verbose=True,
            cache=False,
            llm=llm,
        )

        # State for the flow
        class SynthesisFlowState:
            def __init__(self):
                self.archived_info = ""
                self.current_draft = ""
                self.current_critique = ""
                self.iteration_history = []
                self.num_iterations = 0
                self.start_time = start_time
                self.is_production_ready = False
                self.orchestrator_decision = {}
                self.document_summaries = document_summaries
                self.summary_metadata = summary_metadata

        state = SynthesisFlowState()

        # Define the CrewAI Flow
        class SynthesisFlow(Flow):
            """Recursive synthesis flow with orchestrator-controlled iterations."""

            @start()
            def run_archivist(self):
                """Step 1: Archivist organizes pre-processed document summaries."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                archivist_task = Task(
                    description=(
                        f"Task: {task_description}\n\n"
                        f"Document Summaries (pre-processed from {len(state.document_summaries)} documents):\n{documents_text}\n\n"
                        "Extract and organize key information relevant to the task. "
                        "Provide sections: Key Topics and Themes; Important Facts and Details; "
                        "Cross-document Connections; Relevant Context for the Task."
                    ),
                    agent=archivist_agent,
                    expected_output="Structured summary with key topics, facts, connections, and context organized clearly.",
                )

                crew = Crew(
                    agents=[archivist_agent],
                    tasks=[archivist_task],
                    process=Process.sequential,
                    verbose=True,
                )

                result = crew.kickoff()
                state.archived_info = self._extract_output(result)
                
                # Update metrics
                self._update_metrics(result, "archivist")
                
                return state.archived_info

            @router(run_archivist)
            def route_to_drafter(self):
                """Route from archivist to drafter."""
                return "run_drafter"

            @listen("run_drafter")
            def run_drafter(self, archived_info_or_route: str = None):
                """Step 2: Drafter creates synthesis (receives feedback from orchestrator if iterating)."""
                # Handle both initial call (archived_info) and iteration call (route string)
                if archived_info_or_route == "continue":
                    # This is a continue signal from orchestrator - use state
                    archived_info = state.archived_info
                elif archived_info_or_route:
                    # This is the initial call with archived_info
                    archived_info = archived_info_or_route
                else:
                    # Fallback to state
                    archived_info = state.archived_info
                    
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                # Check timeout
                if time.time() - state.start_time > self.timeout_seconds:
                    logger.warning(f"⚠️  Timeout reached after {state.num_iterations} iterations")
                    state.is_production_ready = True
                    return state.current_draft or "Timeout: synthesis incomplete"

                # Build drafter prompt with feedback if this is an iteration
                if state.num_iterations > 0 and state.orchestrator_decision.get("actionable_improvements"):
                    feedback_section = (
                        f"\n\nPREVIOUS DRAFT:\n{state.current_draft}\n\n"
                        f"CRITIC FEEDBACK:\n{state.current_critique}\n\n"
                        f"ACTIONABLE IMPROVEMENTS TO ADDRESS:\n" +
                        "\n".join([f"- {item}" for item in state.orchestrator_decision.get("actionable_improvements", [])])
                    )
                    drafter_description = (
                        f"Task: {task_description}\n\n"
                        f"Organized Information from Archivist:\n{archived_info}\n\n"
                        f"{feedback_section}\n\n"
                        "Revise the draft to address the actionable improvements while maintaining quality."
                    )
                else:
                    drafter_description = (
                        f"Task: {task_description}\n\n"
                        f"Organized Information from Archivist:\n{archived_info}\n\n"
                        "Create a comprehensive, well-structured synthesis that addresses the task."
                    )

                drafter_task = Task(
                    description=drafter_description,
                    agent=drafter_agent,
                    expected_output="A comprehensive, well-structured synthesis document.",
                )

                crew = Crew(
                    agents=[drafter_agent],
                    tasks=[drafter_task],
                    process=Process.sequential,
                    verbose=True,
                )

                result = crew.kickoff()
                state.current_draft = self._extract_output(result)
                
                # Save draft to file
                draft_dir = Path("data/drafts") / str(int(state.start_time))
                draft_dir.mkdir(parents=True, exist_ok=True)
                draft_file = draft_dir / f"draft_iteration_{state.num_iterations}.md"
                print(f"DEBUG: Saving draft iteration {state.num_iterations} to {draft_file}")
                with open(draft_file, "w", encoding="utf-8") as f:
                    f.write(state.current_draft)
                logger.info(f"Saved draft to {draft_file}")
                
                # Update metrics
                self._update_metrics(result, "drafter")
                
                return state.current_draft

            @listen(run_drafter)
            def run_critic(self, draft: str):
                """Step 3: Critic provides detailed feedback on the draft."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                critic_task = Task(
                    description=(
                        f"Task: {task_description}\n\n"
                        f"Draft to Review:\n{draft}\n\n"
                        f"Original Archived Information (for reference):\n{state.archived_info}\n\n"
                        "Provide detailed, actionable feedback on:\n"
                        "1. Completeness: Does it fully address the task requirements?\n"
                        "2. Coherence: Is it well-structured and logically flowing?\n"
                        "3. Accuracy: Is information correctly integrated from sources?\n"
                        "4. Quality: Is it professional and well-written?\n\n"
                        "For each area, identify specific issues and suggest concrete improvements."
                    ),
                    agent=critic_agent,
                    expected_output="Detailed critique with specific feedback on completeness, coherence, accuracy, and quality.",
                )

                crew = Crew(
                    agents=[critic_agent],
                    tasks=[critic_task],
                    process=Process.sequential,
                    verbose=True,
                )

                result = crew.kickoff()
                state.current_critique = self._extract_output(result)
                
                # Update metrics
                self._update_metrics(result, "critic")
                
                return state.current_critique

            @listen(run_critic)
            def run_orchestrator(self, critique: str):
                """Step 4: Orchestrator decides whether to iterate or finalize."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                state.num_iterations += 1

                # Check iteration limit
                if state.num_iterations >= self.max_iterations:
                    logger.warning(f"⚠️  Max iterations ({self.max_iterations}) reached")
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Max iterations reached")
                    return

                # Check timeout
                if time.time() - state.start_time > self.timeout_seconds:
                    logger.warning(f"⚠️  Timeout reached after {state.num_iterations} iterations")
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Timeout reached")
                    return

                orchestrator_task = Task(
                    description=(
                        f"Task Requirements: {task_description}\n\n"
                        f"Current Draft:\n{state.current_draft}\n\n"
                        f"Critic Feedback:\n{critique}\n\n"
                        "Evaluate whether this draft is production-ready or needs revision.\n\n"
                        "Consider:\n"
                        "- Are there SUBSTANTIAL issues that would significantly improve the output if fixed?\n"
                        "- Does the critic identify missing key information, structural problems, or factual errors?\n"
                        "- Or is the feedback mostly minor suggestions, optional enhancements, or polish?\n\n"
                        "Return ONLY a valid JSON object with this exact structure:\n"
                        "{\n"
                        '  "is_production_ready": true or false,\n'
                        '  "reason": "brief explanation of the decision",\n'
                        '  "actionable_improvements": ["list", "of", "specific", "changes", "needed"] or []\n'
                        "}\n\n"
                        "If is_production_ready is true, actionable_improvements should be empty."
                    ),
                    agent=orchestrator_agent,
                    expected_output="JSON object with is_production_ready, reason, and actionable_improvements fields.",
                )

                crew = Crew(
                    agents=[orchestrator_agent],
                    tasks=[orchestrator_task],
                    process=Process.sequential,
                    verbose=True,
                )

                result = crew.kickoff()
                decision_text = self._extract_output(result)
                
                # Update metrics
                self._update_metrics(result, "orchestrator")

                # Parse JSON decision
                try:
                    # Extract JSON from potential markdown code blocks
                    if "```json" in decision_text:
                        decision_text = decision_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in decision_text:
                        decision_text = decision_text.split("```")[1].split("```")[0].strip()
                    
                    decision = json.loads(decision_text)
                    state.orchestrator_decision = decision
                    is_ready = decision.get("is_production_ready", False)
                    
                    # Record this iteration
                    self._record_iteration(
                        final=is_ready,
                        reason=decision.get("reason", ""),
                        improvements=decision.get("actionable_improvements", [])
                    )
                    
                    if is_ready:
                        state.is_production_ready = True
                        logger.info(f"✓ Draft approved as production-ready after {state.num_iterations} iteration(s)")
                    else:
                        improvements = decision.get("actionable_improvements", [])
                        logger.info(f"⟳ Iteration {state.num_iterations}: Continuing with {len(improvements)} improvements")
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"⚠️  Failed to parse orchestrator decision: {e}")
                    logger.warning(f"Raw decision: {decision_text[:200]}")
                    # Default to finalize on parse error to avoid infinite loops
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Parse error in orchestrator decision")

            @router(run_orchestrator)
            def route_after_orchestrator(self):
                """Route based on orchestrator decision."""
                if state.is_production_ready:
                    return "finalize_output"
                return "run_drafter"

            @listen("finalize")
            def finalize_output(self):
                """Return final output."""
                return state.current_draft

            def _extract_output(self, crew_result) -> str:
                """Extract text output from CrewAI result."""
                return (
                    getattr(crew_result, "raw", None)
                    or getattr(crew_result, "final_output", None)
                    or str(crew_result)
                )

            def _update_metrics(self, crew_result, role: str):
                """Update token metrics from crew result."""
                # CrewAI's token usage tracking varies by version
                # Try multiple common attribute locations
                usage = None
                
                # Try direct attributes on crew_result
                for attr in ["usage_metrics", "token_usage", "usage", "tokens"]:
                    if hasattr(crew_result, attr):
                        usage = getattr(crew_result, attr)
                        if usage:
                            break
                
                # If no usage found, try to get it from tasks
                if not usage and hasattr(crew_result, "tasks_output"):
                    tasks = crew_result.tasks_output
                    if tasks and len(tasks) > 0:
                        task_result = tasks[0] if isinstance(tasks, list) else tasks
                        for attr in ["usage_metrics", "token_usage", "usage"]:
                            if hasattr(task_result, attr):
                                usage = getattr(task_result, attr)
                                if usage:
                                    break
                
                if usage:
                    # Handle both dict and object attribute access
                    def get_val(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key, 0) or 0
                        return getattr(obj, key, 0) or 0
                    
                    prompt = get_val(usage, "prompt_tokens")
                    completion = get_val(usage, "completion_tokens")
                    total = get_val(usage, "total_tokens") or (prompt + completion)
                    
                    self.metrics["prompt_tokens"] += int(prompt)
                    self.metrics["completion_tokens"] += int(completion)
                    self.metrics["total_tokens"] += int(total)
                    self.metrics[f"{role}_tokens"] += int(total)
                
                self.metrics["num_api_calls"] += 1

            def _record_iteration(self, final: bool, reason: str, improvements: Optional[List[str]] = None):
                """Record iteration in history."""
                state.iteration_history.append({
                    "iteration": state.num_iterations,
                    "draft": state.current_draft,
                    "critique": state.current_critique,
                    "decision": {
                        "is_production_ready": final,
                        "reason": reason,
                        "actionable_improvements": improvements or []
                    }
                })

        # Bind state and methods to flow instance
        flow = SynthesisFlow()
        flow.rate_limiter = self.rate_limiter
        flow.timeout_seconds = self.timeout_seconds
        flow.max_iterations = self.max_iterations
        flow.metrics = self.metrics

        # Execute the flow
        final_output = flow.kickoff()

        # Update final metrics
        self.metrics["num_iterations"] = state.num_iterations
        self.metrics["latency_seconds"] = time.time() - start_time
        
        # Ensure final refined draft is captured if iterations occurred
        # If orchestrator approved after revisions, capture that final state
        if state.num_iterations > 0 and state.is_production_ready:
            # Check if we need to add a final iteration entry
            last_iteration = state.iteration_history[-1] if state.iteration_history else None
            if last_iteration and not last_iteration["decision"]["is_production_ready"]:
                # The flow finalized but the last recorded iteration wasn't marked as final
                # This can happen if finalize was triggered by timeout/max iterations
                # Add a final entry capturing the approved state
                state.iteration_history.append({
                    "iteration": state.num_iterations,
                    "draft": state.current_draft,
                    "critique": state.current_critique,
                    "decision": {
                        "is_production_ready": True,
                        "reason": "Flow completed",
                        "actionable_improvements": []
                    }
                })
        
        return {
            "output": final_output or state.current_draft or "",
            "intermediate_outputs": {
                "archived_info": state.archived_info,
                "draft": state.current_draft,
                "iteration_history": state.iteration_history,
                "document_summaries": state.document_summaries,
                "summary_metadata": state.summary_metadata,
            },
            "metrics": self.metrics.copy(),
            "model": self.model,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    agent = EnsembleAgent()
    
    # Load sample documents
    doc_dir = os.path.join(os.path.dirname(__file__), "data", "source_documents")
    from PyPDF2 import PdfReader
    from pathlib import Path
    
    documents = []
    doc_path = Path(doc_dir)
    
    for filepath in sorted(doc_path.glob("*.pdf")):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        documents.append(text.strip())
    
    for filepath in sorted(doc_path.glob("*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            documents.append(f.read())
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    logger.info("Final Synthesized Output:")
    logger.info(result["output"])
    logger.info("\nMetrics:")
    logger.info(result["metrics"])
    logger.info(f"\nIterations: {result['metrics']['num_iterations']}")
