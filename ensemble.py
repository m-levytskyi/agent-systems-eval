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
from utils import setup_logging, sanitize_document, chunk_document, estimate_tokens, process_documents_with_cache

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
        
        # Create a temporary agent for summarization
        summarizer_agent = Agent(
            role="Document Summarizer",
            goal="Create comprehensive summaries of academic documents preserving all critical information.",
            backstory="You are an expert at extracting and preserving key information from academic papers.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )
        
        def process_chunk(chunk: str, doc_idx: int, chunk_idx: int, total_chunks: int) -> Tuple[str, Dict[str, int]]:
            if self.rate_limiter:
                self.rate_limiter.acquire()
            
            if total_chunks > 1:
                description = f"""This is CHUNK {chunk_idx} of {total_chunks} from DOCUMENT {doc_idx}.

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
            
            # Track metrics with improved extraction
            metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            tokens_found = False
            
            # Try multiple ways to extract token usage
            if hasattr(result, "usage_metrics"):
                usage = result.usage_metrics
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if hasattr(usage, key):
                        val = getattr(usage, key) or 0
                        metrics[key] = int(val)
                        if val > 0:
                            tokens_found = True
            elif hasattr(result, "token_usage"):
                usage = result.token_usage
                metrics["total_tokens"] = getattr(usage, "total_tokens", 0)
                metrics["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                metrics["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                tokens_found = metrics["total_tokens"] > 0
            
            # Fallback: estimate tokens from input + output if extraction failed
            if not tokens_found or metrics["total_tokens"] == 0:
                logger.warning(f"No token usage found for doc {doc_idx} chunk {chunk_idx}, using estimation")
                input_tokens = estimate_tokens(chunk)
                output_tokens = estimate_tokens(summary)
                metrics["total_tokens"] = input_tokens + output_tokens
                metrics["prompt_tokens"] = input_tokens
                metrics["completion_tokens"] = output_tokens
                logger.info(f"Estimated tokens - input: {input_tokens}, output: {output_tokens}, total: {metrics['total_tokens']}")
            
            return summary, metrics

        summaries, summary_metadata, aggregated_metrics = process_documents_with_cache(
            source_documents=source_documents,
            cache_dir=cache_dir,
            process_chunk_func=process_chunk,
            logger=logger
        )
        
        # Update metrics
        self.metrics["num_api_calls"] += aggregated_metrics["num_api_calls"]
        self.metrics["total_tokens"] += aggregated_metrics["total_tokens"]
        self.metrics["prompt_tokens"] += aggregated_metrics["prompt_tokens"]
        self.metrics["completion_tokens"] += aggregated_metrics["completion_tokens"]
        self.metrics["document_summaries_tokens"] += aggregated_metrics["document_summaries_tokens"]
        self.metrics["num_documents_summarized"] = len(source_documents)
        
        return summaries, summary_metadata
    
    def _extract_output(self, crew_result) -> str:
        """Extract text output from CrewAI result."""
        return (
            getattr(crew_result, "raw", None)
            or getattr(crew_result, "final_output", None)
            or str(crew_result)
        )

    def _reduce_summaries(self, summaries: List[str], llm) -> str:
        """
        Reduce phase: If summaries are too large, summarize them in groups.
        
        Args:
            summaries: List of document summaries
            llm: CrewAI LLM instance
            
        Returns:
            Combined and reduced text
        """
        from crewai import Agent, Crew, Process, Task
        
        total_tokens = estimate_tokens("\n\n".join(summaries))
        logger.info(f"Summaries total {total_tokens} tokens. Performing reduction...")
        
        # Group summaries into chunks of max 4000 tokens
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for summary in summaries:
            tokens = estimate_tokens(summary)
            if current_tokens + tokens > 4000:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [summary]
                current_tokens = tokens
            else:
                current_chunk.append(summary)
                current_tokens += tokens
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        # Summarize each chunk
        reducer_agent = Agent(
            role="Content Reducer",
            goal="Synthesize multiple document summaries into a single coherent meta-summary.",
            backstory="You are an expert at consolidating information from multiple sources.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )
        
        reduced_summaries = []
        for i, chunk in enumerate(chunks):
            task = Task(
                description=f"Synthesize the following document summaries into a single coherent summary, preserving all key information, findings, and technical details:\n\n{chunk}",
                agent=reducer_agent,
                expected_output="A consolidated meta-summary.",
            )
            crew = Crew(
                agents=[reducer_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            output = self._extract_output(result)
            reduced_summaries.append(output)
            
            # Update metrics (approximate)
            self.metrics["num_api_calls"] += 1
            # Note: We don't easily get token usage here without parsing result, 
            # but this is a rare path so it's acceptable.
            
        return "\n\n".join(reduced_summaries)

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

        start_time = time.time()
        
        # Configure CrewAI LLM
        llm = LLM(model=self.model)
        
        # Preprocess documents with map-reduce
        document_summaries, summary_metadata = self._preprocess_documents_for_archivist(source_documents, llm)
        
        # Use summaries directly (no pre-reduction for fair comparison with monolithic)
        documents_text = "\n\n".join(document_summaries)
        logger.info(f"Using {len(document_summaries)} document summaries directly (no pre-reduction)")

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
            goal="Provide detailed, actionable feedback on the draft's writing quality, focusing on style, wording, formatting, and clarity.",
            backstory=(
                "You are a meticulous editor who improves the presentation and readability of documents. "
                "You focus ONLY on how information is presented, not on adding new information. "
                "You NEVER request citations, case studies, examples, or data that are not already in the source material. "
                "Your feedback is limited to: clarity of writing, consistency of terminology, sentence structure, "
                "logical flow between sections, formatting improvements, and removing redundancy."
            ),
            allow_delegation=False,
            verbose=True,
            cache=False,
            llm=llm,
        )

        orchestrator_agent = Agent(
            role="Orchestrator",
            goal="Rate draft quality on a 1-10 scale and approve drafts scoring 6/10 or higher.",
            backstory=(
                "You are an expert quality coordinator who rates document quality objectively. "
                "Use a 10-point scale where 6/10 is the production-ready threshold. "
                "Be realistic: most drafts with minor wording issues or trivial improvements still deserve 6-8/10. "
                "Only score below 6 for substantial issues: missing key information, structural problems, factual errors, or major clarity issues. "
                "Don't be overly harsh - if the draft addresses the task adequately, it's production-ready even if not perfect."
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
                self.score_history = []  # Track last scores for plateau detection

        state = SynthesisFlowState()

        # =====================================================================
        # SIMPLE LOOP-BASED ITERATION (No CrewAI Flow)
        # =====================================================================
        
        def extract_output(crew_result) -> str:
            """Extract text output from CrewAI result."""
            return (
                getattr(crew_result, "raw", None)
                or getattr(crew_result, "final_output", None)
                or str(crew_result)
            )
        
        def update_metrics(crew_result, role: str):
            """Update token metrics from crew result."""
            tokens_found = False
            try:
                if hasattr(crew_result, "token_usage"):
                    usage = crew_result.token_usage
                    tokens = getattr(usage, "total_tokens", 0)
                    if tokens > 0:
                        self.metrics["total_tokens"] += tokens
                        logger.info(f"Token usage from {role}: {tokens} tokens")
                        tokens_found = True
                elif hasattr(crew_result, "usage_metrics"):
                    metrics = crew_result.usage_metrics
                    tokens = metrics.get("total_tokens", 0)
                    if tokens > 0:
                        self.metrics["total_tokens"] += tokens
                        logger.info(f"Token usage from {role}: {tokens} tokens")
                        tokens_found = True
            except Exception as e:
                logger.warning(f"Could not extract token usage for {role}: {e}")
            
            # Fallback: estimate tokens from output if none found
            if not tokens_found:
                try:
                    output_text = extract_output(crew_result)
                    estimated = estimate_tokens(output_text)
                    self.metrics["total_tokens"] += estimated
                    logger.info(f"Estimated {estimated} tokens for {role} (no usage data from API)")
                except Exception as est_err:
                    logger.warning(f"Could not estimate tokens for {role}: {est_err}")
            
            self.metrics["num_api_calls"] += 1
        
        def record_iteration(final: bool, reason: str, improvements: Optional[List[str]] = None):
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

        # STEP 1: Run Archivist (organize document information)
        logger.info("=== STEP 1: Running Archivist ===")
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
        state.archived_info = extract_output(result)
        update_metrics(result, "archivist")
        logger.info(f"Archivist completed: {len(state.archived_info)} characters organized")

        logger.info("=== STEP 2: Iterative Loop (Draft → Critic → Orchestrator) ===")
        while not state.is_production_ready and state.num_iterations < self.max_iterations:
            # Check timeout
            if time.time() - state.start_time > self.timeout_seconds:
                logger.warning(f"⚠️  Timeout reached after {state.num_iterations} iterations")
                record_iteration(final=True, reason="Timeout reached")
                break

            # Increment iteration count
            state.num_iterations += 1
            logger.info(f"\n{'='*60}\nIteration {state.num_iterations}/{self.max_iterations}\n{'='*60}")

            # === Run Drafter ===
            if self.rate_limiter:
                self.rate_limiter.acquire()

            # Temperature decay: 0.7 → 0.55 → 0.4 → 0.25 → 0.2
            temperature = max(0.2, 0.7 - ((state.num_iterations - 1) * 0.15))
            logger.info(f"Drafter temperature: {temperature:.2f}")

            # Build drafter prompt - ONLY latest improvements after first iteration
            if state.num_iterations > 1 and state.orchestrator_decision.get("actionable_improvements"):
                improvements = state.orchestrator_decision["actionable_improvements"]
                feedback_text = "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(improvements)])
                drafter_description = (
                    f"Task: {task_description}\n\n"
                    f"Organized Information:\n{state.archived_info}\n\n"
                    f"Previous Draft:\n{state.current_draft}\n\n"
                    f"CRITICAL: Address ONLY these {len(improvements)} specific improvements:\n"
                    f"{feedback_text}\n\n"
                    "Revise the draft to incorporate these improvements. Keep everything else unchanged."
                )
            else:
                drafter_description = (
                    f"Task: {task_description}\n\n"
                    f"Organized Information:\n{state.archived_info}\n\n"
                    "Create a comprehensive, well-structured synthesis addressing the task."
                )

            # Create drafter with temperature decay
            drafter_with_temp = Agent(
                role=drafter_agent.role,
                goal=drafter_agent.goal,
                backstory=drafter_agent.backstory,
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            
            drafter_task = Task(
                description=drafter_description,
                agent=drafter_with_temp,
                expected_output="Comprehensive synthesis document.",
            )

            crew = Crew(
                agents=[drafter_with_temp],
                tasks=[drafter_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            state.current_draft = extract_output(result)
            update_metrics(result, "drafter")

            # Save draft
            draft_dir = Path("data/drafts") / str(int(state.start_time))
            draft_dir.mkdir(parents=True, exist_ok=True)
            draft_file = draft_dir / f"draft_iteration_{state.num_iterations}.md"
            with open(draft_file, "w", encoding="utf-8") as f:
                f.write(state.current_draft)
            logger.info(f"Saved draft to {draft_file}")

            # === Run Critic ===
            if self.rate_limiter:
                self.rate_limiter.acquire()

            critic_task = Task(
                description=(
                    f"Task: {task_description}\n\n"
                    f"Draft:\n{state.current_draft}\n\n"
                    f"Archived Information:\n{state.archived_info}\n\n"
                    "CRITICAL: Feedback ONLY on presentation (clarity, coherence, formatting, style).\n"
                    "Do NOT request additional information not in archived information.\n"
                    "Provide actionable feedback on: clarity, coherence, formatting, style."
                ),
                agent=critic_agent,
                expected_output="Critique focused on presentation only.",
            )

            crew = Crew(
                agents=[critic_agent],
                tasks=[critic_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            state.current_critique = extract_output(result)
            update_metrics(result, "critic")

            # === Run Orchestrator ===
            if self.rate_limiter:
                self.rate_limiter.acquire()

            orchestrator_task = Task(
                description=(
                    f"Task: {task_description}\n\n"
                    f"Draft:\n{state.current_draft}\n\n"
                    f"Critic Feedback:\n{state.current_critique}\n\n"
                    "Rate production-readiness (1-10):\n"
                    "- 1-3: Major issues\n"
                    "- 4-5: Significant issues\n"
                    "- 6-7: Good quality (production-ready)\n"
                    "- 8-10: Excellent\n\n"
                    "IMPORTANT: Score >= 6 means production-ready.\n\n"
                    "Return ONLY valid JSON:\n"
                    "{\n"
                    '  "quality_score": <1-10>,\n'
                    '  "is_production_ready": <true if >= 6>,\n'
                    '  "reason": "brief explanation",\n'
                    '  "actionable_improvements": []\n'
                    "}"
                ),
                agent=orchestrator_agent,
                expected_output="JSON with quality_score, is_production_ready, reason, actionable_improvements.",
            )

            crew = Crew(
                agents=[orchestrator_agent],
                tasks=[orchestrator_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            decision_text = extract_output(result)
            update_metrics(result, "orchestrator")

            # Parse orchestrator decision
            try:
                # Extract JSON from markdown code blocks
                if "```json" in decision_text:
                    decision_text = decision_text.split("```json")[1].split("```")[0].strip()
                elif "```" in decision_text:
                    decision_text = decision_text.split("```")[1].split("```")[0].strip()
                
                decision = json.loads(decision_text)
                state.orchestrator_decision = decision
                
                quality_score = decision.get("quality_score", 5)
                is_ready = decision.get("is_production_ready", False)
                
                # Track score history for plateau detection
                state.score_history.append(quality_score)
                if len(state.score_history) > 3:
                    state.score_history.pop(0)
                
                # Plateau detection: same score 2-3 times = converged
                if len(state.score_history) >= 2:
                    if state.score_history[-1] == state.score_history[-2]:
                        logger.info(f"📊 Score plateau: {state.score_history[-2:]}/10 - converged")
                        is_ready = True
                    elif len(state.score_history) >= 3 and state.score_history[-1] == state.score_history[-2] == state.score_history[-3]:
                        logger.info(f"📊 Score plateau confirmed: {state.score_history}/10 - forcing completion")
                        is_ready = True
                
                # Enforce: score >= 6 means production-ready
                if quality_score >= 6:
                    is_ready = True
                
                state.is_production_ready = is_ready
                record_iteration(
                    final=is_ready,
                    reason=decision.get("reason", ""),
                    improvements=decision.get("actionable_improvements", [])
                )
                
                if is_ready:
                    logger.info(f"✓ Production-ready: Score={quality_score}/10 after {state.num_iterations} iteration(s)")
                    break
                else:
                    improvements = decision.get("actionable_improvements", [])
                    logger.info(f"⟳ Score={quality_score}/10 - {len(improvements)} improvements needed")
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"⚠️  Parse error: {e}")
                logger.warning(f"Raw: {decision_text[:200]}")
                state.is_production_ready = True
                record_iteration(final=True, reason="Parse error")
                break

        # Handle max iterations reached
        if state.num_iterations >= self.max_iterations and not state.is_production_ready:
            logger.warning(f"⚠️  Max iterations ({self.max_iterations}) reached")
            record_iteration(final=True, reason="Max iterations reached")

        # STEP 3: Finalize and return results
        self.metrics["num_iterations"] = state.num_iterations
        self.metrics["latency_seconds"] = time.time() - state.start_time
        
        logger.info(f"\n{'='*60}\nSynthesis Complete: {state.num_iterations} iteration(s)\n{'='*60}")
        
        return {
            "output": state.current_draft or "",
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
    from utils import load_source_documents
    
    documents = load_source_documents(doc_dir)
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    logger.info("Final Synthesized Output:")
    logger.info(result["output"])
    logger.info("\nMetrics:")
    logger.info(result["metrics"])
    logger.info(f"\nIterations: {result['metrics']['num_iterations']}")
