"""
Ensemble Agent: CrewAI-based multi-agent system with recursive orchestration.

This module implements a four-agent ensemble using CrewAI Flows:
- Archivist: Extracts and organizes key information from source documents (runs once)
- Drafter: Creates synthesis based on archivist's organization (iterative)
- Critic: Reviews and provides feedback on the draft (iterative)
- Orchestrator: Decides whether to iterate or finalize (recursive control)

The workflow uses CrewAI Flows API for recursive orchestration:
Archivist → [Drafter → Critic → Orchestrator] → (loop or finalize)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from rate_limits import RequestRateLimiter


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
        }

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
        documents_text = "\n\n".join([f"DOCUMENT {i+1}:\n{doc}" for i, doc in enumerate(source_documents)])

        # Configure CrewAI LLM
        llm = LLM(model=self.model)

        # Define agents
        archivist_agent = Agent(
            role="Archivist",
            goal="Extract and organize relevant information from the provided documents for the given task.",
            backstory="You are an expert archivist who creates structured knowledge bases from raw documents.",
            allow_delegation=False,
            verbose=False,
            llm=llm,
        )

        drafter_agent = Agent(
            role="Drafter",
            goal="Write a comprehensive synthesis using the archivist's organized notes and any feedback from previous iterations.",
            backstory="You are an expert technical writer focused on clarity and structure. You incorporate feedback to improve your drafts.",
            allow_delegation=False,
            verbose=False,
            llm=llm,
        )

        critic_agent = Agent(
            role="Critic",
            goal="Provide detailed, actionable feedback on the draft to improve completeness, coherence, and quality.",
            backstory="You are a meticulous editor who identifies specific improvements needed in drafts. Focus on actionable suggestions.",
            allow_delegation=False,
            verbose=False,
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
            verbose=False,
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

        state = SynthesisFlowState()

        # Define the CrewAI Flow
        class SynthesisFlow(Flow):
            """Recursive synthesis flow with orchestrator-controlled iterations."""

            @start()
            def run_archivist(self):
                """Step 1: Archivist runs once to organize source material."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                archivist_task = Task(
                    description=(
                        f"Task: {task_description}\n\n"
                        f"Source Documents:\n{documents_text}\n\n"
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
                    verbose=False,
                )

                result = crew.kickoff()
                state.archived_info = self._extract_output(result)
                
                # Update metrics
                self._update_metrics(result, "archivist")
                
                return state.archived_info

            @listen(run_archivist)
            def run_drafter(self, archived_info: str):
                """Step 2: Drafter creates synthesis (receives feedback from orchestrator if iterating)."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                # Check timeout
                if time.time() - state.start_time > self.timeout_seconds:
                    print(f"⚠️  Timeout reached after {state.num_iterations} iterations")
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
                    verbose=False,
                )

                result = crew.kickoff()
                state.current_draft = self._extract_output(result)
                
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
                    verbose=False,
                )

                result = crew.kickoff()
                state.current_critique = self._extract_output(result)
                
                # Update metrics
                self._update_metrics(result, "critic")
                
                return state.current_critique

            @listen(run_critic)
            @router()
            def orchestrator_decision(self, critique: str):
                """Step 4: Orchestrator decides whether to iterate or finalize."""
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                state.num_iterations += 1

                # Check iteration limit
                if state.num_iterations >= self.max_iterations:
                    print(f"⚠️  Max iterations ({self.max_iterations}) reached")
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Max iterations reached")
                    return "finalize"

                # Check timeout
                if time.time() - state.start_time > self.timeout_seconds:
                    print(f"⚠️  Timeout reached after {state.num_iterations} iterations")
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Timeout reached")
                    return "finalize"

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
                    verbose=False,
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
                        print(f"✓ Draft approved as production-ready after {state.num_iterations} iteration(s)")
                        return "finalize"
                    else:
                        improvements = decision.get("actionable_improvements", [])
                        print(f"⟳ Iteration {state.num_iterations}: Continuing with {len(improvements)} improvements")
                        return "continue"
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️  Failed to parse orchestrator decision: {e}")
                    print(f"Raw decision: {decision_text[:200]}")
                    # Default to finalize on parse error to avoid infinite loops
                    state.is_production_ready = True
                    self._record_iteration(final=True, reason="Parse error in orchestrator decision")
                    return "finalize"

            @listen("continue")
            def continue_iteration(self):
                """Route back to drafter for another iteration."""
                # Trigger drafter again with current state
                return self.run_drafter(state.archived_info)

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
                usage = getattr(crew_result, "usage_metrics", None)
                if usage:
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if hasattr(usage, key):
                            value = int(getattr(usage, key) or 0)
                            self.metrics[key] += value
                            if key == "total_tokens":
                                self.metrics[f"{role}_tokens"] += value
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

        return {
            "output": final_output or state.current_draft or "",
            "intermediate_outputs": {
                "archived_info": state.archived_info,
                "draft": state.current_draft,
                "iteration_history": state.iteration_history,
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
    print("Final Synthesized Output:")
    print(result["output"])
    print("\nMetrics:")
    print(result["metrics"])
    print(f"\nIterations: {result['metrics']['num_iterations']}")
