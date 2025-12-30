"""
Monolithic Agent: Single LLM approach for document synthesis with Map-Reduce.

This module implements a two-phase approach:
- Phase 1 (Map): Summarize each document individually with sanitization
- Phase 2 (Reduce): Synthesize from the intermediate summaries
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from rate_limits import RequestRateLimiter

from llm.base import LLMClient
from llm.factory import create_llm_client, get_max_output_tokens
from utils import setup_logging, sanitize_document, chunk_document, estimate_tokens

load_dotenv()

logger = setup_logging("monolithic_agent")

class MonolithicAgent:
    """A single LLM agent that performs document synthesis using map-reduce."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
        llm_client: Optional[LLMClient] = None,
        rate_limiter: Optional[RequestRateLimiter] = None,
        context_window: int = 32768,
    ):
        """
        Initialize the monolithic agent.
        
        Args:
            model: Gemini model to use (defaults to env GEMINI_MODEL or gemini-2.5-pro)
            api_key: Gemini API key (defaults to env GEMINI_API_KEY or GOOGLE_API_KEY)
            context_window: Maximum context window in tokens (default: 32768)
        """
        # Backwards-compatible args: (model, api_key, client) were Gemini-specific.
        # New path: pass llm_client or set LLM_PROVIDER=ollama.
        self.llm_client: LLMClient
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # If a legacy Gemini client is passed, wrap it.
            if client is not None:
                from llm.gemini import GeminiClient

                self.llm_client = GeminiClient(client=client, model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
            else:
                # If caller passes an api_key, prefer it for Gemini.
                provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
                if provider == "gemini" and api_key:
                    from google import genai
                    from llm.gemini import GeminiClient

                    gemini_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
                    self.llm_client = GeminiClient(client=genai.Client(api_key=api_key), model=gemini_model)
                else:
                    # Factory reads env for provider + model.
                    self.llm_client = create_llm_client(provider=provider)

        # Keep `model` attribute for logging.
        self.model = model or os.getenv("OLLAMA_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.rate_limiter = rate_limiter
        self.max_output_tokens = get_max_output_tokens(default=4000)
        self.context_window = context_window
        self.metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_seconds": 0.0,
            "num_api_calls": 0,
            "document_summaries_tokens": 0,
            "num_documents_summarized": 0,
        }
    
    def _summarize_document_chunk(
        self,
        chunk: str,
        doc_index: int,
        chunk_index: int,
        total_chunks: int,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a high-fidelity summary for a single document chunk.
        Each API call is isolated (no conversation history).
        
        Args:
            chunk: Document chunk text
            doc_index: Document number (1-indexed)
            chunk_index: Chunk number (1-indexed)
            total_chunks: Total number of chunks for this document
            
        Returns:
            Tuple of (summary_text, token_metrics)
        """
        system_prompt = """You are an expert academic document analyzer. Your task is to create a comprehensive, high-fidelity summary that preserves all critical information from the provided document.

Guidelines:
- Extract key research questions, hypotheses, and objectives
- Summarize methodology, experimental design, and data collection
- Capture main findings, results, and statistical evidence
- Note important conclusions and implications
- Preserve technical details, terminology, and domain-specific concepts
- Maintain factual accuracy - do not add interpretations
- Be thorough but concise"""

        if total_chunks > 1:
            user_prompt = f"""This is CHUNK {chunk_index} of {total_chunks} from DOCUMENT {doc_index}.

Document Chunk:
{chunk}

Provide a comprehensive summary of this chunk, preserving all critical information, findings, and technical details."""
        else:
            user_prompt = f"""This is DOCUMENT {doc_index}.

Document:
{chunk}

Provide a comprehensive summary that preserves all critical information, findings, and technical details."""

        # Isolated API call (no conversation history)
        if self.rate_limiter:
            self.rate_limiter.acquire()
        
        llm_result = self.llm_client.chat(
            system=system_prompt,
            user=user_prompt,
            temperature=0.3,  # Lower temperature for factual summarization
            max_tokens=self.max_output_tokens,
        )
        
        # Track metrics
        token_metrics = {
            "prompt_tokens": int(llm_result.prompt_tokens or 0),
            "completion_tokens": int(llm_result.completion_tokens or 0),
            "total_tokens": int(llm_result.total_tokens or 0),
        }
        
        return llm_result.text or "", token_metrics
    
    def _map_phase(self, source_documents: List[str], cache_dir: str = "data/cache/summaries") -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Phase 1 (Map): Process each document individually to generate summaries.
        Uses caching to allow restart on interruption.
        
        Args:
            source_documents: List of raw source documents
            cache_dir: Directory to store cached summaries
            
        Returns:
            Tuple of (summaries, summary_metadata)
        """
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        summaries = []
        summary_metadata = []
        
        logger.info(f"{'='*60}")
        logger.info(f"MAP PHASE: Summarizing {len(source_documents)} documents")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"{'='*60}")
        
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
                continue
            
            logger.info(f"Processing Document {doc_idx}/{len(source_documents)}...")
            
            # Sanitize document
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
                
                summary, metrics = self._summarize_document_chunk(
                    chunk=chunk,
                    doc_index=doc_idx,
                    chunk_index=chunk_idx,
                    total_chunks=len(chunks),
                )
                
                chunk_summaries.append(summary)
                chunk_metrics.append(metrics)
                
                # Update global metrics
                self.metrics["num_api_calls"] += 1
                self.metrics["prompt_tokens"] += metrics["prompt_tokens"]
                self.metrics["completion_tokens"] += metrics["completion_tokens"]
                self.metrics["total_tokens"] += metrics["total_tokens"]
                self.metrics["document_summaries_tokens"] += metrics["total_tokens"]
            
            # Combine chunk summaries for this document
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
    
    def _reduce_phase(
        self,
        summaries: List[str],
        task_description: str,
    ) -> str:
        """
        Phase 2 (Reduce): Synthesize from the intermediate summaries.
        
        Args:
            summaries: List of document summaries from map phase
            task_description: Synthesis task description
            
        Returns:
            Final synthesis output
        """
        logger.info(f"{'='*60}")
        logger.info(f"REDUCE PHASE: Synthesizing from {len(summaries)} summaries")
        logger.info(f"{'='*60}")
        
        # Concatenate summaries
        summaries_text = "\n\n".join(summaries)
        estimated_tokens = estimate_tokens(summaries_text)
        logger.info(f"Total summary tokens: ~{estimated_tokens:,}")
        
        system_prompt = """You are an expert document synthesizer. Your task is to integrate information from multiple document summaries and create a comprehensive, well-structured synthesis that addresses the given task requirements.

Guidelines:
- Integrate information from all provided summaries
- Maintain accuracy and cite key points appropriately
- Create a coherent narrative that flows logically
- Ensure the output directly addresses the task description
- Be thorough and comprehensive"""

        user_prompt = f"""Task: {task_description}

Document Summaries:
{summaries_text}

Please synthesize the above summaries to complete the task. Provide a well-structured, comprehensive response that addresses all aspects of the task."""

        # Final synthesis call
        output_text = self._generate_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return output_text
    
    def synthesize(self, source_documents: List[str], task_description: str) -> Dict[str, Any]:
        """
        Synthesize source documents using map-reduce approach.
        
        Args:
            source_documents: List of source document contents
            task_description: Description of the synthesis task
            
        Returns:
            Dictionary containing the synthesized output and metadata
        """
        start_time = time.time()
        
        # Phase 1: Map - Summarize each document individually
        summaries, summary_metadata = self._map_phase(source_documents)
        
        # Phase 2: Reduce - Synthesize from summaries
        output_text = self._reduce_phase(summaries, task_description)
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["latency_seconds"] = end_time - start_time
        
        return {
            "output": output_text,
            "intermediate_outputs": {
                "document_summaries": summaries,
                "summary_metadata": summary_metadata,
            },
            "metrics": self.metrics.copy(),
            "model": self.model
        }

    def _generate_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempts: int = 2,
        backoff: float = 1.0,
    ) -> str:
        """Generate content with a short retry loop to mitigate empty responses."""
        last_text = ""
        for attempt in range(1, attempts + 1):
            if self.rate_limiter:
                self.rate_limiter.acquire()

            llm_result = self.llm_client.chat(
                system=system_prompt,
                user=user_prompt,
                temperature=0.7,
                max_tokens=self.max_output_tokens,
            )
            # Count this attempt
            self.metrics["num_api_calls"] += 1

            self.metrics["prompt_tokens"] += int(llm_result.prompt_tokens or 0)
            self.metrics["completion_tokens"] += int(llm_result.completion_tokens or 0)
            self.metrics["total_tokens"] += int(llm_result.total_tokens or 0)

            text = llm_result.text or ""
            if text.strip():
                return text

            logger.warning(f"⚠️  Empty response (attempt {attempt}/{attempts})")

            last_text = text
            if attempt < attempts:
                time.sleep(backoff)
        return last_text
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    agent = MonolithicAgent()
    
    # Load sample documents
    doc_dir = os.path.join(os.path.dirname(__file__), "data", "source_documents")
    documents = []
    for filename in sorted(os.listdir(doc_dir)):
        if filename.endswith(".txt") or filename.endswith(".pdf"):
            filepath = os.path.join(doc_dir, filename)
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            # PDF loading will be handled by evaluate.py
    
    # Example synthesis task
    task = "Write a comprehensive executive summary about artificial intelligence"
    
    result = agent.synthesize(documents, task)
    logger.info("Synthesized Output:")
    logger.info(result["output"])
    logger.info("\nMetrics:")
    logger.info(result["metrics"])
