import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from PyPDF2 import PdfReader

def setup_logging(name: str = "agent_eval") -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def sanitize_document(doc: str) -> str:
    """
    Remove references, bibliographies, and appendices from academic papers.
    These consume ~20% of tokens and provide zero semantic value.
    
    Args:
        doc: Raw document text
        
    Returns:
        Sanitized document text
    """
    # Remove references/bibliography sections (case-insensitive)
    doc = re.sub(
        r'\n\s*(references|bibliography|works cited|literature cited)\s*\n.*',
        '',
        doc,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Remove appendix sections
    doc = re.sub(
        r'\n\s*appendix\s+[a-z]?\s*[:\n].*',
        '',
        doc,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Remove standalone reference entries (e.g., "[1] Author et al...")
    doc = re.sub(r'\n\[\d+\]\s+[A-Z].*?\.\s*(?=\n|$)', '', doc)
    
    return doc.strip()

import tiktoken

def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (cl100k_base).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chunk_document(doc: str, max_tokens: int = 16000) -> List[str]:
    """
    Split document into chunks that don't exceed max_tokens.
    Uses simple character-based splitting with paragraph boundaries.
    
    Args:
        doc: Document text
        max_tokens: Maximum tokens per chunk (default: 16000 to prevent memory issues)
        
    Returns:
        List of document chunks
    """
    estimated_tokens = estimate_tokens(doc)
    
    if estimated_tokens <= max_tokens:
        return [doc]
    
    # Calculate how many chunks we need
    num_chunks = (estimated_tokens + max_tokens - 1) // max_tokens  # Ceiling division
    target_chars_per_chunk = len(doc) // num_chunks
    
    chunks = []
    current_pos = 0
    
    for i in range(num_chunks):
        if i == num_chunks - 1:
            # Last chunk: take everything remaining
            chunks.append(doc[current_pos:])
        else:
            # Find a good split point near the target
            target_pos = current_pos + target_chars_per_chunk
            
            # Look for paragraph break within ±2000 chars of target
            search_start = max(current_pos, target_pos - 2000)
            search_end = min(len(doc), target_pos + 2000)
            
            # Find the best paragraph break
            split_pos = doc.rfind('\n\n', search_start, search_end)
            
            if split_pos == -1 or split_pos <= current_pos:
                # No good paragraph break found, just use target
                split_pos = target_pos
            
            chunks.append(doc[current_pos:split_pos])
            current_pos = split_pos
    
    # Validate chunks don't exceed max_tokens
    validated_chunks = []
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        if chunk_tokens > max_tokens * 1.2:  # Allow 20% overflow
            # If still too large, split it in half
            mid = len(chunk) // 2
            split = chunk.rfind('\n\n', mid - 1000, mid + 1000)
            if split == -1:
                split = mid
            validated_chunks.append(chunk[:split])
            validated_chunks.append(chunk[split:])
        else:
            validated_chunks.append(chunk)
    
    return validated_chunks

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

def process_documents_with_cache(
    source_documents: List[str],
    cache_dir: str,
    process_chunk_func: Callable[[str, int, int, int], Tuple[str, Dict[str, int]]],
    logger: logging.Logger,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """
    Process documents with caching, sanitization, and chunking.
    
    Args:
        source_documents: List of raw document texts
        cache_dir: Directory for cache files
        process_chunk_func: Function to process a single chunk. 
                            Signature: (chunk, doc_idx, chunk_idx, total_chunks) -> (summary, metrics)
        logger: Logger instance
        
    Returns:
        Tuple of (summaries, metadata, aggregated_metrics)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    summaries = []
    summary_metadata = []
    aggregated_metrics = {
        "num_api_calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "document_summaries_tokens": 0
    }
    
    logger.info(f"{'='*60}")
    logger.info(f"PROCESSING DOCUMENTS: {len(source_documents)} documents")
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
                
                # Update metrics from cache with validation
                tokens_used = cached['metadata'].get('tokens_used', 0)
                if tokens_used == 0:
                    logger.warning(
                        f"⚠️  Cache for document {doc_idx} has tokens_used=0! "
                        f"This may indicate a bug in cache generation. "
                        f"Cache file: {cache_file}"
                    )
                
                aggregated_metrics["num_api_calls"] += cached['metadata'].get('num_api_calls', 0)
                aggregated_metrics["document_summaries_tokens"] += tokens_used
                aggregated_metrics["total_tokens"] += tokens_used
                # Estimate split if not available
                aggregated_metrics["prompt_tokens"] += int(tokens_used * 0.8)
                aggregated_metrics["completion_tokens"] += int(tokens_used * 0.2)
            continue
        
        logger.info(f"Processing Document {doc_idx}/{len(source_documents)}...")
        
        # Sanitize
        sanitized_doc = sanitize_document(doc)
        original_tokens = estimate_tokens(doc)
        sanitized_tokens = estimate_tokens(sanitized_doc)
        tokens_saved = original_tokens - sanitized_tokens
        
        logger.info(f"  Original: ~{original_tokens:,} tokens")
        logger.info(f"  Sanitized: ~{sanitized_tokens:,} tokens (saved ~{tokens_saved:,})")
        
        # Chunk
        chunks = chunk_document(sanitized_doc, max_tokens=16000)
        logger.info(f"  Chunks: {len(chunks)}")
        
        # Process chunks
        chunk_summaries = []
        chunk_metrics_list = []
        
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_tokens = estimate_tokens(chunk)
            logger.info(f"    Chunk {chunk_idx}/{len(chunks)}: ~{chunk_tokens:,} tokens")
            
            summary, metrics = process_chunk_func(chunk, doc_idx, chunk_idx, len(chunks))
            
            chunk_summaries.append(summary)
            chunk_metrics_list.append(metrics)
            
            # Update aggregated metrics
            aggregated_metrics["num_api_calls"] += 1
            aggregated_metrics["prompt_tokens"] += metrics.get("prompt_tokens", 0)
            aggregated_metrics["completion_tokens"] += metrics.get("completion_tokens", 0)
            aggregated_metrics["total_tokens"] += metrics.get("total_tokens", 0)
            aggregated_metrics["document_summaries_tokens"] += metrics.get("total_tokens", 0)
        
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
            "tokens_used": sum(m.get("total_tokens", 0) for m in chunk_metrics_list),
            "num_api_calls": len(chunks),
        }
        summary_metadata.append(metadata)
        
        # Save checkpoint
        with open(cache_file, 'w') as f:
            json.dump({
                'summary': combined_summary,
                'metadata': metadata
            }, f, indent=2)
        
        logger.info(f"  Summary: {len(combined_summary)} chars, {metadata['tokens_used']} tokens")
        logger.info(f"  ✓ Checkpoint saved to {cache_file}")
        
    return summaries, summary_metadata, aggregated_metrics
