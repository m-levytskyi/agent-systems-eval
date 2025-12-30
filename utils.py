import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
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

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token for English text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4

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
