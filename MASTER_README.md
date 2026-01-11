# MASTER README - Agent Systems Evaluation (Private Documentation)

**Author**: Mykhailo Levytskyi  
**Project**: agent-systems-eval  
**Last Updated**: January 10, 2026  
**Purpose**: Comprehensive private documentation with all technical details

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Implementation Details](#implementation-details)
4. [Configuration & Setup](#configuration--setup)
5. [Usage Patterns](#usage-patterns)
6. [Troubleshooting & Debugging](#troubleshooting--debugging)
7. [Performance Optimization](#performance-optimization)
8. [Technical Notes](#technical-notes)
9. [Future Enhancements](#future-enhancements)

---

## Project Overview

### Purpose
Empirical comparison of two agent architectures for document synthesis:
- **Monolithic Agent**: Single LLM approach (simple, fast)
- **Ensemble Agent**: Multi-agent system with recursive orchestration (complex, higher quality)

### Goals
1. Quantify performance differences (latency, cost, quality)
2. Demonstrate MLflow experiment tracking capabilities
3. Implement LLM-as-a-judge evaluation
4. Build production-ready agent systems with proper metrics

### Key Results
- Ensemble shows ~15-25% higher quality scores
- Ensemble takes ~2-3x longer (multiple iterations)
- Recursive orchestration enables adaptive quality control
- Map-reduce pattern essential for handling large documents

---

## Architecture Deep Dive

### Monolithic Agent (`monolithic.py`)

**Design Philosophy**: Simplicity and efficiency

**Flow**:
```
1. Map Phase: Sanitize + chunk + summarize each document independently
2. Reduce Phase: Synthesize summaries into final output
```

**Key Components**:
- `_summarize_document_chunk()`: Isolated API calls per chunk
- `_map_phase()`: Parallel document processing with caching
- `_reduce_phase()`: Final synthesis from summaries
- Cache location: `data/cache/summaries/`

**Advantages**:
- Fast (single pass)
- Low token usage
- Predictable behavior
- Easy to debug

**Limitations**:
- No iterative refinement
- No specialized processing stages
- Quality depends on single LLM call

### Ensemble Agent (`ensemble.py`)

**Design Philosophy**: Quality through specialization and iteration

**Architecture**: CrewAI Flows with 4 specialized agents

**Agents**:
1. **Archivist** (runs once):
   - Role: Document analysis and organization
   - Map-reduce: Summarizes each document independently
   - Reduce: Consolidates into organized structure
   - Output: `organized_info` for downstream agents

2. **Drafter** (iterative):
   - Role: Create synthesis draft
   - Input: Organized info + task description + previous critique
   - Output: Draft synthesis

3. **Critic** (iterative):
   - Role: Quality assurance and feedback
   - Input: Current draft + task requirements
   - Output: Detailed critique with improvement suggestions

4. **Orchestrator** (recursive control):
   - Role: Decision-making and iteration control
   - Input: Draft + critique
   - Output: "continue" or final draft
   - Logic: Evaluates if production-ready or needs refinement

**Flow Pattern**:
```
Archivist (once)
    ↓
Drafter → Critic → Orchestrator
    ↑                   ↓
    └──── continue ─────┘
            OR
        final draft (terminate)
```

**CrewAI Flow Implementation Details**:

**Key Methods**:
- `start_archivist()`: Initial kickoff, runs once
- `run_drafter()`: Creates/refines draft
- `run_critic()`: Provides feedback
- `run_orchestrator()`: Decides continue/finalize
- `route_after_orchestrator()`: Router that controls flow

**State Management**:
```python
class EnsembleState:
    organized_info: str         # From archivist (immutable)
    current_draft: str          # Latest draft
    current_critique: str       # Latest feedback
    iteration_count: int        # Current iteration
    is_production_ready: bool   # Orchestrator decision
    task_description: str       # Original task
    run_id: int                 # Timestamp for caching
```

**Critical Flow Control Pattern**:
```python
# Orchestrator returns final draft when ready
if is_production_ready:
    return state.current_draft  # Flow ends

# Router returns None to terminate
@router(run_orchestrator)
def route_after_orchestrator(self):
    if state.is_production_ready:
        return None  # No more routing = flow ends
    return "run_drafter"  # Continue iteration
```

**Why This Works** (from CrewAI Flows docs):
- "The final output is determined by the last method that completes"
- `kickoff()` returns output of final method
- Returning `None` from router = no next method = flow terminates
- When orchestrator returns final draft + router returns None = clean termination

**Advantages**:
- Higher quality through iteration
- Adaptive behavior (orchestrator decides when ready)
- Clear separation of concerns
- Full iteration history logged

**Limitations**:
- Higher latency (multiple iterations)
- More token usage
- More complex to debug
- Requires CrewAI Flows

### Map-Reduce Implementation

**Both agents use map-reduce for document processing**:

**Map Phase** (`_map_phase` or `_preprocess_documents_for_archivist`):
```python
1. Sanitize document (remove references, bibliographies, appendices)
2. Chunk if > max_tokens (default: 16000 tokens)
3. For each chunk:
   - Generate comprehensive summary
   - Cache to disk (JSON)
4. Return: (summaries, metadata, metrics)
```

**Reduce Phase** (`_reduce_phase` or `_reduce_summaries`):
```python
1. Combine all document summaries
2. Synthesize into coherent organization
3. Return: final organized output
```

**Sanitization** (`utils.sanitize_document`):
```python
# Remove ~20% of tokens with zero semantic value
- References/Bibliography sections
- Appendices
- Standalone reference entries [1], [2], etc.
```

**Chunking** (`utils.chunk_document`):
```python
# Split at paragraph boundaries
# Target: max_tokens per chunk (default 16000)
# Validation: ensure chunks don't exceed 1.2x max_tokens
```

**Caching** (`utils.process_documents_with_cache`):
```python
# Cache location: data/cache/summaries/ or ensemble_summaries/
# Cache format: JSON with summary + metrics + metadata
# Cache key: MD5 hash of sanitized document
# Resume: load from cache if hash matches
```

---

## Implementation Details

### LLM Client Abstraction (`llm/`)

**Factory Pattern** (`llm/factory.py`):
```python
def create_llm_client(provider: str) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(...)
    elif provider == "gemini":
        return GeminiClient(...)
```

**Base Interface** (`llm/base.py`):
```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Returns: {"text": str, "usage": {...}}"""
```

**Implementations**:
- `llm/ollama.py`: Ollama client with OpenAI-compatible API
- `llm/gemini.py`: Google Gemini client

**Usage**:
```python
client = create_llm_client("ollama")
result = client.generate(system_prompt, user_prompt)
text = result["text"]
tokens = result["usage"]["total_tokens"]
```

### Rate Limiting (`rate_limits.py`)

**Purpose**: Avoid hitting API quotas (primarily for Gemini free tier)

**Implementation**:
```python
class RequestRateLimiter:
    def __init__(self, max_per_minute=10, max_per_day=20):
        self._recent_calls: deque[float] = deque()
        self._day_count = 0
    
    def acquire(self):
        # Blocks until slot available
        # Raises if daily limit reached
```

**Usage**:
```python
rate_limiter = RequestRateLimiter(max_per_minute=10, max_per_day=50)
# Before each API call:
rate_limiter.acquire()
```

**Note**: Only needed for remote providers with strict limits. Ollama has no limits.

### MLflow Integration (`evaluate.py`)

**Experiments**:
- `document_synthesis_monolithic`: Monolithic agent runs
- `document_synthesis_ensemble`: Ensemble agent runs

**Logged Metrics**:
```python
# Process metrics
mlflow.log_metric("latency_seconds", ...)
mlflow.log_metric("total_tokens", ...)
mlflow.log_metric("estimated_cost_usd", ...)
mlflow.log_metric("num_iterations", ...)  # Ensemble only

# Quality metrics (LLM-as-a-judge)
mlflow.log_metric("groundedness_score", ...)
mlflow.log_metric("instruction_adherence_score", ...)
mlflow.log_metric("completeness_score", ...)

# NLP metrics
mlflow.log_metric("bertscore_f1", ...)
mlflow.log_metric("rouge1_fmeasure", ...)
```

**Logged Artifacts**:
```python
mlflow.log_text(final_synthesis, "synthesis.md")
mlflow.log_dict(agent_metrics, "metrics.json")
# Ensemble only:
mlflow.log_text(iteration_history, "iterations.md")
```

**Judge Configuration** (`evaluate.py`):
```python
# Uses MLflow GenAI make_judge
groundedness_judge = make_judge(
    name="groundedness",
    model="openai:/qwen2.5:7b",  # Via Ollama OpenAI-compat
    instructions="...",  # Detailed grading criteria
)

# Evaluation
results = evaluate_with_mlflow_judges(
    task_description=task,
    synthesis=output,
    context=context,
    judge_model="openai:/qwen2.5:7b",
)
```

**NLP Metrics** (`evaluate.py`):
```python
def compute_nlp_metrics(reference: str, hypothesis: str):
    # BERTScore
    bert_scores = bert_score.score([hypothesis], [reference], ...)
    
    # ROUGE
    rouge_scorer = RougeScorer(['rouge1', 'rougeL'], ...)
    rouge_scores = rouge_scorer.score(reference, hypothesis)
    
    return {
        "bertscore_precision": ...,
        "bertscore_recall": ...,
        "bertscore_f1": ...,
        "rouge1_fmeasure": ...,
        "rougeL_fmeasure": ...,
    }
```

### Utilities (`utils.py`)

**Key Functions**:
```python
setup_logging(name: str) -> Logger
sanitize_document(doc: str) -> str
estimate_tokens(text: str) -> int  # tiktoken cl100k_base
chunk_document(doc: str, max_tokens: int) -> List[str]
load_source_documents(doc_dir: str) -> List[str]  # PDF + txt
process_documents_with_cache(...) -> (summaries, metadata, metrics)
```

**Document Loading**:
- Supports PDF (via PyPDF2) and text files
- Extracts text from all pages
- Returns list of document strings

**Caching Strategy**:
- Hash sanitized document → MD5
- Cache file: `{cache_dir}/doc_{idx}_summary.json`
- Resume: If hash matches, load from cache
- Invalidation: Manual (delete cache files)

---

## Configuration & Setup

### Environment Variables (.env)

**Required**:
```bash
# LLM Provider Selection
LLM_PROVIDER=ollama  # or gemini

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_NUM_CTX=32768  # Context window
MAX_OUTPUT_TOKENS=4000

# MLflow Judges (via Ollama OpenAI-compat)
JUDGE_MODEL=openai:/qwen2.5:7b
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama  # Dummy value for compat

# CrewAI Ensemble
CREWAI_MODEL=openai/qwen2.5:7b
MAX_ITERATIONS=5
TIMEOUT_SECONDS=1800  # 30 minutes
```

**Optional**:
```bash
# Rate limiting (for remote providers)
MAX_RPM=10  # Max requests per minute (0 = disabled)
MAX_RPD=50  # Max requests per day (0 = disabled)

# Google Gemini (if LLM_PROVIDER=gemini)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro
```

### Ollama Setup

**Installation**:
```bash
# Download from ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5:7b
```

**Context Window Configuration**:
```bash
# Critical: Set context window to 32k tokens
export OLLAMA_NUM_CTX=32768

# Start Ollama
ollama serve
```

**Verify**:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Test",
  "options": {"num_ctx": 32768}
}'
```

**Note**: The implementation passes `num_ctx: 32768` in every API call to override server defaults.

### Python Dependencies

**Core**:
```
python >= 3.10
crewai >= 0.20.0  # Flows support
mlflow >= 2.10.0
google-generativeai >= 0.3.0  # If using Gemini
python-dotenv
```

**NLP Metrics**:
```
bert-score
rouge-score
tiktoken
```

**Utilities**:
```
PyPDF2  # PDF parsing
requests
```

**Install**:
```bash
pip install -r requirements.txt
```

---

## Usage Patterns

### Quick Test Mode (Recommended First Run)

**Purpose**: Verify setup works with minimal API usage

```bash
python evaluate.py --test
```

**What it does**:
- Uses only 1 paper (paper_1.pdf)
- Runs only first task
- Completes in ~5-10 minutes
- Tests both agents
- Logs to MLflow

**Expected output**:
```
Processing 1 documents...
Running task 1/1: Write executive summary...
Monolithic agent: 8.3s, 2,451 tokens
Ensemble agent: 23.7s, 6,892 tokens, 3 iterations
EVALUATION COMPLETE
```

### Full Evaluation

**Purpose**: Complete comparison across all tasks

```bash
python evaluate.py
```

**What it does**:
- Processes all 10 papers
- Runs all 3 tasks
- Takes 1-2 hours
- Generates 6 MLflow runs (2 agents × 3 tasks)

**Expected output**:
```
Processing 10 documents...
Running task 1/3...
  Monolithic: 45.2s, 18,234 tokens
  Ensemble: 127.8s, 52,891 tokens, 4 iterations
Running task 2/3...
  ...
EVALUATION COMPLETE
```

### Individual Agent Testing

**Monolithic**:
```bash
python monolithic.py
```

**Ensemble**:
```bash
python ensemble.py
```

**Output**: Demonstrates agent on sample task without MLflow tracking

### Custom Evaluation

**1. Add Documents**:
```bash
cp your_paper.pdf data/source_documents/
```

**2. Define Tasks** (`data/tasks/synthesis_tasks.json`):
```json
{
  "task_id": "custom_1",
  "task_description": "Synthesize methodology sections...",
  "expected_elements": [
    "Research design overview",
    "Data collection methods",
    "Analysis approach"
  ]
}
```

**3. Run**:
```bash
python evaluate.py
```

### Viewing Results

**Start MLflow UI**:
```bash
mlflow ui
# Open http://localhost:5000
```

**Navigate**:
1. Experiments → Select experiment
2. Compare runs across agents
3. View metrics table
4. Download artifacts

**Useful Views**:
- Parallel Coordinates: Compare metrics across runs
- Scatter Plot: latency vs quality
- Table View: Sort by metric

---

## Troubleshooting & Debugging

### Common Issues

#### 1. Infinite Loop in Ensemble Agent

**Symptom**: Ensemble runs forever, logs show repeated iterations

**Root Cause**: Router pattern issue (fixed in FLOW_FIX_SUMMARY.md)

**Fix Applied**:
```python
# Orchestrator returns final draft when ready
if is_production_ready:
    return state.current_draft  # Not "finalize"

# Router returns None to terminate
@router(run_orchestrator)
def route_after_orchestrator(self):
    if state.is_production_ready:
        return None  # Not "finalize" label
    return "run_drafter"
```

**Verify**:
```bash
grep "production-ready\|Max iterations" test_output.log
# Should see termination after approval
```

#### 2. Ollama Context Window Too Small

**Symptom**: `context length exceeded` errors

**Fix**:
```bash
# Set in .env
OLLAMA_NUM_CTX=32768

# Restart Ollama
pkill ollama
ollama serve
```

**Verify**:
```python
# Implementation passes this in every call:
options = {"num_ctx": 32768}
```

#### 3. Cache Not Working

**Symptom**: Re-processing documents on every run

**Debug**:
```bash
ls -la data/cache/summaries/
# Should see doc_N_summary.json files

# Check hash calculation
python -c "
import hashlib
from utils import sanitize_document
doc = open('data/source_documents/paper_1.pdf', 'rb').read()
# ... (hash calculation)
"
```

**Fix**: Ensure cache directory exists and is writable

#### 4. MLflow Judge Failures

**Symptom**: `Invalid API key` or judge evaluation fails

**Fix**:
```bash
# Ensure OpenAI-compat endpoint is configured
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama  # Any non-empty value works

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

#### 5. CrewAI Import Errors

**Symptom**: `No module named 'crewai'` or version mismatch

**Fix**:
```bash
pip install --upgrade crewai>=0.20.0
# CrewAI Flows requires v0.20.0+
```

### Debugging Tools

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Monitor MLflow**:
```bash
# Watch MLflow runs directory
watch -n 1 "ls -la mlruns/*/meta.yaml"
```

**Check CrewAI Flow State**:
```python
# In ensemble.py, add:
logger.info(f"State: {state.__dict__}")
```

**Validate Documents**:
```python
from utils import load_source_documents, estimate_tokens
docs = load_source_documents("data/source_documents")
for i, doc in enumerate(docs, 1):
    print(f"Doc {i}: {estimate_tokens(doc)} tokens")
```

---

## Performance Optimization

### Token Usage Optimization

**1. Document Sanitization** (saves ~20% tokens):
```python
sanitized = sanitize_document(raw_doc)
# Removes: references, bibliographies, appendices
```

**2. Chunking Strategy**:
```python
# Default: 16000 tokens per chunk
# Reduces memory usage, enables parallel processing
chunks = chunk_document(doc, max_tokens=16000)
```

**3. Cache Summaries**:
```python
# Avoid re-summarizing same documents
# Cache hit = instant load
# Cache miss = summarize + save
```

### Latency Optimization

**1. Parallel Document Processing**:
```python
# Map phase processes each document independently
# Can be parallelized with ThreadPoolExecutor
# (Not currently implemented - future enhancement)
```

**2. Reduce Iterations**:
```python
# Ensemble: Set lower max_iterations
MAX_ITERATIONS=3  # Instead of 5
```

**3. Skip Ensemble for Simple Tasks**:
```python
# Use monolithic for straightforward tasks
# Reserve ensemble for complex synthesis
```

### Cost Optimization (Gemini)

**1. Use Rate Limiter**:
```python
rate_limiter = RequestRateLimiter(
    max_per_minute=10,  # Free tier limit
    max_per_day=50,
)
```

**2. Enable Caching**:
```bash
# Never re-process same documents
# Saves ~80% of API calls on reruns
```

**3. Test Mode First**:
```bash
python evaluate.py --test
# Test with 1 paper before full run
```

### Ollama Performance

**1. GPU Acceleration**:
```bash
# Ensure CUDA available
nvidia-smi

# Ollama will auto-detect and use GPU
```

**2. Model Selection**:
```bash
# Faster models for development:
ollama pull qwen2.5:3b  # Smaller, faster

# Higher quality for production:
ollama pull qwen2.5:14b  # Larger, slower
```

**3. Concurrent Requests**:
```bash
# Ollama can handle multiple requests
# Limited by available VRAM
```

---

## Technical Notes

### CrewAI Flow Patterns (Lessons Learned)

**1. Terminal Methods Return Final Output**:
```python
# When ready to end flow, return the final result
if ready_to_finish:
    return final_output  # Not a label/string
```

**2. Routers Signal Termination with None**:
```python
@router(some_method)
def route_next(self):
    if should_stop:
        return None  # No next method = flow ends
    return "next_method_name"
```

**3. State Management**:
```python
# Use shared state object, not instance variables
state.field = value  # Not self.field

# State persists across all methods in flow
```

**4. No Explicit Finalize Needed**:
```python
# DON'T do this:
@listen("finalize")
def finalize_output(self):
    return state.result

# Instead: Return final output from terminal method
```

**5. Last Method Determines Output**:
```python
# CrewAI Flow.kickoff() returns output of last method
# Plan your flow so terminal method has final output
```

### Map-Reduce Pattern (Best Practices)

**1. Isolated Map Operations**:
```python
# Each document processed independently
# No shared state between map calls
# Enables caching and parallel processing
```

**2. Reduce Consolidation**:
```python
# Combine map results into coherent output
# This is where synthesis happens
```

**3. Chunk Validation**:
```python
# Always validate chunk size
# Chunks can exceed max_tokens due to paragraph boundaries
# Add 20% buffer for safety
if estimate_tokens(chunk) > max_tokens * 1.2:
    # Re-chunk more aggressively
```

### LLM-as-a-Judge Guidelines

**1. Detailed Instructions**:
```python
# Provide clear grading criteria
# Include examples of each score level
# Define edge cases
```

**2. Reference vs Reference-Free**:
```python
# Reference-free: Judge quality without ground truth
# Reference-based: Compare to reference output
# This project uses reference-free
```

**3. Score Normalization**:
```python
def _score_value_to_float(value):
    # Handle various response formats
    # Convert "yes"/"no" to 1.0/0.0
    # Convert "fully"/"partially"/"not" to 1.0/0.5/0.0
```

### MLflow Best Practices

**1. Experiment Organization**:
```python
# Separate experiments per agent type
# Enables clean comparison
mlflow.set_experiment("document_synthesis_monolithic")
```

**2. Run Naming**:
```python
# Descriptive run names
mlflow.start_run(run_name=f"{agent_type}_{task_id}")
```

**3. Artifact Logging**:
```python
# Log all intermediate outputs
# Enables debugging and analysis
mlflow.log_text(draft, "iteration_N_draft.md")
```

**4. Metric Consistency**:
```python
# Use same metric names across runs
# Enables comparison in UI
mlflow.log_metric("latency_seconds", ...)
```

---

## Future Enhancements

### High Priority

**1. Parallel Document Processing**:
```python
# Use ThreadPoolExecutor for map phase
# 5-10x speedup for multi-document synthesis
from concurrent.futures import ThreadPoolExecutor
```

**2. Adaptive Chunking**:
```python
# Smart chunking based on document structure
# Respect section boundaries, not just paragraphs
```

**3. Streaming Output**:
```python
# Stream ensemble iterations to user
# Provide real-time feedback during synthesis
```

### Medium Priority

**4. Custom Judge Models**:
```python
# Train domain-specific judges
# Fine-tune on expert evaluations
```

**5. Multi-Modal Support**:
```python
# Support images, tables, charts in PDFs
# Extract and describe visual elements
```

**6. Agent Comparison Dashboard**:
```python
# Custom Streamlit/Gradio UI
# Interactive comparison and visualization
```

### Low Priority

**7. A/B Testing Framework**:
```python
# Automated A/B tests for prompt variations
# Statistical significance testing
```

**8. Cost Prediction**:
```python
# Estimate cost before running
# Token usage prediction based on document size
```

**9. Export to Production Formats**:
```python
# Generate LaTeX, DOCX, HTML from synthesis
# Professional formatting templates
```

---

## Appendices

### A. File Structure Reference

```
agent-systems-eval/
├── README.md                    # Public documentation (streamlined)
├── MASTER_README.md            # This file (private, comprehensive)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .env                       # Your config (gitignored)
├── .gitignore                 # Git ignore rules
│
├── monolithic.py              # Monolithic agent
├── ensemble.py                # Ensemble agent (CrewAI Flows)
├── evaluate.py                # MLflow evaluation framework
├── utils.py                   # Shared utilities
├── rate_limits.py            # Rate limiter for API calls
│
├── llm/                       # LLM client abstraction
│   ├── __init__.py
│   ├── base.py               # Abstract interface
│   ├── factory.py            # Client factory
│   ├── ollama.py             # Ollama implementation
│   ├── gemini.py             # Gemini implementation
│   └── types.py              # Type definitions
│
├── data/
│   ├── source_documents/     # Input PDFs
│   ├── tasks/                # Task definitions (JSON)
│   ├── cache/                # Cached summaries
│   │   ├── summaries/        # Monolithic cache
│   │   └── ensemble_summaries/  # Ensemble cache
│   └── drafts/               # Iteration history (ensemble)
│
└── mlruns/                   # MLflow tracking data
    ├── 0/                    # Default experiment
    ├── {experiment_id}/      # Per-experiment runs
    └── models/               # Registered models (unused)
```

### B. Environment Variables Quick Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | LLM provider selection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model name |
| `OLLAMA_NUM_CTX` | `32768` | Context window (tokens) |
| `MAX_OUTPUT_TOKENS` | `4000` | Max output length |
| `JUDGE_MODEL` | `openai:/qwen2.5:7b` | MLflow judge model |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compat endpoint |
| `OPENAI_API_KEY` | `ollama` | Dummy key for compat |
| `CREWAI_MODEL` | `openai/qwen2.5:7b` | CrewAI model ID |
| `MAX_ITERATIONS` | `5` | Max ensemble iterations |
| `TIMEOUT_SECONDS` | `1800` | Max synthesis time (30m) |
| `MAX_RPM` | `0` | Rate limit (req/min) |
| `MAX_RPD` | `0` | Rate limit (req/day) |

### C. Metric Definitions

**Process Metrics**:
- `latency_seconds`: Total wall-clock time
- `total_tokens`: Sum of prompt + completion tokens
- `prompt_tokens`: Input tokens
- `completion_tokens`: Output tokens
- `num_api_calls`: Count of LLM API calls
- `estimated_cost_usd`: `0.0` for Ollama, estimated for Gemini
- `num_iterations`: Ensemble only, iteration count

**Quality Metrics (LLM Judge, 0-5 scale)**:
- `groundedness_score`: Claims traceable to context
- `instruction_adherence_score`: Follows task requirements
- `completeness_score`: Addresses all expected elements

**NLP Metrics (0-1 scale)**:
- `bertscore_precision`: Semantic precision
- `bertscore_recall`: Semantic recall
- `bertscore_f1`: Harmonic mean
- `rouge1_fmeasure`: Unigram overlap
- `rougeL_fmeasure`: Longest common subsequence

### D. Common Commands

**Setup**:
```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:7b
```

**Run**:
```bash
python evaluate.py --test    # Quick test
python evaluate.py           # Full evaluation
mlflow ui                    # View results
```

**Debug**:
```bash
python -m pytest test_system.py  # Run tests
python monolithic.py         # Test monolithic
python ensemble.py           # Test ensemble
```

**Clean**:
```bash
rm -rf data/cache/summaries/*
rm -rf data/cache/ensemble_summaries/*
rm -rf data/drafts/*
rm -rf mlruns/*
```

### E. Quick Decision Matrix

**Use Monolithic When**:
- ✅ Simple synthesis task
- ✅ Need fast results
- ✅ Limited API budget
- ✅ Prototype/testing

**Use Ensemble When**:
- ✅ Complex synthesis requiring refinement
- ✅ Quality is top priority
- ✅ Need iteration transparency
- ✅ Can afford higher latency/cost

**Use Test Mode When**:
- ✅ First run / setup verification
- ✅ Testing prompts
- ✅ Debugging
- ✅ Quick experiments

**Use Full Evaluation When**:
- ✅ Production comparison
- ✅ Complete metrics needed
- ✅ Publication/reporting
- ✅ Final validation

---

## Changelog

**2026-01-10**:
- Created comprehensive master documentation
- Consolidated QUICKSTART.md, USAGE.md, IMPLEMENTATION.md
- Added CrewAI Flow technical details
- Documented infinite loop fix
- Added troubleshooting section
- Added performance optimization notes

**2026-01-08**:
- Fixed CrewAI Flow infinite loop issue
- Documented fix in FLOW_FIX_SUMMARY.md

**Previous**:
- Initial implementation of monolithic and ensemble agents
- MLflow integration
- LLM-as-a-judge evaluation
- Map-reduce pattern with caching

---

**End of Master Documentation**
