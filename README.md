# Agent Systems Evaluation: Monolithic vs Ensemble

An empirical comparison of a Monolithic Agent (single LLM) vs. a Multi-Agent Ensemble for document synthesis tasks. This project evaluates both approaches using MLflow for experiment tracking, LLM-as-a-judge, and NLP metrics.

## Overview

This project implements and compares two approaches to document synthesis:

1. **Monolithic Agent** (`monolithic.py`): A single LLM that directly synthesizes source documents according to task requirements.

2. **Ensemble Agent** (`ensemble.py`): A four-agent system using CrewAI Flows with recursive orchestration:
   - **Archivist**: Extracts and organizes key information from source documents (runs once)
   - **Drafter**: Creates synthesis based on archivist's organization (iterative)
   - **Critic**: Reviews and provides detailed feedback on the draft (iterative)
   - **Orchestrator**: Evaluates feedback and decides whether to iterate or finalize (recursive control)
   
   The workflow uses CrewAI Flows API for recursive refinement:
   - Archivist runs once to organize material
   - Drafter → Critic → Orchestrator loop continues until production-ready
   - Maximum 5 iterations or 30-minute timeout
   - Full iteration history tracked in MLflow artifacts

## Features

- 🤖 Two distinct agent architectures for document synthesis
- 🔄 Recursive orchestration with quality-controlled iteration (ensemble only)
- 📊 MLflow integration for experiment tracking and comparison
- 💰 Cost and latency metrics for each approach
- 🎯 MLflow GenAI LLM-judge evaluation for quality assessment
- 📈 NLP metrics: BERTScore and ROUGE for quantitative evaluation
- 📄 PDF document support for realistic document processing
- 📝 Sample PDF documents and synthesis tasks included

## Requirements

- Python 3.10+
- Ollama (for local inference)
- Dependencies listed in `requirements.txt`

Optional:
- CrewAI (for orchestrating the ensemble)
- Google Gemini API key (only if you want `LLM_PROVIDER=gemini`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/m-levytskyi/agent-systems-eval.git
cd agent-systems-eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env (defaults are set up for local Ollama)
```

4. Pull the local model once:

```bash
ollama pull qwen2.5:7b
```

### Configuring Ollama Context Window

By default, Ollama models use a 2048-4096 token context window. For processing academic papers, you should increase this to 32k tokens.

**The context window is set via environment variable and passed in every API call:**

```bash
# Set in your environment or .env file
export OLLAMA_NUM_CTX=32768

# Then start/restart Ollama
ollama serve
```

Or add to your `.env` file:
```
OLLAMA_NUM_CTX=32768
```

The implementation automatically includes `num_ctx: 32768` in every API request to Ollama, overriding the server's default. This ensures the full context window is available for processing large documents.

The map-reduce implementation handles documents that exceed the context window by:
- Sanitizing documents (removing references, bibliographies, appendices)
- Chunking large documents into logical sections
- Processing each document independently with isolated API calls

## Usage

### Running the Evaluation

**Quick Test (Single Paper - Recommended for first run):**

```bash
python evaluate.py --test
```

This will:
- Process only 1 paper (paper_1.pdf)
- Run only the first task
- Complete in ~5-10 minutes
- Verify everything works correctly

**Full Evaluation (All Papers):**

```bash
python evaluate.py
```

This will:
- Process all 10 papers
- Run all 3 tasks  
- Take 1-2 hours depending on hardware
- Use checkpoint/caching for resilience

Both modes automatically:
- Cache document summaries to `data/cache/` for instant resume on interruption
- Load source PDF documents from `data/source_documents/`
- Load synthesis tasks from `data/tasks/synthesis_tasks.json`
- Run both monolithic and ensemble agents on all tasks
- Track metrics in MLflow
- Evaluate outputs using LLM-as-a-judge and NLP metrics
- Save all results and artifacts

### Checkpoint/Resume Support

If evaluation is interrupted (crash, Ctrl+C, etc.):
- Simply rerun `python evaluate.py`
- Already-processed documents load from cache instantly
- Only unprocessed documents will be summarized
- Saves significant time on reruns

**Clear cache to start fresh:**
```bash
rm -rf data/cache/summaries/* data/cache/ensemble_summaries/*
```

### Viewing Results

After running the evaluation, view results in the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to:
- Compare runs across both agent types
- View metrics (cost, latency, quality scores, NLP metrics)
- Examine generated outputs and intermediate results
- Analyze performance across different tasks

### Running Individual Agents

You can also run each agent independently:

**Monolithic Agent:**
```bash
python monolithic.py
```

**Ensemble Agent:**
```bash
python ensemble.py
```

## Project Structure

```
agent-systems-eval/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── monolithic.py              # Single LLM agent implementation
├── ensemble.py                # Multi-agent ensemble implementation
├── evaluate.py                # Main evaluation script with MLflow
├── data/
│   ├── source_documents/      # Sample PDF documents
│   │   ├── doc1_ai_history.pdf
│   │   ├── doc2_ml_fundamentals.pdf
│   │   └── doc3_ai_ethics.pdf
│   └── tasks/                 # Synthesis task definitions
│       └── synthesis_tasks.json
└── mlruns/                    # MLflow tracking data (generated)
```

## Metrics Tracked

### Process Metrics
- **Latency**: Total time to complete synthesis
- **Token Usage**: Prompt, completion, and total tokens
- **API Calls**: Number of LLM API calls
- **Estimated Cost**: Logged as `0.0` for local Ollama; estimated for remote providers

### Quality Metrics (LLM-as-a-judge)
- **Completeness**: How fully the task requirements are addressed
- **Coherence**: Clarity, logic, and structure of the writing
- **Accuracy**: Correctness and integration of information
- **Quality**: Overall professional quality
- **Overall**: Aggregate quality score

### NLP Metrics
- **BERTScore**: Precision, Recall, F1 measuring semantic similarity
- **ROUGE**: ROUGE-1 and ROUGE-L measuring n-gram overlap

### Ensemble-Specific Metrics
- Token usage per agent (archivist, drafter, critic, orchestrator)
- Number of iterations until production-ready
- Iteration history with per-iteration drafts and critiques
- Intermediate outputs at each stage

## Configuration

Environment variables (set in `.env`):
- `LLM_PROVIDER`: `ollama` (default) or `gemini`
- `OLLAMA_BASE_URL`: Ollama HTTP endpoint (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: local model name (default: `qwen2.5:7b`)
- `JUDGE_MODEL`: MLflow GenAI judge model URI (default: `openai:/qwen2.5:7b`)
- `OPENAI_BASE_URL`: for using Ollama via OpenAI-compat (`http://localhost:11434/v1`)
- `OPENAI_API_KEY`: dummy value for OpenAI-compat (e.g. `ollama`)
- `CREWAI_MODEL`: model identifier for CrewAI ensemble (e.g. `openai/qwen2.5:7b`)
- `MAX_ITERATIONS`: maximum iterations for ensemble (default: 5)
- `TIMEOUT_SECONDS`: maximum time for synthesis (default: 1800 = 30 minutes)

Optional (Gemini):
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- `GEMINI_MODEL`

## Adding Custom Tasks

To add your own synthesis tasks:

1. Add source documents (PDF or text) to `data/source_documents/`
2. Edit `data/tasks/synthesis_tasks.json` to add new tasks:

```json
{
  "task_id": "task4",
  "task_description": "Your synthesis task description",
  "expected_elements": [
    "Element 1",
    "Element 2"
  ]
}
```

## Expected Results

The ensemble approach typically shows:
- ✅ Higher quality scores (better organization, iterative refinement)
- ✅ Higher NLP metric scores (more comprehensive coverage)
- ✅ Adaptive quality control (orchestrator decides when ready)
- ✅ Transparent iteration history (all drafts and feedback logged)
- ⚠️ Higher latency (multiple iterations with 4 agents)
- ⚠️ Higher cost (more total tokens, though minimal with local Ollama)
- ✅ Better handling of complex synthesis tasks requiring refinement

The monolithic approach typically shows:
- ✅ Lower latency (single LLM call)
- ✅ Lower cost (fewer tokens)
- ⚠️ May miss nuances that benefit from specialized processing
- ✅ Efficient for straightforward tasks

## API Costs & Limits

Local Ollama runs have no per-token API costs.

Optional: using **Google Gemini**:
- Rate limits and pricing depend on your Gemini tier.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.