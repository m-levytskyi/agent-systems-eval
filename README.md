# Agent Systems Evaluation: Monolithic vs Ensemble

Empirical comparison of two agent architectures for document synthesis:
- **Monolithic Agent**: Single LLM approach (fast, simple)
- **Ensemble Agent**: Multi-agent system with recursive orchestration (higher quality, iterative refinement)

Evaluation uses MLflow tracking, LLM-as-a-judge scoring, and NLP metrics (BERTScore, ROUGE).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Defaults are configured for local Ollama
```

### 3. Pull Model
```bash
ollama pull qwen2.5:7b
```

### 4. Run Test Evaluation
```bash
python evaluate.py --test
# Fast test with 1 paper, completes in ~5-10 minutes
# Use --agents-model=gemini to test with Google Gemini instead of Ollama
```

### 5. View Results
```bash
mlflow ui
# Open http://localhost:5000
```

## Usage

### Test Mode (Recommended First Run)
```bash
python evaluate.py --test
# Or with Gemini:
python evaluate.py --test --agents-model=gemini
```
- Processes 1 paper, 1 task
- Completes in ~5-10 minutes
- Verifies setup works

### Full Evaluation
```bash
python evaluate.py
# Or with Gemini:
python evaluate.py --agents-model=gemini
```
- Processes all 10 papers, 3 tasks
- Takes 1-2 hours
- Generates complete comparison

### Model Selection
```bash
# Use local Ollama (default, free)
python evaluate.py --agents-model=ollama

# Use Google Gemini (requires API key)
python evaluate.py --agents-model=gemini
```
- Judges always use Gemini for consistency
- MLflow experiments get `_gemini` suffix when using Gemini agents
- See `CLI_USAGE.md` for detailed usage guide

### Resume from Cache
If interrupted, simply rerun the command. Already-processed documents load from `data/cache/` instantly.

### Clear Cache
```bash
rm -rf data/cache/summaries/* data/cache/ensemble_summaries/*
```

## Project Structure

```
agent-systems-eval/
├── README.md                   # Public documentation
├── MASTER_README.md           # Comprehensive private docs
├── monolithic.py              # Single LLM agent
├── ensemble.py                # Multi-agent ensemble (CrewAI Flows)
├── evaluate.py                # MLflow evaluation framework
├── utils.py                   # Shared utilities
├── rate_limits.py            # API rate limiter
├── llm/                       # LLM client abstraction
│   ├── ollama.py             # Ollama implementation
│   ├── gemini.py             # Gemini implementation
│   └── factory.py            # Client factory
├── data/
│   ├── source_documents/     # PDF inputs
│   ├── tasks/                # Task definitions
│   └── cache/                # Cached summaries
└── mlruns/                   # MLflow tracking data
```

## Key Features

- **Two Agent Architectures**: Compare monolithic vs multi-agent approaches
- **Recursive Orchestration**: Ensemble uses CrewAI Flows for iterative refinement
- **MLflow Tracking**: Complete experiment management and comparison
- **LLM-as-a-Judge**: Automated quality evaluation (groundedness, adherence, completeness)
- **NLP Metrics**: BERTScore and ROUGE for quantitative analysis
- **Map-Reduce Processing**: Efficient handling of large documents with caching
- **PDF Support**: Processes academic papers directly

## Expected Results

**Monolithic Agent**:
- ✅ Fast (~5-15 seconds per task)
- ✅ Low token usage
- ✅ Good for straightforward synthesis

**Ensemble Agent**:
- ✅ Higher quality scores (~15-25% improvement)
- ✅ Adaptive iteration (orchestrator decides when ready)
- ✅ Full iteration history logged
- ⚠️ Slower (~2-3x latency)
- ⚠️ Higher token usage (minimal cost with local Ollama)

## Configuration

### CLI Arguments

- `--agents-model {ollama,gemini}`: Choose model provider for agents (default: `ollama`)
- `-t, --test`: Run in test mode (1 paper, 1 task)

Run `python evaluate.py --help` for all options.

### Environment Variables

Environment variables (`.env`):

**Required**:
- `OLLAMA_MODEL`: Model name for Ollama agents (default: `qwen2.5:7b`)
- `OLLAMA_NUM_CTX`: Context window (default: `32768`)
- `CREWAI_MODEL`: Model for Ensemble when using Ollama (default: `openai/qwen2.5:7b`)
- `JUDGE_MODEL`: MLflow judge - always Gemini (default: `gemini:/gemini-2.5-flash-lite`)

**Required for Gemini agents** (`--agents-model=gemini`):
- `GEMINI_API_KEY`: Your Gemini API key

**Note**: When using `--agents-model=gemini`, agents use `gemini-2.5-flash-lite` and `gemini/gemini-2.5-flash-lite` (hardcoded), overriding `OLLAMA_MODEL` and `CREWAI_MODEL`.

See `.env.example` for full configuration.

## Adding Custom Tasks

1. Add PDF/text files to `data/source_documents/`
2. Edit `data/tasks/synthesis_tasks.json`:
```json
{
  "task_id": "custom_1",
  "task_description": "Your task description...",
  "expected_elements": ["Element 1", "Element 2"]
}
```
3. Run `python evaluate.py`

## Documentation

- **README.md** (this file): Quick start and essential usage
- **CLI_USAGE.md**: Command-line interface guide with all options and examples
- **MASTER_README.md**: Comprehensive private documentation with:
  - Detailed architecture and implementation notes
  - Troubleshooting and debugging guides
  - Performance optimization tips
  - Technical deep dives

## Requirements

- Python 3.10+
- Ollama (for local inference)
- See `requirements.txt` for Python dependencies

Optional: Google Gemini API key (if using `LLM_PROVIDER=gemini`)

## License

MIT License

## Contributing

Contributions welcome! Please submit a Pull Request.