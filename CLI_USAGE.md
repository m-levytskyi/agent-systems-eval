# CLI Usage Guide for Evaluation Script

## Overview

The `evaluate.py` script now supports switching between Ollama and Gemini for agent models via a CLI argument, while judges always use Gemini for consistency.

## Command-Line Arguments

### `--agents-model {ollama,gemini}`

Select the model provider for agents (MonolithicAgent and EnsembleAgent):

- **`ollama`** (default): Uses local Ollama models
  - MonolithicAgent: Uses `OLLAMA_MODEL` from .env (default: `qwen2.5:7b`)
  - EnsembleAgent: Uses `CREWAI_MODEL` from .env (default: `openai/qwen2.5:7b`)
  - Judges: Always use Gemini (`JUDGE_MODEL` from .env)

- **`gemini`**: Uses Google Gemini API for all agents
  - MonolithicAgent: `gemini-2.5-flash-lite`
  - EnsembleAgent: `gemini/gemini-2.5-flash-lite`
  - Judges: Always use Gemini (`JUDGE_MODEL` from .env)
  - **Requires**: `GEMINI_API_KEY` environment variable must be set

### `-t, --test`

Run in test mode (single paper, single task) for quick testing.

## Usage Examples

### Standard Evaluation with Ollama (default)

```bash
python evaluate.py
# or explicitly:
python evaluate.py --agents-model=ollama
```

**MLflow Experiments Created:**
- `document_synthesis_monolithic`
- `document_synthesis_ensemble`

### Standard Evaluation with Gemini

```bash
python evaluate.py --agents-model=gemini
```

**MLflow Experiments Created:**
- `document_synthesis_monolithic_gemini`
- `document_synthesis_ensemble_gemini`

### Test Mode with Ollama

```bash
python evaluate.py --test
# or:
python evaluate.py -t --agents-model=ollama
```

### Test Mode with Gemini

```bash
python evaluate.py -t --agents-model=gemini
```

## MLflow Experiment Naming

The experiment naming scheme automatically appends `_gemini` suffix when running with Gemini agents:

| Agent Type   | Ollama Experiment Name               | Gemini Experiment Name                      |
|--------------|--------------------------------------|---------------------------------------------|
| Monolithic   | `document_synthesis_monolithic`      | `document_synthesis_monolithic_gemini`      |
| Ensemble     | `document_synthesis_ensemble`        | `document_synthesis_ensemble_gemini`        |

This allows you to:
- Compare Ollama vs Gemini performance side-by-side in MLflow UI
- Track experiments separately by model provider
- Avoid mixing results from different model providers

## Environment Variables

### Required for Ollama (when `--agents-model=ollama`)

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_NUM_CTX=32768
CREWAI_MODEL=openai/qwen2.5:7b
```

### Required for Gemini (when `--agents-model=gemini`)

```bash
GEMINI_API_KEY=your_api_key_here
```

### Always Required (Judges)

```bash
JUDGE_MODEL=gemini:/gemini-2.5-flash-lite
```

## Error Handling

If you run with `--agents-model=gemini` without setting `GEMINI_API_KEY`, the script will:
1. Display a clear error message
2. Exit with code 1
3. Prompt you to set the API key in your `.env` file

Example error:
```
ERROR: GEMINI_API_KEY environment variable is not set.
Please set GEMINI_API_KEY in your .env file to use --agents-model=gemini
```

## Help

View all available options:

```bash
python evaluate.py --help
```

Output:
```
usage: evaluate.py [-h] [--agents-model {ollama,gemini}] [-t]

Evaluate Monolithic vs Ensemble agents for document synthesis

options:
  -h, --help            show this help message and exit
  --agents-model {ollama,gemini}
                        Model to use for agents: 'ollama' (default) or 'gemini'. 
                        Judges always use Gemini.
  -t, --test            Run in test mode (single paper, single task)
```
