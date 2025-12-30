# Quick Start Guide

This guide will help you get started with the agent systems evaluation in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- Ollama ([Install](https://ollama.com))
- pip (Python package manager)

Optional:
- Google Gemini API key (only if you want `LLM_PROVIDER=gemini`)

Note: The ensemble agent uses CrewAI Flows for recursive orchestration.

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` for local Ollama (default):

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# MLflow judges (use Ollama via OpenAI-compat endpoint)
JUDGE_MODEL=openai:/qwen2.5:7b
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama

# CrewAI Ensemble with recursive orchestration
CREWAI_MODEL=openai/qwen2.5:7b
MAX_ITERATIONS=5
TIMEOUT_SECONDS=1800
```

Pull the model once:

```bash
ollama pull qwen2.5:7b
```

### 3. Verify Installation

Run the test suite to ensure everything is set up correctly:

```bash
python test_system.py
```

You should see: `🎉 All tests passed!`

## Running Your First Evaluation

### Option A: Quick Example (Minimal API Usage)

Run the example script with small documents:

```bash
python example_usage.py
```

This demonstrates both agents on simple tasks.

### Option B: Full Evaluation (Complete Comparison)

Run the complete evaluation on all tasks:

```bash
python evaluate.py
```

This will:
- Process 3 synthesis tasks with both agents (6 total runs)
- Track all metrics in MLflow (cost, latency, quality, NLP metrics)
- Generate quality scores using MLflow GenAI LLM-judge scorers
- Compute BERTScore and ROUGE metrics

Note: When running locally via Ollama, `estimated_cost_usd` logs as `0.0`.

### Option C: Individual Agents

Test each agent separately:

**Monolithic Agent:**
```bash
python monolithic.py
```

**Ensemble Agent:**
```bash
python ensemble.py
```

## Viewing Results

After running `evaluate.py`, view the results in MLflow:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### What to Look For in MLflow

1. **Experiments Tab**: See "document_synthesis_monolithic" and "document_synthesis_ensemble"
2. **Runs**: Compare individual runs across tasks
3. **Metrics**: 
   - Cost and latency (process metrics)
   - Quality scores (LLM judge ratings: completeness, coherence, accuracy)
   - NLP metrics (BERTScore F1, ROUGE-1/L scores)
4. **Artifacts**: View generated syntheses and intermediate outputs

## Understanding the Results

### Monolithic Agent
- ✅ **Faster**: Single LLM call
- ✅ **Cheaper**: Fewer tokens
- ⚠️ **Simpler**: Direct synthesis without specialized processing

### Ensemble Agent
- ✅ **Higher Quality**: Four specialized agents with iterative refinement
- ✅ **Recursive Orchestration**: Orchestrator decides when output is production-ready
- ✅ **More Structured**: Clear separation with quality feedback loops
- ✅ **Better NLP Scores**: More comprehensive document coverage through iteration
- ✅ **Transparent**: Full iteration history logged to MLflow
- ⚠️ **Slower**: Multiple iterations with 4 agents (Archivist, Drafter, Critic, Orchestrator)
- ⚠️ **More Token Usage**: Iterative process uses more API calls (but still minimal cost with local Ollama)

## Customizing for Your Use Case

### Adding Your Own Documents

1. Add PDF or text files to `data/source_documents/`
2. Update `data/tasks/synthesis_tasks.json` with your tasks
3. Run `evaluate.py`

### Changing the Model

Edit `.env` to use a different Ollama model:

```
OLLAMA_MODEL=qwen2.5:7b
CREWAI_MODEL=openai/qwen2.5:7b
```

### Adjusting Agent Behavior

Edit the system prompts in:
- `monolithic.py`: Line ~60
- `ensemble.py`: Agent definitions for Archivist, Drafter, Critic, and Orchestrator

## Troubleshooting

### "Invalid API Key" Error
- If using Ollama judges via OpenAI-compat, ensure `OPENAI_BASE_URL` and `OPENAI_API_KEY` are set.
- If using Gemini (`LLM_PROVIDER=gemini`), ensure `GEMINI_API_KEY` is set.

### "Rate Limit" Error (Free Tier)
- For remote providers, set `MAX_RPM` and `MAX_RPD` in `.env` to enable throttling.

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.10 or higher

### MLflow UI Not Loading
- Check if port 5000 is available
- Try: `mlflow ui --port 5001`

### PDF Loading Issues
- Ensure PyPDF2 is installed: `pip install PyPDF2`
- Check that PDF files are in `data/source_documents/`

## Next Steps

- 📊 Compare metrics across multiple runs
- 🔧 Tune prompts for your specific domain
- 📈 Add more synthesis tasks
- 🎯 Experiment with different Gemini models
- 📝 Analyze intermediate outputs from the ensemble
- 📄 Try with your own PDF documents

## Cost Estimation

When running locally via Ollama, `estimated_cost_usd` is logged as `0.0`.

## Support

For issues or questions:
- Check the main README.md
- Review MLflow logs for detailed error messages
- Ensure you have a valid Google Gemini API key
- Visit: https://ai.google.dev/gemini-api/docs for API documentation

Happy evaluating! 🚀
