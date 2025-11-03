# Quick Start Guide

This guide will help you get started with the agent systems evaluation in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- pip (Python package manager)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key

Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_api_key_here` with your actual API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4
```

**Note:** Using GPT-4 is recommended for best quality, but you can also use `gpt-3.5-turbo` for lower costs.

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

This demonstrates both agents on simple tasks with minimal API costs (~$0.10).

### Option B: Full Evaluation (Complete Comparison)

Run the complete evaluation on all tasks:

```bash
python evaluate.py
```

This will:
- Process 3 synthesis tasks with both agents (6 total runs)
- Track all metrics in MLflow
- Generate quality scores using LLM-as-a-judge
- Estimated cost: ~$2-5 depending on model

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
   - Quality scores (LLM judge ratings)
4. **Artifacts**: View generated syntheses and intermediate outputs

## Understanding the Results

### Monolithic Agent
- ✅ **Faster**: Single LLM call
- ✅ **Cheaper**: Fewer tokens
- ⚠️ **Simpler**: Direct synthesis without specialized processing

### Ensemble Agent
- ✅ **Higher Quality**: Specialized agents for extraction, drafting, and refinement
- ✅ **More Structured**: Clear separation of concerns
- ⚠️ **Slower**: 3 sequential LLM calls
- ⚠️ **More Expensive**: 3x the API calls

## Customizing for Your Use Case

### Adding Your Own Documents

1. Add text files to `data/source_documents/`
2. Update `data/tasks/synthesis_tasks.json` with your tasks
3. Run `evaluate.py`

### Changing the Model

Edit `.env` to use a different model:

```
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper option
# or
OPENAI_MODEL=gpt-4-turbo    # Faster GPT-4 variant
```

### Adjusting Agent Behavior

Edit the system prompts in:
- `monolithic.py`: Line ~40
- `ensemble.py`: Lines ~70, ~110, ~150 (for each agent role)

## Troubleshooting

### "Invalid API Key" Error
- Check your `.env` file
- Ensure your API key starts with `sk-`
- Verify your OpenAI account has credits

### "Rate Limit" Error
- Wait a few seconds and try again
- Consider using `gpt-3.5-turbo` for higher rate limits

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

### MLflow UI Not Loading
- Check if port 5000 is available
- Try: `mlflow ui --port 5001`

## Next Steps

- 📊 Compare metrics across multiple runs
- 🔧 Tune prompts for your specific domain
- 📈 Add more synthesis tasks
- 🎯 Experiment with different models
- 📝 Analyze intermediate outputs from the ensemble

## Cost Estimation

Typical costs for the full evaluation (3 tasks, both agents):

- **GPT-4**: ~$2-5 per full evaluation
- **GPT-3.5-turbo**: ~$0.10-0.20 per full evaluation

Each task uses approximately:
- Monolithic: 1,500-2,500 tokens
- Ensemble: 4,500-7,500 tokens (3x agents)

## Support

For issues or questions:
- Check the main README.md
- Review MLflow logs for detailed error messages
- Ensure your API key has sufficient credits

Happy evaluating! 🚀
