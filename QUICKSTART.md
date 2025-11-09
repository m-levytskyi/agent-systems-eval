# Quick Start Guide

This guide will help you get started with the agent systems evaluation in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one FREE here](https://ai.google.dev/gemini-api/docs/api-key))
- pip (Python package manager)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key

Create a `.env` file with your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_api_key_here` with your actual API key:

```
GOOGLE_API_KEY=your-actual-api-key-here
GEMINI_MODEL=gemini-2.0-flash-exp
```

**Note:** The Gemini free tier provides 15 RPM, 1M TPM, 1500 RPD - perfect for development and testing!

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

This demonstrates both agents on simple tasks with minimal API usage (essentially free with Gemini).

### Option B: Full Evaluation (Complete Comparison)

Run the complete evaluation on all tasks:

```bash
python evaluate.py
```

This will:
- Process 3 synthesis tasks with both agents (6 total runs)
- Track all metrics in MLflow (cost, latency, quality, NLP metrics)
- Generate quality scores using LLM-as-a-judge
- Compute BERTScore and ROUGE metrics
- Estimated cost: $0.001-0.002 (essentially free with Gemini free tier!)

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
   - NLP metrics (BERTScore F1, ROUGE-1/2/L scores)
4. **Artifacts**: View generated syntheses and intermediate outputs

## Understanding the Results

### Monolithic Agent
- ✅ **Faster**: Single LLM call
- ✅ **Cheaper**: Fewer tokens
- ⚠️ **Simpler**: Direct synthesis without specialized processing

### Ensemble Agent
- ✅ **Higher Quality**: Specialized agents for extraction, drafting, and refinement
- ✅ **More Structured**: Clear separation of concerns
- ✅ **Better NLP Scores**: More comprehensive document coverage
- ⚠️ **Slower**: 3 sequential LLM calls
- ⚠️ **More Token Usage**: 3x the API calls (but still minimal cost with Gemini)

## Customizing for Your Use Case

### Adding Your Own Documents

1. Add PDF or text files to `data/source_documents/`
2. Update `data/tasks/synthesis_tasks.json` with your tasks
3. Run `evaluate.py`

### Changing the Model

Edit `.env` to use a different Gemini model:

```
GEMINI_MODEL=gemini-1.5-flash  # Alternative model
# or
GEMINI_MODEL=gemini-1.5-pro    # Higher capability model
```

### Adjusting Agent Behavior

Edit the system prompts in:
- `monolithic.py`: Line ~60
- `ensemble.py`: Lines ~90, ~120, ~150 (for each agent role)

## Troubleshooting

### "Invalid API Key" Error
- Check your `.env` file
- Get your free API key at https://ai.google.dev/gemini-api/docs/api-key
- Verify your key is correctly set in GOOGLE_API_KEY

### "Rate Limit" Error (Free Tier)
- Free tier limits: 15 RPM, 1M TPM, 1500 RPD
- Wait a few seconds between runs
- Space out your evaluation runs if hitting limits

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

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

Using **Google Gemini 2.0 Flash** (free tier):

- **Free Tier**: 15 RPM, 1M TPM, 1500 RPD - perfect for this evaluation!
- **Paid Tier**: ~$0.000075/1K input tokens, ~$0.0003/1K output tokens

Typical usage for the full evaluation (3 tasks, both agents):
- **Total Cost**: ~$0.001-0.002 (essentially free!)

Each task uses approximately:
- Monolithic: 1,500-2,500 tokens (~$0.0002)
- Ensemble: 4,500-7,500 tokens (~$0.0006)

**Total for 6 runs: Less than $0.002!**

Perfect for experimentation and development without worrying about costs.

## Support

For issues or questions:
- Check the main README.md
- Review MLflow logs for detailed error messages
- Ensure you have a valid Google Gemini API key
- Visit: https://ai.google.dev/gemini-api/docs for API documentation

Happy evaluating! 🚀
