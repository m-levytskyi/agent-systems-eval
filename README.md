# Agent Systems Evaluation: Monolithic vs Ensemble

An empirical comparison of a Monolithic Agent (single LLM) vs. a Multi-Agent Ensemble for document synthesis tasks. This project evaluates both approaches using MLflow for experiment tracking, LLM-as-a-judge, and NLP metrics.

## Overview

This project implements and compares two approaches to document synthesis:

1. **Monolithic Agent** (`monolithic.py`): A single LLM that directly synthesizes source documents according to task requirements.

2. **Ensemble Agent** (`ensemble.py`): A three-agent system with specialized roles:
   - **Archivist**: Extracts and organizes key information from source documents
   - **Drafter**: Creates initial synthesis based on the archivist's organization
   - **Critic**: Reviews and refines the draft for quality and completeness

## Features

- 🤖 Two distinct agent architectures for document synthesis
- 📊 MLflow integration for experiment tracking and comparison
- 💰 Cost and latency metrics for each approach
- 🎯 LLM-as-a-judge evaluation for quality assessment
- 📈 NLP metrics: BERTScore and ROUGE for quantitative evaluation
- 📄 PDF document support for realistic document processing
- 📝 Sample PDF documents and synthesis tasks included

## Requirements

- Python 3.8+
- Google Gemini API key (free tier available)
- Dependencies listed in `requirements.txt`

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

3. Set up your Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (GOOGLE_API_KEY also supported)
# Get your free API key at: https://ai.google.dev/gemini-api/docs/api-key
```

## Usage

### Running the Evaluation

To run the complete evaluation comparing both agents:

```bash
python evaluate.py
```

This will:
- Load source PDF documents from `data/source_documents/`
- Load synthesis tasks from `data/tasks/synthesis_tasks.json`
- Run both monolithic and ensemble agents on all tasks
- Track metrics in MLflow
- Evaluate outputs using LLM-as-a-judge and NLP metrics
- Save all results and artifacts

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
- **Estimated Cost**: Calculated based on token usage and model pricing (Gemini free tier)

### Quality Metrics (LLM-as-a-judge)
- **Completeness**: How fully the task requirements are addressed
- **Coherence**: Clarity, logic, and structure of the writing
- **Accuracy**: Correctness and integration of information
- **Quality**: Overall professional quality
- **Overall**: Aggregate quality score

### NLP Metrics
- **BERTScore**: Precision, Recall, F1 measuring semantic similarity
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L measuring n-gram overlap

### Ensemble-Specific Metrics
- Token usage per agent (archivist, drafter, critic)
- Intermediate outputs at each stage

## Configuration

Environment variables (set in `.env`):
- `GEMINI_API_KEY`: Your Google Gemini API key (preferred) - Get it at https://ai.google.dev/gemini-api/docs/api-key
- `GOOGLE_API_KEY`: Backward-compatible fallback for the API key
- `GEMINI_MODEL`: Model to use (default: gemini-2.5-pro)
- `GEMINI_JUDGE_MODEL`: Model for LLM-as-a-judge (default: same as GEMINI_MODEL)

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
- ✅ Higher quality scores (better organization and refinement)
- ✅ Higher NLP metric scores (more comprehensive coverage)
- ⚠️ Higher latency (3 sequential LLM calls)
- ⚠️ Higher cost (more total tokens, though minimal with Gemini free tier)
- ✅ Better handling of complex synthesis tasks

The monolithic approach typically shows:
- ✅ Lower latency (single LLM call)
- ✅ Lower cost (fewer tokens)
- ⚠️ May miss nuances that benefit from specialized processing
- ✅ Efficient for straightforward tasks

## API Costs & Limits

Using **Google Gemini 2.0 Flash** (free tier):
- **Rate Limits**: 15 requests per minute, 1M tokens per minute, 1500 requests per day
- **Cost**: Free tier available, or ~$0.000075/1K input tokens, ~$0.0003/1K output tokens
- **Typical Cost per Task**: ~$0.0001-0.0003 per task (essentially free for development)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.