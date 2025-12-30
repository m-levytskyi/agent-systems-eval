# Usage Guide

## Test Mode (Single Paper)

For quick testing with just one paper and one task:

```bash
python evaluate.py --test
# or
python evaluate.py -t
```

This will:
- Use only `paper_1.pdf`
- Run only the first task
- Complete in ~5-10 minutes instead of hours
- Perfect for verifying the setup works

## Full Evaluation (All Papers)

For complete evaluation with all 10 papers and all tasks:

```bash
python evaluate.py
```

Expected time: 1-2 hours depending on hardware.

## Checkpoint/Resume Support

The system automatically saves summaries to `data/cache/` after processing each document.

**If interrupted:**
- Simply rerun `python evaluate.py`
- Already-processed documents will load from cache instantly
- Only unprocessed documents will be summarized
- Saves hours on reruns

**To clear cache and start fresh:**

```bash
rm -rf data/cache/summaries/*
rm -rf data/cache/ensemble_summaries/*
```

## Cache Locations

- Monolithic agent: `data/cache/summaries/doc_N_summary.json`
- Ensemble agent: `data/cache/ensemble_summaries/doc_N_summary.json`

Each cache file contains:
- The document summary
- Token usage metrics
- Processing metadata (chunks, original length, etc.)

## Viewing Results

After evaluation completes:

```bash
mlflow ui
```

Then open http://localhost:5000 to view:
- Metrics comparison
- Quality scores
- Generated outputs
- Document summaries (as artifacts)
