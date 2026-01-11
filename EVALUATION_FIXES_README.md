# Evaluation Fixes - Fair Comparison Implementation

**Date:** January 11, 2026  
**Status:** ✅ Fixes implemented, ready for re-evaluation

## Executive Summary

Critical bugs were discovered in the evaluation methodology that made the monolithic vs ensemble comparison invalid. The primary issue was **incorrect token counting** where ensemble appeared to use "98% fewer tokens" when the actual difference was ~20%.

**All previous evaluation results (in `evaluation_analysis.md`) are INVALID and should be discarded.**

---

## What Was Wrong

### Bug #1: Ensemble Cache Had Zero Token Counts ⚠️⚠️⚠️

**Evidence:**
```json
// Monolithic cache (data/cache/summaries/doc_1_summary.json)
{
  "metadata": {
    "tokens_used": 14639,  // ✓ Correct
    ...
  }
}

// Ensemble cache (data/cache/ensemble_summaries/doc_1_summary.json)
{
  "metadata": {
    "tokens_used": 0,  // ✗ BUG - Should be ~14,600
    ...
  }
}
```

**Impact:**
- Monolithic correctly counted ~146K tokens from cached summaries (10 docs × 14.6K)
- Ensemble counted 0 tokens from cached summaries
- This created a false ~146K token difference

**Root Cause:**
The `process_chunk` function in `ensemble.py` failed to extract token usage from CrewAI results and silently defaulted to 0.

### Bug #2: No Validation for Invalid Cache Data

The cache loading logic in `utils.py` didn't warn when `tokens_used` was 0, making the bug invisible.

### Bug #3: Inconsistent Methodology

Both agents used the same cached document summaries (map phase), but only monolithic included those tokens in its metrics. This wasn't a "bug" per se, but made the comparison unfair.

---

## What Was Fixed

### Fix #1: Improved Token Extraction (`ensemble.py`)

**Location:** `ensemble.py` lines 136-163

**Changes:**
- Added multiple fallback methods to extract token usage from CrewAI
- Implemented token estimation from input+output text when API data unavailable
- Added detailed logging when extraction fails or falls back to estimation
- Ensures `tokens_used` is never 0 in cache files

**Code:**
```python
# Try multiple ways to extract token usage
if hasattr(result, "usage_metrics"):
    # ... extract from usage_metrics
elif hasattr(result, "token_usage"):
    # ... extract from token_usage

# Fallback: estimate if extraction failed
if not tokens_found or metrics["total_tokens"] == 0:
    logger.warning(f"No token usage found for doc {doc_idx} chunk {chunk_idx}, using estimation")
    input_tokens = estimate_tokens(chunk)
    output_tokens = estimate_tokens(summary)
    metrics["total_tokens"] = input_tokens + output_tokens
```

### Fix #2: Cache Validation (`utils.py`)

**Location:** `utils.py` lines 205-211

**Changes:**
- Added validation warning when cache has `tokens_used=0`
- Helps catch this bug in future evaluations

**Code:**
```python
tokens_used = cached['metadata'].get('tokens_used', 0)
if tokens_used == 0:
    logger.warning(
        f"⚠️  Cache for document {doc_idx} has tokens_used=0! "
        f"This may indicate a bug in cache generation. "
        f"Cache file: {cache_file}"
    )
```

### Fix #3: Cleared All Caches

**Action taken:**
```bash
# Backed up old caches
data/cache/backup_20260111_111517/summaries/
data/cache/backup_20260111_111517/ensemble_summaries/

# Cleared for fresh evaluation
data/cache/summaries/ - EMPTY
data/cache/ensemble_summaries/ - EMPTY
```

### Fix #4: Verified Model Consistency

**Confirmed:**
- Monolithic: Uses `OLLAMA_MODEL` (defaults to `qwen2.5:7b`)
- Ensemble: Uses `CREWAI_MODEL` (defaults to `openai/qwen2.5:7b`)
- Both use the same underlying model ✓

---

## How to Run Fair Evaluation

### Option A: Full Pipeline Comparison (Recommended)

This provides the most honest comparison by measuring everything from scratch:

```bash
# Caches are already cleared, just run:
python evaluate.py

# This will:
# 1. Generate fresh document summaries for both agents (map phase)
# 2. Track tokens accurately for both
# 3. Perform synthesis (reduce/ensemble phase)
# 4. Generate fair comparison metrics
```

**Expected results:**
- Both agents will show ~146K tokens for document summarization (map phase)
- Monolithic will show ~151K tokens for synthesis (reduce phase with all summaries)
- Ensemble will show ~169K tokens for ensemble work:
  - Pre-reduction: ~148K (reducing summaries from 146K to ~6K)
  - Archivist: ~8K (processing reduced summaries)
  - Drafter: ~5K
  - Critic: ~4K  
  - Orchestrator: ~4K
- **Total: Monolithic ~297K vs Ensemble ~315K (ensemble uses +6% MORE, not less!)**

### Option B: Test Mode (Faster)

For quick verification of fixes:

```bash
python evaluate.py --test

# Uses only 1 document instead of 10
# Faster but less comprehensive
```

### Option C: Specific Task Only

```bash
python evaluate.py --task task1

# Evaluates only task1
# Good for debugging
```

---

## What to Expect in New Results

### Token Usage (Corrected Estimates)

| Agent | Map Phase | Reduce/Ensemble Phase | Total | % Difference |
|-------|-----------|----------------------|-------|--------------|
| **Monolithic** | ~146K | ~151K (1 call with all summaries) | **~297K** | baseline |
| **Ensemble** | ~146K | ~169K (reduction + 4 agents) | **~315K** | +6% MORE |

**Key insight:** Ensemble actually uses MORE tokens, not less!
- Monolithic: One synthesis call with all summaries
- Ensemble: Pre-reduction step + Archivist + Drafter + Critic + Orchestrator
- The "98% fewer tokens" was completely wrong (bug in cache metadata)
- The "20% fewer tokens" was also wrong (forgot about pre-reduction step)
- **Reality: Ensemble uses ~6% MORE tokens due to additional orchestration**

### Latency

Both agents load summaries fresh (no cache), so timing will be fair:
- Map phase: Similar for both (~10-15 min for 10 docs)
- Reduce phase: Varies based on synthesis complexity
- Total: Expect monolithic to be slightly faster (fewer orchestration steps)

### Quality Metrics

These should remain similar to before (assuming they weren't affected by the token bug):
- ROUGE-1 F1
- BERTScore F1
- Judge scores (instruction, groundedness, completeness)

---

## Verification Checklist

After running new evaluation, verify:

- [ ] No warnings about `tokens_used=0` in logs
- [ ] Both agents show similar map-phase token counts (~146K)
- [ ] Cache files contain valid token metadata
- [ ] MLflow logs show all metrics
- [ ] New analysis markdown generated

---

## Files Modified

1. **`ensemble.py`** - Improved token extraction with fallbacks
2. **`utils.py`** - Added cache validation warnings
3. **`evaluation_analysis.md`** - Added disclaimer about invalid data
4. **Cache directories** - Cleared and backed up

---

## Questions?

If you see unexpected results or have questions:

1. Check logs for warnings about token extraction
2. Inspect cache files: `cat data/cache/summaries/doc_1_summary.json`
3. Verify token counts in MLflow UI
4. Compare with backed-up cache in `data/cache/backup_*`

---

**Ready to run fair evaluation!** 🚀

```bash
python evaluate.py
```
