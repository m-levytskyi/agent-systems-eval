# Why Ensemble Uses MORE Tokens Than Monolithic

## TL;DR

**Ensemble should use ~6% MORE tokens than Monolithic, not less.**

The claim that "ensemble uses 98% fewer tokens" was completely false (bug in cache metadata).  
Even the corrected "20% fewer tokens" is wrong because it ignored the pre-reduction step.

---

## Token Flow Comparison

### MONOLITHIC (Simple Map-Reduce)

```
MAP PHASE (cached):
├─ Summarize doc 1: ~14.6K tokens
├─ Summarize doc 2: ~14.6K tokens  
├─ ... (8 more docs)
└─ Total: ~146K tokens

REDUCE PHASE (one API call):
├─ Input: ALL 10 summaries concatenated (~146K tokens)
├─ System prompt: ~100 tokens
├─ User prompt + task: ~50 tokens
├─ Output: synthesis (~5K tokens)
└─ Total: ~151K tokens

GRAND TOTAL: ~297K tokens
```

### ENSEMBLE (Multi-Agent with Pre-Reduction)

```
MAP PHASE (cached, identical to monolithic):
├─ Summarize doc 1: ~14.6K tokens
├─ Summarize doc 2: ~14.6K tokens
├─ ... (8 more docs)
└─ Total: ~146K tokens

ENSEMBLE PHASE:

1. PRE-REDUCTION (summaries > 6K token limit):
   ├─ Input: 10 summaries (~146K tokens)
   ├─ Chunk and reduce to ~6K tokens
   └─ Cost: ~148K tokens

2. ARCHIVIST:
   ├─ Input: Reduced summaries (~6K) + task (~100)
   ├─ Output: Organized notes (~2K)
   └─ Cost: ~8K tokens

3. DRAFTER:
   ├─ Input: Archivist notes (~2K) + task (~100)
   ├─ Output: Draft (~3K)
   └─ Cost: ~5K tokens

4. CRITIC:
   ├─ Input: Draft (~3K)
   ├─ Output: Feedback (~1K)
   └─ Cost: ~4K tokens

5. ORCHESTRATOR:
   ├─ Input: Draft + critique (~4K)
   ├─ Output: Decision (~0.5K)
   └─ Cost: ~4K tokens

Ensemble phase total: ~169K tokens

GRAND TOTAL: ~315K tokens
```

---

## Why Ensemble Uses More

| Component | Monolithic | Ensemble | Difference |
|-----------|------------|----------|------------|
| Map Phase | ~146K | ~146K | Same (cached) |
| Pre-reduction | N/A | ~148K | +148K |
| Main synthesis | ~151K | ~21K | -130K |
| **TOTAL** | **~297K** | **~315K** | **+6% MORE** |

**Key insight:** Ensemble adds a pre-reduction step that costs ~148K tokens to reduce the summaries from 146K down to ~6K. This allows the subsequent agents (archivist, drafter, etc.) to work with smaller context, but the total cost is HIGHER.

---

## Why The Bug Was Misleading

### What we saw (BUG):
```
Monolithic: 382K tokens
Ensemble:     6K tokens  
Difference: -98%  ← COMPLETELY FALSE
```

**Problem:** Ensemble cache had `tokens_used: 0` for map phase

### What we thought after fixing (STILL WRONG):
```
Monolithic: 191K tokens
Ensemble:   152K tokens
Difference: -20%  ← STILL WRONG
```

**Problem:** Forgot that ensemble does pre-reduction (~148K tokens)

### What's actually correct:
```
Monolithic: 297K tokens
Ensemble:   315K tokens
Difference: +6% MORE  ← CORRECT
```

---

## Why This Makes Sense

Ensemble uses MORE tokens because it has:
1. **Same map phase** as monolithic (~146K)
2. **Pre-reduction step** to fit summaries in context (~148K)
3. **Multiple agents** with smaller contexts (~21K total)

The trade-off is:
- **Tokens:** Ensemble costs more (+6%)
- **Quality:** Ensemble may produce better output (iterative refinement)
- **Architecture:** Ensemble is more modular and interpretable

Monolithic is more **token-efficient** because it does everything in one shot.  
Ensemble is potentially more **quality-focused** with its specialized agents.

---

## Implications for Evaluation

When running fresh evaluation (no cache):

**Expected token counts:**
- Monolithic: ~297K per task
- Ensemble: ~315K per task  
- Difference: Ensemble +6% more expensive

**The actual benefit of ensemble** (if any) will be in:
- **Quality metrics** (ROUGE, BERTScore, judge scores)
- **Iterative refinement** capability
- **Modularity** and interpretability

NOT in token efficiency. Ensemble is designed for quality, not cost reduction.

---

## Bottom Line

✅ **Monolithic should use FEWER tokens** (simpler, one-shot approach)  
✅ **Ensemble should use MORE tokens** (pre-reduction + multi-agent orchestration)  
❌ The "98% fewer tokens" claim was a complete bug  
❌ The "20% fewer tokens" claim forgot the pre-reduction step  
✅ The correct expectation is "Ensemble uses ~6% MORE tokens"

The evaluation should compare them on **quality**, not cost!
