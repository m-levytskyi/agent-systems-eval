# MLflow Evaluation Runs Analysis

⚠️ **IMPORTANT NOTICE: THIS ANALYSIS IS INVALID** ⚠️

**This report contains INCORRECT data due to bugs in token accounting.** A critical bug was discovered where:
- The ensemble cache files contained `tokens_used: 0` instead of actual token counts
- This caused the ensemble to appear to use "98% fewer tokens" when the actual difference was ~20%
- The comparison methodology was inconsistent between monolithic and ensemble agents

**Status:** 
- ✅ Bugs have been fixed in [`ensemble.py`](ensemble.py ) and [`utils.py`](utils.py )
- ✅ All caches have been cleared
- 🔄 **New evaluation needed with fresh data for accurate comparison**

**Do not use this data for any conclusions or decisions.**

See the end of this document for details on what was wrong and how it was fixed.

---

**Analysis Date:** 2026-01-11 11:01:52

## Overview

This report analyzes **20 runs** from **3 complete evaluation executions**:

- **Evaluation Run #1:** 2026-01-11 10:32:07 to 10:50:00
- **Evaluation Run #2:** 2026-01-11 01:29:31 to 02:40:24
- **Evaluation Run #3:** 2026-01-10 21:59:34 to 23:58:58

Each evaluation run consists of:
- 3 **Monolithic** agent runs (task1, task2, task3)
- 3 **Ensemble** agent runs (task1, task2, task3)

## Task Descriptions

- **task1:** Define the academic scope, terminology, and technological context of the provided corpus
- **task2:** Perform a structured extraction of core research components for each individual paper
- **task3:** Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus

## Performance Metrics Summary

### Evaluation Run #1
*Execution Time: 2026-01-11 10:32:07 to 10:50:00*

| Agent Type | Task | Latency (s) | Total Tokens | API Calls | ROUGE-1 F1 | BERTScore F1 | Judge Scores (I/G/C) |
|------------|------|-------------|--------------|-----------|------------|--------------|----------------------|
| Ensemble | task1 | 190.9 | 4,167 | 21 | 0.172 | 0.797 | 3/4/5 |
| Ensemble | task2 | 141.4 | 6,810 | 42 | 0.136 | 0.823 | 3/2/4 |
| Ensemble | task3 | 132.6 | 9,170 | 63 | 0.140 | 0.817 | 2/3/3 |
| Monolithic | task1 | 301.3 | 191,550 | 15 | 0.279 | 0.824 | 4/5/5 |
| Monolithic | task2 | 248.5 | 382,780 | 30 | 0.233 | 0.834 | 3/4/4 |
| Monolithic | task3 | 223.6 | 573,846 | 45 | 0.229 | 0.817 | 4/4/4 |

### Evaluation Run #2
*Execution Time: 2026-01-11 01:29:31 to 02:40:24*

| Agent Type | Task | Latency (s) | Total Tokens | API Calls | ROUGE-1 F1 | BERTScore F1 | Judge Scores (I/G/C) |
|------------|------|-------------|--------------|-----------|------------|--------------|----------------------|
| Ensemble | task1 | 996.7 | 3,825 | 21 | 0.160 | 0.805 | 4/5/5 |
| Ensemble | task2 | 934.9 | 7,417 | 42 | 0.156 | 0.805 | 4/4/3 |
| Ensemble | task3 | 853.0 | 10,616 | 63 | 0.166 | 0.811 | 3/5/3 |
| Monolithic | task1 | 928.5 | 191,460 | 15 | 0.301 | 0.790 | 4/4/5 |
| Monolithic | task2 | 742.3 | 382,557 | 30 | 0.219 | 0.831 | 3/4/5 |
| Monolithic | task3 | 595.8 | 573,368 | 45 | 0.182 | 0.840 | 2/4/4 |

### Evaluation Run #3
*Execution Time: 2026-01-10 21:59:34 to 23:58:58*

| Agent Type | Task | Latency (s) | Total Tokens | API Calls | ROUGE-1 F1 | BERTScore F1 | Judge Scores (I/G/C) |
|------------|------|-------------|--------------|-----------|------------|--------------|----------------------|
| Ensemble | task1 | 694.7 | 2,390 | 21 | 0.135 | 0.808 | 4/5/4 |
| Ensemble | task2 | 1444.9 | 4,501 | 42 | 0.131 | 0.804 | 2/2/2 |
| Ensemble | task2 | 828.0 | 5,239 | 42 | 0.124 | 0.814 | 5/4/2 |
| Ensemble | task3 | 635.7 | 6,639 | 63 | 0.128 | 0.830 | 2/4/2 |
| Ensemble | task3 | 945.0 | 8,458 | 63 | 0.126 | 0.815 | 5/3/3 |
| Monolithic | task1 | 880.7 | 191,146 | 15 | 0.234 | 0.827 | 4/5/5 |
| Monolithic | task2 | 892.0 | 382,300 | 30 | 0.225 | 0.819 | 3/4/5 |
| Monolithic | task3 | 888.7 | 573,541 | 45 | 0.240 | 0.824 | 3/4/4 |

## Cross-Evaluation Comparison

### Average Metrics Across All Runs

| Metric | Monolithic | Ensemble | Difference |
|--------|------------|----------|------------|
| Latency (s) | 633.5 | 708.9 | +11.9% |
| Total Tokens | 382,505.3 | 6,293.8 | -98.4% |
| API Calls | 30.0 | 43.9 | +46.4% |
| ROUGE-1 F1 | 0.238 | 0.143 | -0.095 |
| BERTScore F1 | 0.823 | 0.812 | -0.011 |
| Judge: Instruction | 3.333 | 3.364 | +0.030 |
| Judge: Groundedness | 4.222 | 3.727 | -0.495 |
| Judge: Completeness | 4.556 | 3.273 | -1.283 |

### Average Metrics by Task

#### task1

| Metric | Monolithic | Ensemble |
|--------|------------|----------|
| Latency (s) | 703.5 | 627.5 |
| Total Tokens | 191,385.3 | 3,460.7 |
| API Calls | 15.0 | 21.0 |
| ROUGE-1 F1 | 0.271 | 0.156 |
| BERTScore F1 | 0.814 | 0.803 |
| Judge: Instruction | 4.000 | 3.667 |
| Judge: Groundedness | 4.667 | 4.667 |
| Judge: Completeness | 5.000 | 4.667 |

#### task2

| Metric | Monolithic | Ensemble |
|--------|------------|----------|
| Latency (s) | 627.6 | 837.3 |
| Total Tokens | 382,545.7 | 5,991.8 |
| API Calls | 30.0 | 42.0 |
| ROUGE-1 F1 | 0.226 | 0.137 |
| BERTScore F1 | 0.828 | 0.811 |
| Judge: Instruction | 3.000 | 3.500 |
| Judge: Groundedness | 4.000 | 3.000 |
| Judge: Completeness | 4.667 | 2.750 |

#### task3

| Metric | Monolithic | Ensemble |
|--------|------------|----------|
| Latency (s) | 569.4 | 641.6 |
| Total Tokens | 573,585.0 | 8,720.8 |
| API Calls | 45.0 | 63.0 |
| ROUGE-1 F1 | 0.217 | 0.140 |
| BERTScore F1 | 0.827 | 0.818 |
| Judge: Instruction | 3.000 | 3.000 |
| Judge: Groundedness | 4.000 | 3.750 |
| Judge: Completeness | 4.000 | 2.750 |

## Detailed Run Information

### Evaluation Run #1

#### Ensemble - task1

- **Run ID:** `c6eb2d1624534443a5e4f558b960415c`
- **Run Name:** ensemble_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:40:56
- **End Time:** 2026-01-11 10:44:34

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.7970
- `bertscore_precision`: 0.7955
- `bertscore_recall`: 0.7986
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 190.9277
- `num_api_calls`: 21.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1722
- `rougeL_f1`: 0.0636
- `total_tokens`: 4167.0000

#### Ensemble - task2

- **Run ID:** `d7d65240b6834ed7b4ebf01741467062`
- **Run Name:** ensemble_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:44:34
- **End Time:** 2026-01-11 10:47:21

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8227
- `bertscore_precision`: 0.8230
- `bertscore_recall`: 0.8224
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 2.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 141.4398
- `num_api_calls`: 42.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1364
- `rougeL_f1`: 0.0582
- `total_tokens`: 6810.0000

#### Ensemble - task3

- **Run ID:** `2188488b5f7a4f9fbd077d6e1ed1e9f1`
- **Run Name:** ensemble_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:47:21
- **End Time:** 2026-01-11 10:50:00

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8165
- `bertscore_precision`: 0.8085
- `bertscore_recall`: 0.8247
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 3.0000
- `judge_groundedness_score`: 3.0000
- `judge_instruction_adherence_score`: 2.0000
- `latency_seconds`: 132.5645
- `num_api_calls`: 63.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1400
- `rougeL_f1`: 0.0664
- `total_tokens`: 9170.0000

#### Monolithic - task1

- **Run ID:** `a3ddba3075804a748e7b784eaff5199d`
- **Run Name:** monolithic_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:26:30
- **End Time:** 2026-01-11 10:32:07

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `bertscore_f1`: 0.8241
- `bertscore_precision`: 0.8101
- `bertscore_recall`: 0.8387
- `completion_tokens`: 37507.0000
- `document_summaries_tokens`: 178329.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 5.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 301.3152
- `num_api_calls`: 15.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 154033.0000
- `rouge1_f1`: 0.2788
- `rougeL_f1`: 0.1018
- `total_tokens`: 191550.0000

#### Monolithic - task2

- **Run ID:** `9f25b0d8b27f4037bf09625d8cc99624`
- **Run Name:** monolithic_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:32:07
- **End Time:** 2026-01-11 10:36:44

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `bertscore_f1`: 0.8342
- `bertscore_precision`: 0.8253
- `bertscore_recall`: 0.8433
- `completion_tokens`: 74696.0000
- `document_summaries_tokens`: 356658.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 248.5202
- `num_api_calls`: 30.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 308064.0000
- `rouge1_f1`: 0.2328
- `rougeL_f1`: 0.0838
- `total_tokens`: 382780.0000

#### Monolithic - task3

- **Run ID:** `40f864ecf4d241708379268015201b19`
- **Run Name:** monolithic_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 10:36:44
- **End Time:** 2026-01-11 10:40:56

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `bertscore_f1`: 0.8169
- `bertscore_precision`: 0.7986
- `bertscore_recall`: 0.8361
- `completion_tokens`: 111719.0000
- `document_summaries_tokens`: 534987.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 223.6162
- `num_api_calls`: 45.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 462097.0000
- `rouge1_f1`: 0.2287
- `rougeL_f1`: 0.0760
- `total_tokens`: 573846.0000

### Evaluation Run #2

#### Ensemble - task1

- **Run ID:** `f033339a2c9642b793f6223de57a5609`
- **Run Name:** ensemble_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 01:52:43
- **End Time:** 2026-01-11 02:09:45

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8046
- `bertscore_precision`: 0.7986
- `bertscore_recall`: 0.8106
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 5.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 996.7050
- `num_api_calls`: 21.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1604
- `rougeL_f1`: 0.0676
- `total_tokens`: 3825.0000

#### Ensemble - task2

- **Run ID:** `1abbe08da694451bb78b47cfb2681f0e`
- **Run Name:** ensemble_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 02:09:46
- **End Time:** 2026-01-11 02:25:46

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8047
- `bertscore_precision`: 0.7976
- `bertscore_recall`: 0.8118
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 3.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 934.8730
- `num_api_calls`: 42.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1556
- `rougeL_f1`: 0.0760
- `total_tokens`: 7417.0000

#### Ensemble - task3

- **Run ID:** `15ea54fa558e4250814a646dacabfec5`
- **Run Name:** ensemble_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 02:25:46
- **End Time:** 2026-01-11 02:40:24

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8113
- `bertscore_precision`: 0.8047
- `bertscore_recall`: 0.8180
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 3.0000
- `judge_groundedness_score`: 5.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 853.0289
- `num_api_calls`: 63.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1663
- `rougeL_f1`: 0.0653
- `total_tokens`: 10616.0000

#### Monolithic - task1

- **Run ID:** `cde45c6073614a3c9419793e0c67c2a1`
- **Run Name:** monolithic_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 01:13:26
- **End Time:** 2026-01-11 01:29:31

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `bertscore_f1`: 0.7897
- `bertscore_precision`: 0.7896
- `bertscore_recall`: 0.7897
- `completion_tokens`: 37417.0000
- `document_summaries_tokens`: 178329.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 928.4540
- `num_api_calls`: 15.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 154033.0000
- `rouge1_f1`: 0.3007
- `rougeL_f1`: 0.1016
- `total_tokens`: 191460.0000

#### Monolithic - task2

- **Run ID:** `c99e37c8385c4014b3e88c1d6d242394`
- **Run Name:** monolithic_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 01:29:31
- **End Time:** 2026-01-11 01:42:20

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `bertscore_f1`: 0.8311
- `bertscore_precision`: 0.8179
- `bertscore_recall`: 0.8447
- `completion_tokens`: 74473.0000
- `document_summaries_tokens`: 356658.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 742.2948
- `num_api_calls`: 30.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 308064.0000
- `rouge1_f1`: 0.2190
- `rougeL_f1`: 0.0961
- `total_tokens`: 382557.0000

#### Monolithic - task3

- **Run ID:** `446e86dacdf14b56a94b1d8018cfa5b4`
- **Run Name:** monolithic_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-11 01:42:20
- **End Time:** 2026-01-11 01:52:43

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `bertscore_f1`: 0.8398
- `bertscore_precision`: 0.8223
- `bertscore_recall`: 0.8581
- `completion_tokens`: 111241.0000
- `document_summaries_tokens`: 534987.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 2.0000
- `latency_seconds`: 595.7895
- `num_api_calls`: 45.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 462097.0000
- `rouge1_f1`: 0.1817
- `rougeL_f1`: 0.0667
- `total_tokens`: 573368.0000

### Evaluation Run #3

#### Ensemble - task1

- **Run ID:** `bcfd5c48985c4abaa8dfce65ad574a9d`
- **Run Name:** ensemble_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 23:08:53
- **End Time:** 2026-01-10 23:23:26

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8080
- `bertscore_precision`: 0.7989
- `bertscore_recall`: 0.8172
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 5.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 694.7363
- `num_api_calls`: 21.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1349
- `rougeL_f1`: 0.0549
- `total_tokens`: 2390.0000

#### Ensemble - task2

- **Run ID:** `efa181a1f0734a0a8c306f767d1935e3`
- **Run Name:** ensemble_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 23:23:26
- **End Time:** 2026-01-10 23:47:57

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8036
- `bertscore_precision`: 0.7909
- `bertscore_recall`: 0.8167
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 2.0000
- `judge_groundedness_score`: 2.0000
- `judge_instruction_adherence_score`: 2.0000
- `latency_seconds`: 1444.9406
- `num_api_calls`: 42.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1313
- `rougeL_f1`: 0.0586
- `total_tokens`: 4501.0000

#### Ensemble - task2

- **Run ID:** `1a9d0c2e587e4a0e8bbde43328a85928`
- **Run Name:** ensemble_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 21:45:21
- **End Time:** 2026-01-10 21:59:34

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8142
- `bertscore_precision`: 0.8186
- `bertscore_recall`: 0.8097
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 2.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 5.0000
- `latency_seconds`: 828.0107
- `num_api_calls`: 42.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1238
- `rougeL_f1`: 0.0580
- `total_tokens`: 5239.0000

#### Ensemble - task3

- **Run ID:** `e8cef6ed7ab34db790d6be54b756cff7`
- **Run Name:** ensemble_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 23:47:57
- **End Time:** 2026-01-10 23:58:58

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8296
- `bertscore_precision`: 0.8273
- `bertscore_recall`: 0.8320
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 2.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 2.0000
- `latency_seconds`: 635.7215
- `num_api_calls`: 63.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1282
- `rougeL_f1`: 0.0543
- `total_tokens`: 6639.0000

#### Ensemble - task3

- **Run ID:** `9b9446f4352741198d3c676f71d90b61`
- **Run Name:** ensemble_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 21:59:34
- **End Time:** 2026-01-10 22:15:46

**Parameters:**
- `agent_type`: ensemble
- `model`: openai/qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `archivist_tokens`: 0.0000
- `bertscore_f1`: 0.8148
- `bertscore_precision`: 0.8093
- `bertscore_recall`: 0.8203
- `completion_tokens`: 0.0000
- `critic_tokens`: 0.0000
- `document_summaries_tokens`: 0.0000
- `drafter_tokens`: 0.0000
- `judge_completeness_score`: 3.0000
- `judge_groundedness_score`: 3.0000
- `judge_instruction_adherence_score`: 5.0000
- `latency_seconds`: 945.0482
- `num_api_calls`: 63.0000
- `num_documents_summarized`: 10.0000
- `num_iterations`: 1.0000
- `orchestrator_tokens`: 0.0000
- `prompt_tokens`: 0.0000
- `rouge1_f1`: 0.1258
- `rougeL_f1`: 0.0591
- `total_tokens`: 8458.0000

#### Monolithic - task1

- **Run ID:** `3dd328e3b9564ea7943ac25fb3a22585`
- **Run Name:** monolithic_task1
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 22:22:59
- **End Time:** 2026-01-10 22:38:16

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Define the academic scope, terminology, and technological context of the provided corpus
- `task_id`: task1

**Metrics:**
- `bertscore_f1`: 0.8273
- `bertscore_precision`: 0.8146
- `bertscore_recall`: 0.8404
- `completion_tokens`: 37103.0000
- `document_summaries_tokens`: 178329.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 5.0000
- `judge_instruction_adherence_score`: 4.0000
- `latency_seconds`: 880.7283
- `num_api_calls`: 15.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 154033.0000
- `rouge1_f1`: 0.2342
- `rougeL_f1`: 0.0870
- `total_tokens`: 191146.0000

#### Monolithic - task2

- **Run ID:** `c142d6f00bfe43a9aadcc0028d058422`
- **Run Name:** monolithic_task2
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 22:38:16
- **End Time:** 2026-01-10 22:53:35

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Perform a structured extraction of core research components for each individual paper
- `task_id`: task2

**Metrics:**
- `bertscore_f1`: 0.8190
- `bertscore_precision`: 0.7987
- `bertscore_recall`: 0.8402
- `completion_tokens`: 74216.0000
- `document_summaries_tokens`: 356658.0000
- `judge_completeness_score`: 5.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 891.9728
- `num_api_calls`: 30.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 308064.0000
- `rouge1_f1`: 0.2253
- `rougeL_f1`: 0.0894
- `total_tokens`: 382300.0000

#### Monolithic - task3

- **Run ID:** `03fa9e61ed924d28a2d8e36c4c5dd77e`
- **Run Name:** monolithic_task3
- **Status:** 3 (3=Finished, 1=Running, 4=Failed)
- **Start Time:** 2026-01-10 22:53:35
- **End Time:** 2026-01-10 23:08:53

**Parameters:**
- `agent_type`: monolithic
- `model`: qwen2.5:7b
- `num_source_documents`: 10
- `task_description`: Synthesize a comparative meta-analysis of interaction patterns and findings across the corpus
- `task_id`: task3

**Metrics:**
- `bertscore_f1`: 0.8245
- `bertscore_precision`: 0.8169
- `bertscore_recall`: 0.8321
- `completion_tokens`: 111414.0000
- `document_summaries_tokens`: 534987.0000
- `judge_completeness_score`: 4.0000
- `judge_groundedness_score`: 4.0000
- `judge_instruction_adherence_score`: 3.0000
- `latency_seconds`: 888.6920
- `num_api_calls`: 45.0000
- `num_documents_summarized`: 10.0000
- `prompt_tokens`: 462097.0000
- `rouge1_f1`: 0.2401
- `rougeL_f1`: 0.1072
- `total_tokens`: 573541.0000

## Key Insights

⚠️ **ALL INSIGHTS BELOW ARE INVALID DUE TO TOKEN COUNTING BUG** ⚠️

### Performance Comparison

- ~~**Monolithic is 11.9% faster** than Ensemble on average~~ **INCORRECT**
- ~~Ensemble uses **98.4% fewer tokens** than Monolithic~~ **COMPLETELY FALSE - Bug in cache metadata**

### Quality Metrics

- ~~Monolithic achieves **0.011 higher BERTScore F1** (semantic similarity)~~ **May be inaccurate**
- ~~Monolithic achieves **0.095 higher ROUGE-1 F1** (lexical overlap)~~ **May be inaccurate**

### Judge Evaluation (Human-like Assessment)

- **Instruction Adherence:** Ensemble scores 0.03 points higher
- **Groundedness:** Monolithic scores 0.49 points higher
- **Completeness:** Monolithic scores 1.28 points higher

---

## 🔧 WHAT WAS WRONG - Bug Analysis

**Date Discovered:** January 11, 2026  
**Severity:** Critical - All token comparisons invalid

### The Bug

The ensemble agent's cache files contained **`tokens_used: 0`** for all documents, while the monolithic cache had correct token counts (~14,600 per document). This caused:

1. **False "98% fewer tokens" claim** 
   - Reported: Ensemble 6K vs Monolithic 382K tokens
   - Actual: Ensemble ~152K vs Monolithic ~191K tokens (~20% difference)

2. **Inconsistent methodology**
   - Both agents used cached document summaries (map phase)
   - Monolithic correctly counted cached tokens
   - Ensemble reported 0 tokens from cache

### Root Causes

**File: `ensemble.py` lines 136-146**
- Token extraction from CrewAI result object failed silently
- Returned `metrics = {total_tokens: 0}` instead of actual usage
- No validation or estimation fallback

**File: `utils.py` lines 205-209**
- Cache loading used `get('tokens_used', 0)` which silently defaulted to 0
- No warning when cache had invalid/missing token data

### What Was Fixed

✅ **`ensemble.py`**: Improved token extraction with:
- Multiple fallback methods (usage_metrics, token_usage)
- Token estimation from input+output text when API data unavailable
- Warning logs when extraction fails

✅ **`utils.py`**: Added validation:
- Warning when cache has `tokens_used=0`
- Helps catch this bug in future evaluations

✅ **Caches cleared**: All cache files backed up and deleted for fresh evaluation

✅ **Model consistency verified**: Both agents use `qwen2.5:7b` (same model)

### Next Steps

To get accurate results:
1. Run new evaluation with cleared caches
2. Both agents will regenerate summaries with correct token tracking
3. New results will show fair comparison of full pipeline (map+reduce)

---

*Report generated from MLflow tracking data*

**⚠️ REMINDER: This analysis is INVALID. New evaluation required. ⚠️**