---
name: free-scaling
description: "$0 test-time scaling with NVIDIA NIM free tier. Smart cascade routes questions to the best free model based on measured capability profiles, escalating only on uncertainty. 15 models, 7 capability categories, data-driven panels. Use for auditing, code review, fact-checking, compliance, or any judgment task."
---

# NIM Ensemble

$0 multi-model reasoning using NVIDIA NIM free tier. Two modes:

- **`smart_vote()`** — cascade: routes to the best expert for the task type, escalates only on uncertainty. Average 1.2 API calls per question.
- **`vote()`** — flat ensemble: asks N models, majority vote. Simple but uses more calls.

## Setup

1. Go to [build.nvidia.com](https://build.nvidia.com) and sign in (free NVIDIA account)
2. Pick any model (e.g. [Llama 3.3 70B](https://build.nvidia.com/meta/llama-3_3-70b-instruct)) and click **"Get API Key"**
3. One key works for all NIM models — no per-model setup needed
4. Set it in your environment:
   ```bash
   export NVIDIA_API_KEY="nvapi-..."
   ```
5. No pip dependencies — stdlib only (Python 3.10+)

## Quick Start

```python
from nim_ensemble import smart_vote

# Auto-classifies as "code", routes to qwen-80b (100% on code)
result = smart_vote("Is eval(input()) safe?", answer_patterns=["SAFE", "VULNERABLE"])
print(result.answer)      # VULNERABLE
print(result.calls_made)  # 1 (resolved at stage 1, no escalation)
print(result.confidence)  # 1.0
```

## CLI

```bash
# Smart cascade (recommended)
python3 -m nim_ensemble.cli smart "Is this code vulnerable?" --answers "SAFE,VULNERABLE"
# → VULNERABLE (conf=100%, primary, 1 call, 0.5s)

# Flat ensemble
python3 -m nim_ensemble.cli ask "Is X true?" --panel general --answers "YES,NO"

# Classify task type
python3 -m nim_ensemble.cli classify "Is this SQL injection?"
# → code → ['qwen-80b', 'mistral-large', 'mistral-nemotron']

# List models and panels
python3 -m nim_ensemble.cli models
python3 -m nim_ensemble.cli panels

# Benchmark all models on a question
python3 -m nim_ensemble.cli bench "Is 91 prime? YES or NO." --speed fast
```

## How Smart Cascade Works

```
Question → classify task type (code/compliance/reasoning/factual/nuance)
        → call best expert for that type (1 call)
        → confident? (weight ≥ 85%) → done
        → uncertain? → call arbiter (mistral-large, 100% all categories)
        → still split? → full panel, weighted vote by measured accuracy
```

Most questions resolve at stage 1. Hallucinating models never get called because the capability map routes around their blind spots.

## Capability Map (measured, not assumed)

Every model was profiled across 7 categories with multiple trials:

| Model | Accuracy | Latency | Strengths | Blind Spots |
|-------|----------|---------|-----------|-------------|
| mistral-large | 100% | 1.0s | All categories | None (arbiter) |
| llama-3.3 | 96% | 1.9s | nuance, code, agreeableness | factual 78% |
| qwen-80b | 95% | 0.7s | code, nuance, agreeableness | reasoning 67% |
| mistral-nemotron | 95% | 1.1s | factual, reasoning, code | agreeableness 67% |
| gemma-27b | 93% | 1.2s | reasoning, code, agreeableness | factual 67% |
| kimi-k2 | 79% | 0.6s | instruction, factual, nuance | **agreeableness 22%, code 44%** |
| qwen-397b | 58% | 4.4s | nuance, calibration | **code 0%, agreeableness 33%** |

Key insights:
- **Size ≠ accuracy**: qwen-397B (58%) < qwen-80B (95%)
- **Thinking ≠ judgment**: MiniMax M2.5 scores 0% on nuance despite chain-of-thought
- **Speed ≠ accuracy**: kimi-k2 (0.6s, 79%) vs qwen-80b (0.7s, 95%)

## Data-Driven Panels

Panels are built from measured capability data, not intuition:

| Panel | Models | Use Case |
|-------|--------|----------|
| `general` | mistral-large, llama-3.3, qwen-80b | Default, >95% accuracy |
| `fast` | qwen-80b, mistral-nemotron, gemma-27b | All ≤1.2s |
| `code` | qwen-80b, mistral-large, mistral-nemotron | Code review (kimi-k2 excluded) |
| `compliance` | llama-3.3, qwen-80b, mistral-large | Behavioral/rule compliance |
| `reasoning` | mistral-large, llama-3.3, mistral-nemotron | Logic, math, inference |
| `nuance` | llama-3.3, qwen-80b, mistral-large | Subtle violations, gray areas |
| `arbiter` | mistral-large | Tiebreaker (100% all categories) |
| `max` | 5 models | High-stakes, maximum confidence |

## Python API

```python
from nim_ensemble import smart_vote, smart_vote_batch, vote, vote_batch

# Smart cascade (recommended)
result = smart_vote("Is X correct?", answer_patterns=["YES", "NO"])
# result.answer, result.confidence, result.stage, result.calls_made

# Batch (parallel)
results = smart_vote_batch([
    {"text": "Q1?", "task_type": "code", "answer_patterns": ["SAFE", "VULNERABLE"]},
    {"text": "Q2?", "task_type": "compliance", "answer_patterns": ["COMPLIANT", "VIOLATED"]},
], max_parallel=5)

# Flat ensemble
result = vote("Is X true?", panel="general", answer_patterns=["YES", "NO"])

# Single model call
from nim_ensemble import call_model
answer, raw = call_model("Is X true?", "mistral-large")
```

## Profiling Your Own Models

```bash
# Profile specific models (3 trials each)
python3 -m nim_ensemble.capability_map --models llama-3.3 kimi-k2 --trials 3

# Profile all fast models
python3 -m nim_ensemble.capability_map --speed fast --trials 2

# Output: capability_map.json with profiles + correlation matrix + routing policy
```

## Prompt Tips

For best results with ensemble voting:
- Ask for the answer on the **first line**: "Answer YES or NO on the first line, then explain."
- Give **explicit answer options**: "Answer SAFE, UNSAFE, or NEEDS_REVIEW."
- Include **context/evidence** in the question, not just the judgment call.

## Architecture

```
nim_ensemble/
├── __init__.py       # Public API: smart_vote, vote, call_model
├── cascade.py        # Smart cascade with capability routing
├── voter.py          # Flat ensemble voting engine
├── models.py         # Model registry + data-driven panels
├── parser.py         # Answer extraction (thinking models, word boundaries)
├── cli.py            # CLI interface
├── benchmark.py      # Single-trial model profiling
└── capability_map.py # Multi-trial profiling with error correlation
```

## Requirements

- `NVIDIA_API_KEY` environment variable (free at [build.nvidia.com](https://build.nvidia.com))
- Python 3.10+
- No pip dependencies (stdlib only, uses `urllib` for API calls)
