# Free Scaling

$0 test-time scaling using [NVIDIA NIM](https://build.nvidia.com) free tier. Ask multiple models, get one reliable answer.

**Two modes:**
- **`smart_vote()`** — cascade: routes to the best expert for the task type, escalates only on uncertainty. ~1 API call per question on average.
- **`vote()`** — flat ensemble: asks N models, majority vote.

## Why

Single models hallucinate. Ensembles don't (as much). NIM gives you 15 models for free. This library turns them into a single reliable oracle — routing questions to the right expert, escalating only when uncertain, and weighting votes by measured accuracy.

**Zero cost. Zero dependencies. Just stdlib Python + a free API key.**

## Setup

1. Get a free API key at [build.nvidia.com](https://build.nvidia.com) — sign in, pick any model, click "Get API Key"
2. One key works for all models:
   ```bash
   export NVIDIA_API_KEY="nvapi-..."
   ```
3. Clone this repo and use it:
   ```bash
   git clone https://github.com/isotrivial/free-scaling.git
   cd free-scaling
   ```

No pip install needed — stdlib only (Python 3.10+).

## Quick Start

### CLI
```bash
# Smart cascade (recommended)
python3 -m nim_ensemble.cli smart "Is eval(input()) safe?" --answers "SAFE,VULNERABLE"
# → VULNERABLE (conf=100%, primary, 1 call, 0.5s)

# Flat ensemble vote
python3 -m nim_ensemble.cli ask "Is 91 prime?" --panel general --answers "YES,NO"

# List available models and panels
python3 -m nim_ensemble.cli models
python3 -m nim_ensemble.cli panels
```

### Python
```python
from nim_ensemble import smart_vote

result = smart_vote("Is this code safe?", answer_patterns=["SAFE", "VULNERABLE"])
print(result.answer)      # VULNERABLE
print(result.calls_made)  # 1 (resolved at stage 1)
print(result.confidence)  # 1.0
```

## How the Cascade Works

```
Question → classify task type (code/compliance/reasoning/factual/nuance)
        → call best expert for that type (1 call)
        → confident? → done
        → uncertain? → call arbiter (mistral-large)
        → still split? → full panel, weighted vote
```

Most questions resolve at stage 1 with a single API call.

## Capability Profiling

No hardcoded scores — profile models on your own tasks:

```bash
python3 -m nim_ensemble.capability_map --models llama-3.3 qwen-80b mistral-large --trials 3
```

Generates `capability_map.json` with per-model accuracy, latency, and error correlations. The cascade loads it automatically for data-driven routing.

Without profiling, sensible defaults work out of the box.

## 15 Models Included

| Speed | Models |
|-------|--------|
| Fast (<1s) | llama-3.3 70B, gemma-27b, mistral-nemotron, kimi-k2, qwen-80b |
| Medium (1-3s) | mistral-large 675B, llama-405b, qwen-397b, deepseek-v3 |
| Slow (3s+) | minimax-m2.5 (thinking) |

All free via NVIDIA NIM. One API key covers everything.

## Use Cases

- **Code review** — "Is this code vulnerable?" across multiple models
- **Fact-checking** — consensus answers to factual questions
- **Compliance auditing** — check if outputs follow rules/policies
- **Agent self-evaluation** — verify agent behavior against specs
- **Any binary/categorical judgment** — route to experts, vote on uncertainty

## Also an OpenClaw Skill

If you use [OpenClaw](https://github.com/openclaw/openclaw), install via:
```bash
clawhub install free-scaling
```

See [SKILL.md](SKILL.md) for the full agent skill reference.

## License

MIT
