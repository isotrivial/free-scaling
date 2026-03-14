"""NIM model registry — verified working models with characteristics."""

NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Verified working models (benchmarked 2026-03-13)
MODELS = {
    # Fast tier (<1s)
    "llama-3.3": {
        "id": "meta/llama-3.3-70b-instruct",
        "speed": "fast",
        "family": "meta",
        "params": "70B",
        "thinking": False,
    },
    "gemma-27b": {
        "id": "google/gemma-3-27b-it",
        "speed": "fast",
        "family": "google",
        "params": "27B",
        "thinking": False,
    },
    "mistral-nemotron": {
        "id": "mistralai/mistral-nemotron",
        "speed": "fast",
        "family": "mistral",
        "params": "?",
        "thinking": False,
    },
    "jamba-mini": {
        "id": "ai21labs/jamba-1.5-mini-instruct",
        "speed": "fast",
        "family": "ai21",
        "params": "?",
        "thinking": False,
    },
    "dracarys-70b": {
        "id": "abacusai/dracarys-llama-3.1-70b-instruct",
        "speed": "fast",
        "family": "abacusai",
        "params": "70B",
        "thinking": False,
    },

    # Medium tier (1-2s)
    "kimi-k2": {
        "id": "moonshotai/kimi-k2-instruct",
        "speed": "medium",
        "family": "moonshot",
        "params": "?",
        "thinking": False,
    },
    "mistral-large": {
        "id": "mistralai/mistral-large-3-675b-instruct-2512",
        "speed": "medium",
        "family": "mistral",
        "params": "675B",
        "thinking": False,
    },
    "llama-405b": {
        "id": "meta/llama-3.1-405b-instruct",
        "speed": "medium",
        "family": "meta",
        "params": "405B",
        "thinking": False,
    },
    "qwen-397b": {
        "id": "qwen/qwen3.5-397b-a17b",
        "speed": "medium",
        "family": "qwen",
        "params": "397B",
        "thinking": False,
    },
    "qwen-80b": {
        "id": "qwen/qwen3-next-80b-a3b-instruct",
        "speed": "medium",
        "family": "qwen",
        "params": "80B",
        "thinking": False,
    },
    "mistral-medium": {
        "id": "mistralai/mistral-medium-3-instruct",
        "speed": "medium",
        "family": "mistral",
        "params": "?",
        "thinking": False,
    },
    "nemotron-super-49b": {
        "id": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "speed": "medium",
        "family": "nvidia",
        "params": "49B",
        "thinking": False,
    },
    "deepseek-v3.1-term": {
        "id": "deepseek-ai/deepseek-v3.1-terminus",
        "speed": "slow",
        "family": "deepseek",
        "params": "?",
        "thinking": False,
    },

    # Thinking models (5-40s, need special handling)
    "minimax-m2.5": {
        "id": "minimaxai/minimax-m2.5",
        "speed": "slow",
        "family": "minimax",
        "params": "?",
        "thinking": True,
        "think_style": "inline",  # <think>...</think> in content field
    },
    "kimi-k2.5": {
        "id": "moonshotai/kimi-k2.5",
        "speed": "slow",
        "family": "moonshot",
        "params": "?",
        "thinking": True,
        "think_style": "separate",  # reasoning_content field, content can be null
    },
}

# Models that give false passes (too agreeable) — excluded from panels by default
EXCLUDED = {"phi-4-mini", "phi-3-small", "deepseek-v3.1", "chatglm3-6b", "italia-10b"}

# Data-driven panels (from capability_map.json, 2026-03-14)
# Each panel is designed around measured strengths, not assumed capability.
PANELS = {
    # === General purpose ===
    # Top 3 by overall accuracy, architecture-diverse (Mistral/Meta/Qwen)
    "general": ["mistral-large", "llama-3.3", "qwen-80b"],
    # Fast + accurate: all ≤1.2s, all ≥93%
    "fast": ["qwen-80b", "mistral-nemotron", "gemma-27b"],
    
    # === Task-specific (data-driven) ===
    # Code: all 100% on code category. AVOID kimi-k2 (44%) and qwen-397b (0%)
    "code": ["qwen-80b", "mistral-large", "mistral-nemotron"],
    # Compliance/behavioral: all 100% on nuance + agreeableness
    "compliance": ["llama-3.3", "qwen-80b", "mistral-large"],
    # Reasoning: all 100% on reasoning + factual
    "reasoning": ["mistral-large", "llama-3.3", "mistral-nemotron"],
    # Nuance: best at subtle violations + pushback
    "nuance": ["llama-3.3", "qwen-80b", "mistral-large"],
    
    # === Escalation ===
    # Arbiter: single model, 100% across ALL 7 categories
    "arbiter": ["mistral-large"],
    # Max: 5 models for highest confidence
    "max": ["mistral-large", "llama-3.3", "qwen-80b", "mistral-nemotron", "gemma-27b"],
    
    # === Legacy (kept for compatibility) ===
    "balanced": ["mistral-large", "llama-3.3", "qwen-80b"],  # alias for general
    "deep": ["mistral-large", "llama-3.3", "qwen-80b"],      # alias for general
    "diverse": ["llama-3.3", "qwen-80b", "mistral-large"],   # same models, different order
}

# Models to AVOID for specific task types (measured weaknesses)
AVOID = {
    "code": {"kimi-k2", "qwen-397b"},           # kimi 44%, qwen 0%
    "agreeableness": {"kimi-k2", "qwen-397b"},   # kimi 22%, qwen 33%
    "nuance": {"minimax-m2.5"},                   # 0% on nuance despite thinking
}


def get_model(alias: str) -> dict:
    """Look up model by alias. Returns model dict with 'id' key."""
    if alias in MODELS:
        return MODELS[alias]
    # Try matching by full NIM model ID
    for m in MODELS.values():
        if m["id"] == alias:
            return m
    raise KeyError(f"Unknown model: {alias}. Available: {list(MODELS.keys())}")


def get_panel(name: str) -> list[str]:
    """Get a panel by name. Returns list of model aliases."""
    if name in PANELS:
        return PANELS[name]
    raise KeyError(f"Unknown panel: {name}. Available: {list(PANELS.keys())}")


def is_thinking(alias: str) -> bool:
    """Check if a model is a thinking model (needs special handling)."""
    m = MODELS.get(alias, {})
    return m.get("thinking", False)


def list_models(speed: str = None, family: str = None) -> list[str]:
    """List model aliases, optionally filtered."""
    results = []
    for alias, m in MODELS.items():
        if speed and m.get("speed") != speed:
            continue
        if family and m.get("family") != family:
            continue
        results.append(alias)
    return results
