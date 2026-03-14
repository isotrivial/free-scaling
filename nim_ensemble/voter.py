"""Core voting engine — parallel NIM API calls with majority vote."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .models import MODELS, PANELS, NIM_ENDPOINT, get_model, get_panel, is_thinking
from .parser import extract_content, parse_answer


@dataclass
class VoteResult:
    """Result of an ensemble vote."""
    answer: str
    confidence: str          # "3/3", "2/3", etc.
    votes: list[str]         # individual model answers
    raw_responses: list[str] # full model outputs
    models_used: list[str]
    unanimous: bool = False
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _get_nim_key() -> str:
    """Load NIM API key from environment."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com"
        )
    return key


def call_model(
    prompt: str,
    model_alias: str,
    system_prompt: str = None,
    max_tokens: int = 150,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """Call a single NIM model. Returns (parsed_answer, raw_content)."""
    model_info = get_model(model_alias)
    api_model = model_info["id"]
    key = _get_nim_key()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Thinking models need more tokens and time
    effective_max_tokens = max_tokens
    curl_timeout = 30
    if model_info.get("thinking"):
        effective_max_tokens = max(max_tokens * 4, 600)
        curl_timeout = 45
    
    payload = {
        "model": api_model,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
    }
    
    try:
        # Use urllib instead of curl to avoid leaking API key in process args
        import urllib.request
        import urllib.error
        
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            NIM_ENDPOINT,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=curl_timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return "ERROR", f"HTTP {e.code}: {body[:200]}"
        except urllib.error.URLError as e:
            return "ERROR", f"URL error: {e.reason}"
        
        proc = None  # compatibility marker
        
        if not raw:
            return "ERROR", "Empty response"
        
        if not raw.startswith("{"):
            return "ERROR", f"Non-JSON: {raw[:100]}"
        
        resp = json.loads(raw)
        
        if "error" in resp or resp.get("status") in (404, 410):
            err = resp.get("error", {})
            detail = resp.get("detail", "")
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return "ERROR", f"API error: {msg or detail}"
        
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = extract_content(msg)
        
        return parse_answer(content), content
    
    except subprocess.TimeoutExpired:
        return "ERROR", f"Timeout after {curl_timeout}s"
    except Exception as e:
        return "ERROR", str(e)


def vote(
    question: str,
    panel: str | list[str] = "balanced",
    system_prompt: str = None,
    answer_patterns: list[str] = None,
    short_circuit: bool = True,
    max_tokens: int = 150,
    parallel: bool = True,
) -> VoteResult:
    """Run a question through a model panel and return majority vote.
    
    Args:
        question: The question/prompt to vote on
        panel: Panel name (str) or list of model aliases
        system_prompt: Optional system prompt for all models
        answer_patterns: Custom answer patterns for parsing
        short_circuit: Stop early when first 2 models agree
        max_tokens: Max tokens per model call
        parallel: Run models in parallel (default True)
    
    Returns:
        VoteResult with majority answer and individual votes
    """
    t0 = time.time()
    
    # Resolve panel
    if isinstance(panel, str):
        model_aliases = get_panel(panel)
    else:
        model_aliases = panel
    
    votes = []
    raw_responses = []
    errors = []
    
    # Normalize answer patterns to uppercase
    if answer_patterns:
        answer_patterns = [p.strip().upper() for p in answer_patterns]
    
    def _call(alias):
        ans, raw = call_model(question, alias, system_prompt, max_tokens)
        # Re-parse with custom patterns if provided
        if answer_patterns and ans not in answer_patterns and ans != "ERROR":
            ans = parse_answer(raw, patterns=answer_patterns)
        return alias, ans, raw
    
    models_used_ordered = []
    
    if parallel and not short_circuit:
        # Full parallel execution — collect results with model identity
        with ThreadPoolExecutor(max_workers=len(model_aliases)) as pool:
            futures = {pool.submit(_call, alias): alias for alias in model_aliases}
            for fut in as_completed(futures):
                alias, ans, raw = fut.result()
                models_used_ordered.append(alias)
                votes.append(ans)
                raw_responses.append(raw)
                if ans == "ERROR":
                    errors.append(f"{alias}: {raw[:100]}")
    
    elif short_circuit:
        # Sequential with early exit on agreement
        for i, alias in enumerate(model_aliases):
            _, ans, raw = _call(alias)
            votes.append(ans)
            raw_responses.append(raw)
            if ans == "ERROR":
                errors.append(f"{alias}: {raw[:100]}")
            
            # Check for short circuit after 2 votes
            if i >= 1 and short_circuit:
                non_error = [v for v in votes if v != "ERROR"]
                if len(non_error) >= 2 and len(set(non_error)) == 1:
                    break
    
    else:
        # Sequential, no short circuit
        for alias in model_aliases:
            _, ans, raw = _call(alias)
            votes.append(ans)
            raw_responses.append(raw)
            if ans == "ERROR":
                errors.append(f"{alias}: {raw[:100]}")
    
    # Count votes (exclude errors)
    non_error_votes = [v for v in votes if v != "ERROR"]
    if not non_error_votes:
        return VoteResult(
            answer="ERROR",
            confidence=f"0/{len(votes)}",
            votes=votes,
            raw_responses=raw_responses,
            models_used=models_used_ordered if models_used_ordered else model_aliases[:len(votes)],
            errors=errors,
            elapsed_s=time.time() - t0,
        )
    
    # Majority vote
    from collections import Counter
    counts = Counter(non_error_votes)
    majority_answer, majority_count = counts.most_common(1)[0]
    total_valid = len(non_error_votes)
    
    sc_tag = " (short-circuit)" if len(votes) < len(model_aliases) else ""
    
    return VoteResult(
        answer=majority_answer,
        confidence=f"{majority_count}/{total_valid}{sc_tag}",
        votes=votes,
        raw_responses=raw_responses,
        models_used=models_used_ordered if models_used_ordered else model_aliases[:len(votes)],
        unanimous=majority_count == total_valid and total_valid >= 2,
        errors=errors,
        elapsed_s=time.time() - t0,
    )


def vote_batch(
    questions: list[dict],
    parallel_questions: int = 5,
    **vote_kwargs,
) -> list[VoteResult]:
    """Vote on multiple questions in parallel.
    
    Each question dict must have 'text' key, and optionally:
    - 'panel', 'system_prompt', 'answer_patterns', 'short_circuit'
    
    Returns list of VoteResults in same order as input.
    """
    results = [None] * len(questions)
    
    def _vote_one(idx, q):
        kwargs = {**vote_kwargs}
        kwargs.update({k: v for k, v in q.items() if k != "text" and k != "id"})
        text = q.get("text", q.get("question_text", ""))
        return idx, vote(text, **kwargs)
    
    with ThreadPoolExecutor(max_workers=parallel_questions) as pool:
        futures = {
            pool.submit(_vote_one, i, q): i 
            for i, q in enumerate(questions)
        }
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
    
    return results
