#!/usr/bin/env python3
"""Audit preset — run behavioral compliance checks via scale().

Uses the system-audit collector + question generator, pipes everything
through nim_ensemble.scale(k=N) for ensemble judgment.

Usage:
    python3 -m presets.audit                    # k=3 (default)
    python3 -m presets.audit -k 5               # more models
    python3 -m presets.audit --json              # machine-readable
    python3 -m presets.audit --skip-collect      # reuse cached state
"""

import argparse
import json
import os
import re
import sys
import time

# Find the audit scripts
AUDIT_SCRIPTS = os.path.join(
    os.environ.get("OPENCLAW_WORKSPACE", os.path.expanduser("~/.openclaw/workspace")),
    "skills", "system-audit", "scripts"
)

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, AUDIT_SCRIPTS)


def extract_patterns(question_text: str) -> list[str]:
    """Extract answer options from 'Answer X, Y, or Z' in question text."""
    # Match "Answer X, Y, or Z" or "Answer X or Y"
    m = re.search(
        r'Answer\s+([\w_]+)(?:\s*\(.*?\))?'        # first option
        r'(?:,\s*([\w_]+)(?:\s*\(.*?\))?)?'          # second (optional)
        r'(?:,?\s*or\s+([\w_]+)(?:\s*\(.*?\))?)?',   # third (optional)
        question_text
    )
    if m:
        return [x.upper() for x in m.groups() if x]
    return []


def collect_state(timeout_per_job: int = 10):
    """Collect system state using the audit collector."""
    from collect import collect_all
    return collect_all()


def run_audit(k: int = 3, verbose: bool = False, json_output: bool = False,
              state: dict = None, timeout: int = 15):
    """Run the full audit pipeline through scale(k=)."""
    from nim_ensemble import scale
    from questions import generate_questions
    
    t0 = time.time()
    
    # Step 1: Collect
    if state is None:
        if verbose:
            print("Collecting system state...", file=sys.stderr)
        state = collect_state()
    
    # Step 2: Generate questions
    questions = generate_questions(state)
    if verbose:
        print(f"Generated {len(questions)} questions", file=sys.stderr)
    
    # Step 3: Run each through scale(k=)
    results = []
    total_calls = 0
    
    for q in questions:
        qid = q.get("id", "?")
        text = q.get("question_text", "")
        
        if not text:
            continue
        
        # Skip pre-answered (deterministic) questions
        if q.get("pre_answered"):
            results.append({
                "id": qid,
                "category": q.get("category", "?"),
                "answer": q.get("answer", "?"),
                "confidence": 1.0,
                "calls": 0,
                "elapsed_s": 0,
                "source": "deterministic",
                "models": [],
            })
            if not json_output:
                print(f"  [{qid}] {q.get('answer', '?')} (deterministic)")
            continue
        
        # Extract answer patterns from question text
        patterns = extract_patterns(text)
        if not patterns:
            patterns = ["COMPLIANT", "DRIFTING", "VIOLATED"]  # default
        
        try:
            result = scale(text, k=k, answer_patterns=patterns)
            total_calls += result.calls_made
            
            models_detail = [(m, a) for m, a, _ in result.votes]
            
            results.append({
                "id": qid,
                "category": q.get("category", "?"),
                "answer": result.answer,
                "confidence": result.confidence,
                "calls": result.calls_made,
                "elapsed_s": round(result.elapsed_s, 1),
                "source": f"scale-{k}",
                "models": models_detail,
            })
            
            if not json_output:
                conf_str = f"{result.confidence:.0%}"
                models_str = ", ".join(f"{m}={a}" for m, a in models_detail)
                print(f"  [{qid}] {result.answer} ({conf_str}, {result.elapsed_s:.1f}s) — {models_str}")
        
        except Exception as e:
            results.append({
                "id": qid,
                "category": q.get("category", "?"),
                "answer": "ERROR",
                "confidence": 0,
                "calls": 0,
                "elapsed_s": 0,
                "source": "error",
                "error": str(e),
                "models": [],
            })
            if not json_output:
                print(f"  [{qid}] ERROR: {e}")
    
    elapsed = time.time() - t0
    
    # Summary
    summary = {
        "total_questions": len(results),
        "total_calls": total_calls,
        "elapsed_s": round(elapsed, 1),
        "k": k,
        "by_answer": {},
    }
    for r in results:
        ans = r["answer"]
        summary["by_answer"][ans] = summary["by_answer"].get(ans, 0) + 1
    
    if json_output:
        print(json.dumps({"results": results, "summary": summary}, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Audit complete: {len(results)} checks, {total_calls} API calls, {elapsed:.0f}s")
        print(f"k={k} | ", end="")
        for ans, count in sorted(summary["by_answer"].items()):
            print(f"{ans}:{count} ", end="")
        print()
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run system audit via scale(k=)")
    parser.add_argument("-k", type=int, default=3, help="Models per question (default 3)")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--state", help="Path to cached state JSON")
    args = parser.parse_args()
    
    state = None
    if args.state:
        with open(args.state) as f:
            state = json.load(f)
    
    try:
        run_audit(k=args.k, verbose=args.verbose, json_output=args.json, state=state)
    except KeyboardInterrupt:
        sys.exit(130)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
