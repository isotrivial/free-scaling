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
        print(format_report(results, summary))
    
    return results, summary


# Answer classification for the report
PASS_ANSWERS = {"COMPLIANT", "FOLLOWED", "HEALTHY", "CLEAN", "CONSISTENT", 
                "EFFICIENT", "RELIABLE", "WITHIN_CAPACITY", "PATCHED", "OK"}
WARN_ANSWERS = {"DRIFTING", "MINOR_DRIFT", "PARTIALLY", "FLAKY", "UNCLEAR",
                "WASTEFUL", "OVER_CAPACITY", "HAS_ISSUES"}
FAIL_ANSWERS = {"VIOLATED", "REPEATING", "FAILING", "INCONSISTENT", "NOT_PATCHED",
                "DEGRADING", "CRITICAL"}


def classify_severity(answer: str) -> str:
    """Classify an answer into OK/WARNING/CRITICAL."""
    a = answer.upper()
    if a in PASS_ANSWERS:
        return "OK"
    if a in FAIL_ANSWERS:
        return "CRITICAL"
    if a in WARN_ANSWERS:
        return "WARNING"
    return "WARNING"  # unknown → warning


def format_report(results: list[dict], summary: dict) -> str:
    """Format a structured audit report."""
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    
    lines.append(f"\n{'='*60}")
    lines.append(f"  SYSTEM AUDIT REPORT")
    lines.append(f"  {ts} | k={summary['k']} | {summary['total_calls']} API calls | {summary['elapsed_s']:.0f}s | $0")
    lines.append(f"{'='*60}")
    
    # Group by severity
    ok, warn, crit, err = [], [], [], []
    for r in results:
        sev = classify_severity(r["answer"])
        if r["answer"] == "ERROR":
            err.append(r)
        elif sev == "OK":
            ok.append(r)
        elif sev == "CRITICAL":
            crit.append(r)
        else:
            warn.append(r)
    
    # Critical first
    if crit:
        lines.append(f"\n🔴 CRITICAL ({len(crit)})")
        lines.append("-" * 40)
        for r in crit:
            conf = f"{r['confidence']:.0%}" if isinstance(r['confidence'], float) else r['confidence']
            models = " ".join(f"{m}={a}" for m, a in r.get("models", []))
            lines.append(f"  [{r['id']}] {r['answer']} ({conf})")
            if models:
                lines.append(f"    votes: {models}")
    
    if warn:
        lines.append(f"\n🟡 WARNING ({len(warn)})")
        lines.append("-" * 40)
        for r in warn:
            conf = f"{r['confidence']:.0%}" if isinstance(r['confidence'], float) else r['confidence']
            models = " ".join(f"{m}={a}" for m, a in r.get("models", []))
            lines.append(f"  [{r['id']}] {r['answer']} ({conf})")
            if models:
                lines.append(f"    votes: {models}")
    
    if ok:
        lines.append(f"\n✅ OK ({len(ok)})")
        lines.append("-" * 40)
        for r in ok:
            conf = f"{r['confidence']:.0%}" if isinstance(r['confidence'], float) else r['confidence']
            lines.append(f"  [{r['id']}] {r['answer']} ({conf})")
    
    if err:
        lines.append(f"\n❌ ERROR ({len(err)})")
        lines.append("-" * 40)
        for r in err:
            lines.append(f"  [{r['id']}] {r.get('error', 'unknown')}")
    
    # Summary bar
    lines.append(f"\n{'='*60}")
    total = len(results)
    lines.append(f"  {total} checks: ✅ {len(ok)} OK | 🟡 {len(warn)} WARNING | 🔴 {len(crit)} CRITICAL | ❌ {len(err)} ERROR")
    
    health = len(ok) / total * 100 if total else 0
    if health >= 80:
        grade = "HEALTHY"
    elif health >= 60:
        grade = "NEEDS ATTENTION"
    else:
        grade = "DEGRADED"
    lines.append(f"  Health: {health:.0f}% — {grade}")
    lines.append(f"{'='*60}")
    
    return "\n".join(lines)


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
