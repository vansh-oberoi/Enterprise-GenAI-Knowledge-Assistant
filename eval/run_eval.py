"""Evaluate ranked source retrieval independently of LLM answer generation.

Start the API and index every document referenced by eval/questions.jsonl, then run:
    python -m eval.run_eval --top-k 5
"""
import argparse
import json
from pathlib import Path

import requests


def score_case(expected_sources: set[str], ranked_sources: list[str]) -> dict:
    """Compute source-level ranking metrics for one answerable query."""
    if not expected_sources:
        return {"hit_at_k": None, "recall_at_k": None, "reciprocal_rank": None}

    relevant_ranks = [
        rank for rank, source in enumerate(ranked_sources, start=1)
        if source in expected_sources
    ]
    relevant_found = len(set(ranked_sources) & expected_sources)
    return {
        "hit_at_k": int(bool(relevant_ranks)),
        "recall_at_k": relevant_found / len(expected_sources),
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10")

    cases = [
        json.loads(line)
        for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise SystemExit("No evaluation cases found in eval/questions.jsonl")

    answerable_scores = []
    unanswerable_cases = 0
    for case in cases:
        response = requests.post(
            f"{args.api_url.rstrip('/')}/retrieve",
            json={"question": case["question"], "top_k": args.top_k},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        ranked_sources = [item["source"] for item in result.get("sources", [])]
        expected = set(case.get("expected_sources", []))
        scores = score_case(expected, ranked_sources)
        if expected:
            answerable_scores.append(scores)
        else:
            unanswerable_cases += 1

        print(json.dumps({
            "question": case["question"],
            "expected_sources": sorted(expected),
            "ranked_sources": ranked_sources,
            **scores,
        }))

    count = len(answerable_scores)
    summary = {
        "cases_total": len(cases),
        "answerable_cases": count,
        "unanswerable_cases": unanswerable_cases,
        "top_k": args.top_k,
        "hit_at_k": (
            sum(item["hit_at_k"] for item in answerable_scores) / count if count else None
        ),
        "recall_at_k": (
            sum(item["recall_at_k"] for item in answerable_scores) / count if count else None
        ),
        "mrr_at_k": (
            sum(item["reciprocal_rank"] for item in answerable_scores) / count if count else None
        ),
        "note": "Source-level retrieval metrics only; they do not measure answer faithfulness or citation entailment.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
