"""
test_disagreement.py

Standalone harness to inspect the most-disagreed aspect for a given state,
using BOTH signals:
    - embedding disagreement   (aspect_pairwise.full_similarity_report)
    - epistemic disagreement   (logprob_disagreement, BALD/JSD over agents)

It prints a full per-aspect report and the most-disagreed component from each
signal plus the combined pick, without touching the live LangGraph pipeline.

USAGE
-----
Put this next to similarity.py / aspect_pairwise.py / logprob_disagreement.py
(and the states/ folder), then:

    # embedding-only (no Ollama chat model needed, just the embedding model):
    python test_disagreement.py --state states/stage1_state.txt --embed-only

    # full (embedding + logprobs; needs Ollama >= 0.12.11 + states/model.txt):
    python test_disagreement.py --state states/stage1_state.txt

Or import and call directly:

    from test_disagreement import run_disagreement_report
    report = run_disagreement_report(state_dict)          # state already loaded
    print(report["combined"]["most_disagreed"])
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from aspect_pairwise import full_similarity_report, AGENT_ORDER, ASPECTS
from logprob_disagreement import (
    aspect_logprob_report,
    LogprobsUnavailable,
)


# Same schema/lens mapping the orchestrator uses. When running OUTSIDE the
# pipeline we read the schema files directly; if they're missing we fall back to
# short placeholder schemas so the harness still runs (logprob scores will be
# less meaningful, but the plumbing is exercised).
def _load_schemas(schema_dir: str = "states/schemas") -> Dict[str, str]:
    mapping = {
        "care": "schema0.txt",
        "justice": "schema1.txt",
        "utilitarian": "schema2.txt",
        "common_good": "schema3.txt",
    }
    schemas: Dict[str, str] = {}
    for agent, fname in mapping.items():
        try:
            with open(f"{schema_dir}/{fname}", "r") as f:
                schemas[agent] = f.read().strip()
        except FileNotFoundError:
            schemas[agent] = f"You reason using the {agent} ethical lens."
    return schemas


AGENT_LENS_NAMES = {
    "care": "care ethics",
    "justice": "justice",
    "utilitarian": "utilitarian",
    "common_good": "common good",
}

# Blend weight: 0.0 = logprob only, 1.0 = embedding only, 0.5 = equal.
ALPHA = 0.5


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def run_disagreement_report(
    state: Dict[str, Any],
    embed_only: bool = False,
    schema_dir: str = "states/schemas",
    embedding_model: str = "qwen3-embedding:latest",
    model: str = "qwen3.5",
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Compute embedding + (optionally) epistemic disagreement for every aspect.

    Returns a dict:
        {
          "embedding": {aspect: disagreement, ..., "most_disagreed": ...},
          "epistemic": {aspect: disagreement, ..., "most_disagreed": ...} | None,
          "combined":  {aspect: score, ..., "most_disagreed": ...},
        }
    """
    # ---- (1) embedding disagreement ------------------------------------------
    sim_report = full_similarity_report(state, model_name=embedding_model)
    embed_dis: Dict[str, float] = {
        aspect: float(sim_report["aspect_avg_disagreement"][aspect])
        for aspect in ASPECTS
    }
    embed_block = dict(embed_dis)
    embed_block["most_disagreed"] = max(embed_dis, key=embed_dis.get)

    result: Dict[str, Any] = {
        "embedding": embed_block,
        "epistemic": None,
        "combined": None,
        "_full_similarity_report": sim_report,
    }

    if embed_only:
        combined = dict(embed_dis)
        combined["most_disagreed"] = max(embed_dis, key=embed_dis.get)
        result["combined"] = combined
        return result

    # ---- (2) epistemic disagreement (logprobs) -------------------------------
    schemas = _load_schemas(schema_dir)
    epistemic: Dict[str, float] = {}
    logprob_scorecards: Dict[str, Any] = {}
    try:
        for aspect in ASPECTS:
            candidates_by_agent = {
                agent: state.get(f"{agent}_{aspect}", "") for agent in AGENT_ORDER
            }
            rep = aspect_logprob_report(
                aspect=aspect,
                schemas=schemas,
                lens_names=AGENT_LENS_NAMES,
                candidates_by_agent=candidates_by_agent,
                user_prompt=state["user_prompt"],
                model=model,
            )
            epistemic[aspect] = float(rep["epistemic_disagreement"])
            logprob_scorecards[aspect] = rep
    except LogprobsUnavailable as e:
        print(f"** logprobs unavailable: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        print("** returning embedding-only result **")
        combined = dict(embed_dis)
        combined["most_disagreed"] = max(embed_dis, key=embed_dis.get)
        result["combined"] = combined
        return result
    except Exception as e:
        # Surface the REAL error instead of hiding it behind a generic message.
        print(f"** logprob scoring failed: {type(e).__name__}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        print("** returning embedding-only result **")
        combined = dict(embed_dis)
        combined["most_disagreed"] = max(embed_dis, key=embed_dis.get)
        result["combined"] = combined
        return result

    epi_block = dict(epistemic)
    epi_block["most_disagreed"] = max(epistemic, key=epistemic.get)
    result["epistemic"] = epi_block
    result["_logprob_scorecards"] = logprob_scorecards

    # ---- combine -------------------------------------------------------------
    e_norm = _minmax(embed_dis)
    p_norm = _minmax(epistemic)
    combined_scores = {
        aspect: ALPHA * e_norm[aspect] + (1.0 - ALPHA) * p_norm[aspect]
        for aspect in ASPECTS
    }
    combined = dict(combined_scores)
    combined["most_disagreed"] = max(combined_scores, key=combined_scores.get)
    result["combined"] = combined

    return result


def print_report(result: Dict[str, Any]) -> None:
    """Pretty-print the report to the console."""
    print("\n" + "=" * 60)
    print("DISAGREEMENT REPORT")
    print("=" * 60)

    print("\n[1] EMBEDDING DISAGREEMENT (1 - avg pairwise similarity)")
    for aspect in ASPECTS:
        print(f"    {aspect:9s}  {result['embedding'][aspect]:.4f}")
    print(f"    --> most disagreed (embedding): {result['embedding']['most_disagreed']}")

    if result["epistemic"] is not None:
        print("\n[2] EPISTEMIC DISAGREEMENT (BALD / JSD over agents, nats)")
        for aspect in ASPECTS:
            print(f"    {aspect:9s}  {result['epistemic'][aspect]:.4f}")
        print(f"    --> most disagreed (epistemic): {result['epistemic']['most_disagreed']}")
    else:
        print("\n[2] EPISTEMIC DISAGREEMENT: skipped (embed-only or unavailable)")

    print("\n[3] COMBINED (min-max normalized blend, ALPHA=%.2f)" % ALPHA)
    for aspect in ASPECTS:
        print(f"    {aspect:9s}  {result['combined'][aspect]:.4f}")
    print("\n" + "-" * 60)
    print(f"  MOST DISAGREED COMPONENT: {result['combined']['most_disagreed'].upper()}")
    print("-" * 60 + "\n")


def _load_state(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect most-disagreed ToM aspect.")
    parser.add_argument(
        "--state",
        default="states/stage1_state.txt",
        help="Path to a populated state json (e.g. states/stage1_state.txt).",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Skip logprobs; use embedding disagreement only.",
    )
    parser.add_argument(
        "--schema-dir",
        default="states/schemas",
        help="Directory holding schema0..schema3.txt.",
    )
    parser.add_argument(
        "--model",
        default="gemma4",
        help="Ollama chat model used for logprob scoring (must expose logprobs).",
    )
    args = parser.parse_args()

    state = _load_state(args.state)
    result = run_disagreement_report(
        state,
        embed_only=args.embed_only,
        schema_dir=args.schema_dir,
        model=args.model,
    )
    print_report(result)


if __name__ == "__main__":
    main()