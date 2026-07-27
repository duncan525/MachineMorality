"""

Thin convenience layer on top of similarity.py / aspect_similarity.py that works
DIRECTLY with the flat per-aspect fields already stored in the state files
(care_beliefs, justice_motives, ...), rather than requiring four concatenated
full-ToM hypotheses.

Why this exists
---------------
 These helpers take the already-separated texts and give you:

    - pairwise similarity between two agents' texts for one aspect
    - a per-aspect report (avg similarity, avg disagreement, pairwise matrix)
    - a full report across all four aspects, plus the most-disagreed aspect

Similarity is in [0, 1] (1 = identical meaning). Disagreement = 1 - similarity.

All embedding work is delegated to the existing SimilarityScorer via
get_scorer(), so the same fixed embedding model is used everywhere.
"""

# from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.similarity import get_scorer

AGENT_ORDER = ["care", "justice", "utilitarian", "common_good"]
ASPECTS = ["beliefs", "emotions", "motives", "knowledge"]


def _round(x: Any, ndigits: int = 4) -> float:
    try:
        return round(float(x), ndigits)
    except Exception:
        return 0.0


def pairwise_aspect_similarity(
    text_a: str,
    text_b: str,
    model_name: str = "qwen3-embedding:latest",
    device: Optional[str] = None,
) -> float:
    """Similarity in [0,1] between two agents' texts for a single aspect."""
    scorer = get_scorer(model_name=model_name, device=device)
    return float(scorer.text_similarity(text_a, text_b))


def aspect_report(
    aspect_texts: Dict[str, str],
    model_name: str = "qwen3-embedding:latest",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One aspect across all four agents.

    Args:
        aspect_texts: {agent_name -> that agent's text for this aspect}

    Returns per-aspect similarity/disagreement stats and the full pairwise list.
    """
    scorer = get_scorer(model_name=model_name, device=device)

    names = [a for a in AGENT_ORDER if a in aspect_texts]
    texts = [aspect_texts[a] for a in names]

    matrix = scorer.similarity_matrix(texts)

    pairwise: List[Dict[str, Any]] = []
    sims: List[float] = []
    for i, j in combinations(range(len(names)), 2):
        both_nonempty = bool(str(texts[i]).strip()) and bool(str(texts[j]).strip())
        sim = float(matrix[i, j])
        if both_nonempty:
            sims.append(sim)
        pairwise.append(
            {
                "agent_i": names[i],
                "agent_j": names[j],
                "similarity": _round(sim),
                "disagreement": _round(1.0 - sim),
                "both_nonempty": both_nonempty,
            }
        )

    if sims:
        avg_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))
        max_sim = float(np.max(sims))
        std_sim = float(np.std(sims))
    else:
        avg_sim = min_sim = max_sim = std_sim = 0.0

    return {
        "agents": names,
        "avg_similarity": _round(avg_sim),
        "avg_disagreement": _round(1.0 - avg_sim),
        "min_similarity": _round(min_sim),
        "max_similarity": _round(max_sim),
        "std_similarity": _round(std_sim),
        "pairwise": pairwise,
        "similarity_matrix": [[_round(v) for v in row] for row in matrix.tolist()],
    }


def full_similarity_report(
    state: Dict[str, Any],
    model_name: str = "qwen3-embedding:latest",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a similarity/disagreement report for ALL aspects straight from the
    flat state fields (e.g. state["care_beliefs"], state["justice_motives"]).

    Returns:
        {
          "aspects": { "beliefs": {...aspect_report...}, ... },
          "aspect_avg_disagreement": { "beliefs": 0.31, ... },
          "most_disagreed_aspect": "motives",
          "most_agreed_aspect": "emotions",
        }
    """
    report: Dict[str, Any] = {"aspects": {}, "aspect_avg_disagreement": {}}

    for aspect in ASPECTS:
        aspect_texts = {
            agent: state.get(f"{agent}_{aspect}", "") for agent in AGENT_ORDER
        }
        ar = aspect_report(aspect_texts, model_name=model_name, device=device)
        report["aspects"][aspect] = ar
        report["aspect_avg_disagreement"][aspect] = ar["avg_disagreement"]

    disagreements = report["aspect_avg_disagreement"]
    report["most_disagreed_aspect"] = max(disagreements, key=disagreements.get)
    report["most_agreed_aspect"] = min(disagreements, key=disagreements.get)

    return report