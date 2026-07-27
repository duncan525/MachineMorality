"""
orchestrator_rubric.py

New component-selection scoring for the debate orchestrator that REPLACES the
logprob-epistemic signal with a moral-criterion rubric, and demotes embeddings
to a cheap pre-filter.

DESIGN (as agreed):
  Step 1  cheap embedding dissimilarity on ALL 4 components -> shortlist top K
  Step 2  criterion rubric (evidential support / belief projection /
          plausibility) on the SHORTLIST ONLY, via a neutral framework agent
  Step 3  per shortlisted component:
             quality      = rubric total (0-100), higher = better hypothesis
             need_to_debate = (1 - quality/100)          # low quality -> debate
          non-shortlisted components get need_to_debate = 0 (won't be picked)

The orchestrator's existing 3-way logic (resolve / plateau / continue) is reused
UNCHANGED, but it now consumes `need_to_debate` instead of `combined
disagreement`. Semantics line up: high value = "worth debating", and it FALLS as
the hypothesis improves across rounds -> RESOLVE/PLATEAU fire correctly.

This module is import-light so it can be unit-tested without Ollama. The two
functions that need the live agent / embedding scorer receive them as arguments
(dependency injection) so tests can stub them.
"""

from __future__ import annotations

import re
from typing import Dict, List, Callable, Any, Optional


ASPECTS_DEFAULT = ["beliefs", "emotions", "motives", "knowledge"]
AGENTS_DEFAULT = ["care", "justice", "utilitarian", "common_good"]

# how many components survive the embedding pre-filter into the (costly) rubric
SHORTLIST_K = 2


# robust numeric parsing (shared with voting_fix philosophy: never slice)    
def _extract_rubric_scores(structured: str, raw: str) -> Dict[str, Optional[float]]:
    """
    get the three rubric criteria from the model output, tolerant to format.
    Returns {'plausibility':x,'evidential_support':y,'belief_projection':z}
    with None where a value could not be found.

    Strategy per field: labelled match first, clamped to its max; caller sums.
    """
    fields = {
        "plausibility": 50.0,          # max 50
        "evidential_support": 20.0,     # max 20
        "belief_projection": 30.0,      # max 30
    }
    out: Dict[str, Optional[float]] = {}
    for key, cap in fields.items():
        m = re.search(rf"{key}\s*=?\s*['\"]?\s*(-?\d+(?:\.\d+)?)", structured)
        if not m:
            m = re.search(rf"{key}\s*=?\s*['\"]?\s*(-?\d+(?:\.\d+)?)", raw)
        if m:
            v = float(m.group(1))
            v = max(0.0, min(v, cap))   # clamp into [0, cap]
            out[key] = v
        else:
            out[key] = None
    return out


def rubric_total(scores: Dict[str, Optional[float]]) -> Optional[float]:
    """Sum the three criteria into a 0-100 quality. None if any is missing."""
    if any(v is None for v in scores.values()):
        return None
    return float(sum(scores.values()))


# Step 1: embedding pre-filter                                                
def embedding_shortlist(
    state: Dict[str, Any],
    full_similarity_report_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    aspects: List[str],
    k: int = SHORTLIST_K,
    already_resolved: Optional[List[str]] = None,
) -> List[str]:
    """
    Rank components by embedding dissimilarity; return the top-k UNRESOLVED ones.
    Cheap: one embedding per hypothesis, no generation.
    """
    resolved = set(already_resolved or [])
    report = full_similarity_report_fn(state)
    dis = {
        a: float(report["aspect_avg_disagreement"].get(a, 0.0))
        for a in aspects if a not in resolved
    }
    if not dis:
        return []
    ranked = sorted(dis, key=dis.get, reverse=True)
    return ranked[:k]


# Step 2: rubric on ONE component                                             
def build_rubric_prompt(user_prompt: str, component: str, combined_hypothesis: str,
                        format_instructions: str) -> str:
    """The 3-criterion rubric prompt (ported from the framework-agent design)."""
    return f""" User prompt: [{user_prompt}]
                    Hypothesis about the {component} of the subject of the user prompt in regards
                    to the situation described in the user prompt: [{combined_hypothesis}]

                    Score the hypothesis for each of the following criteria:
                    1. Evidential support (max score: 20): Does the information in the user prompt
                    directly support each claim made about the subject's thought process, in regards
                    to the subject's {component}? A low score (closer to 1) means that the hypothesis
                    makes no, or very few, claims about the subject's state of mind that are supported by
                    evidence from the user prompt. A high score (closer to 20) means that
                    all or almost all of the claims about the subject's state of mind in the
                    hypothesis are well-supported by evidence from the user prompt. *Be critical.* It's
                    okay to give low scores.
                    2. Belief projection (max score: 30): Does the hypothesis project ethical
                    thoughts onto the subject of the user prompt? Specifically, do the ethics agents
                    that produced the  hypothesis (care, justice, utilitarian, and common-good ethics
                    agents) inappropriately project their own ethical worldviews about the subject's
                    {component} onto the subject? A low score (closer to 1) means that the ethics agents
                    significantly and implausibly project their own perspectives onto the subject's
                    mental state. A high score (closer to 30) means that the ethics agents engage in
                    little to no belief projection. *Be critical.* It's okay to give low scores.
                    3. Plausibility (max score: 50): Given the user prompt, could each hypothesis'
                    claims plausibly reflect the subject's inner state of {component}? A low score
                    (closer to 1) means that the hypothesis' claims generally cannot plausibly reflect
                    the subject's {component}. A high score (closer to 50) means that the hypothesis'
                    claims about the subject's state of mind are generally quite plausible. *Be critical.*
                    It's okay to give low scores.
                    
                    For each criterion, your response should contain a single number (the hypothesis'
                    score for that criterion). For example, your response for 'plausibility' should
                    look like '[score]', where [score] is replaced by a single number representing
                    your score for the criterion.
                    
                    Return ONLY valid JSON in the format specified by:
                    {format_instructions}"""


def rubric_score_component(
    state: Dict[str, Any],
    component: str,
    framework_agent,                 # live agent (injected)
    sub_score_parser,                # PydanticOutputParser(SubScoreSchema)
    sub_score_retry_parser,          # OutputFixingParser
    parse_output_fn: Callable,       # host parse_output
    agents: List[str],
) -> Optional[float]:
    """
    Run the rubric on one component's COMBINED hypothesis (all 4 agents joined).
    Returns quality in [0,100], or None if scoring failed.
    """
    combined_hypothesis = " ".join(
        str(state.get(f"{agent}_{component}", "")) for agent in agents
    )

    prompt = build_rubric_prompt(
        user_prompt=state["user_prompt"],
        component=component,
        combined_hypothesis=combined_hypothesis,
        format_instructions=sub_score_parser.get_format_instructions(),
    )

    raw = framework_agent.invoke({"messages": [{"role": "system", "content": prompt}]})
    response = raw["messages"][-1].content
    structured = parse_output_fn(state, sub_score_retry_parser, response)

    scores = _extract_rubric_scores(structured, response)
    total = rubric_total(scores)

    print(f"** rubric[{component}] plaus={scores['plausibility']} "
          f"evid={scores['evidential_support']} proj={scores['belief_projection']} "
          f"-> total={total} **")
    return total


# Step 3: combined per-component "need to debate"                             
def compute_need_to_debate(
    state: Dict[str, Any],
    full_similarity_report_fn: Callable,
    framework_agent,
    sub_score_parser,
    sub_score_retry_parser,
    parse_output_fn: Callable,
    aspects: List[str] = None,
    agents: List[str] = None,
    resolved: Optional[List[str]] = None,
    k: int = SHORTLIST_K,
) -> Dict[str, float]:
    """
    Returns {component: need_to_debate in [0,1]} for ALL aspects.
    Only shortlisted components get a real (rubric-derived) value; the rest are 0
    so they will not be selected. need_to_debate = 1 - quality/100.
    """
    aspects = aspects or ASPECTS_DEFAULT
    agents = agents or AGENTS_DEFAULT

    need = {a: 0.0 for a in aspects}

    shortlist = embedding_shortlist(
        state, full_similarity_report_fn, aspects, k=k, already_resolved=resolved
    )
    print(f"** embedding shortlist (top {k}): {shortlist} **")

    for component in shortlist:
        quality = rubric_score_component(
            state, component, framework_agent,
            sub_score_parser, sub_score_retry_parser, parse_output_fn, agents,
        )
        if quality is None:
            # scoring failed -> treat as maximally in need (so it gets attention),
            # but not above a genuine measured low-quality; use 1.0 as sentinel.
            need[component] = 1.0
        else:
            need[component] = 1.0 - (quality / 100.0)

    print("** need-to-debate per component **")
    for a in aspects:
        print(f"   {a:9s}  need={need[a]:.3f}")
    return need