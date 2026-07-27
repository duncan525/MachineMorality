"""
Epistemic-disagreement scoring for the multi-agent ToM debate.

WHAT THIS DOES
----------------------------
For one aspect (belief / emotion / motive / knowledge) we have four candidate
texts, one per agent (care, justice, utilitarian, common_good). This module
asks each agent's own model -- filtered through that agent's schema + lens --
"which of these four candidates is the most accurate?" and reads the model's
*real* probability over the four letter options via Ollama logprobs.

That gives a 4x4 scorecard (4 raters x 4 candidates). From it we compute the
BALD / Jensen-Shannon "epistemic disagreement":

    epistemic = H(mean opinion)  -  mean( H(each agent's opinion) )

High epistemic = each agent is individually confident, but they point at
DIFFERENT candidates -> a genuine values clash worth debating.
Low epistemic  = either everyone agrees, or everyone is equally unsure (the
input is just ambiguous) -> not a real clash.


If logprobs come back empty (older Ollama, or a model/endpoint that doesn't
expose them), `get_agent_opinion` raises LogprobsUnavailable so the caller can
fall back to embedding-only selection.
"""

# from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Any

import requests


# Fixed agent order used everywhere in the pipeline.
AGENT_ORDER = ["care", "justice", "utilitarian", "common_good"]

# Candidate letters, one per agent, in AGENT_ORDER.
LETTERS = ["A", "B", "C", "D"]

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# logprob floor for a letter that never appears in top_logprobs (~ "impossible")
_LOGPROB_FLOOR = -20.0


class LogprobsUnavailable(RuntimeError):
    """Raised when the Ollama response carried no usable logprobs."""


def _read_model_name(model_file: str = "states/model.txt") -> str:
    with open(model_file, "r") as f:
        return f.read().strip()


def _softmax_from_logprobs(logprob_by_letter: Dict[str, float]) -> Dict[str, float]:
    """Exponentiate the four letter logprobs and renormalize to sum to 1."""
    exp_vals = {k: math.exp(v) for k, v in logprob_by_letter.items()}
    total = sum(exp_vals.values())
    if total <= 0.0:
        # Degenerate; fall back to uniform so downstream entropy is well-defined.
        n = len(logprob_by_letter)
        return {k: 1.0 / n for k in logprob_by_letter}
    return {k: v / total for k, v in exp_vals.items()}


def _build_prompt(
    schema: str,
    lens_name: str,
    user_prompt: str,
    aspect: str,
    candidates: Sequence[str],
) -> str:
    """
    One clean decision point: force a single-letter answer so we can read the
    model's probability mass over exactly A/B/C/D at that position.
    """
    options = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(candidates))
    return f"""Your ethics schema: {schema}

Your ethical lens: {lens_name}

User prompt: {user_prompt}

Here are four possible interpretations of the user's {aspect.upper()},
each proposed by a different observer:

{options}

From the perspective of your own schema and lens, which single interpretation
is the most accurate? Respond with EXACTLY one letter: A, B, C, or D.
Do not explain. Output only the letter."""


def get_agent_opinion(
    schema: str,
    lens_name: str,
    user_prompt: str,
    aspect: str,
    candidates: Sequence[str],
    model: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    top_logprobs: int = 20,
    timeout: int = 60,
) -> Dict[str, float]:
    """
    Return one agent's probability distribution over the 4 candidates, e.g.
        {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}

    Keys are LETTERS (A..D), aligned to `candidates` order (= AGENT_ORDER).
    """
    if len(candidates) != 4:
        raise ValueError(f"Expected 4 candidates, got {len(candidates)}.")

    # if model is None:
    #     model = _read_model_name()

    model = "gemma4"

    prompt = _build_prompt(schema, lens_name, user_prompt, aspect, candidates)

    resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "options": {
                "num_predict": 1,   # single-token answer
                "temperature": 0,   # read the raw distribution, undistorted
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    logprobs = data.get("logprobs")
    if not logprobs:
        raise LogprobsUnavailable(
            "Ollama returned no logprobs. Requires Ollama >= 0.12.11 and a model "
            "that exposes token logprobs. Fall back to embedding-only selection."
        )

    considered = logprobs[0].get("top_logprobs") or []
    # Some builds omit top_logprobs and only give the chosen token; handle both.
    if not considered and "token" in logprobs[0]:
        considered = [logprobs[0]]

    # Pull each letter's logprob (leading-space tokenization tolerated via strip()).
    logprob_by_letter: Dict[str, float] = {}
    for letter in LETTERS:
        match = next(
            (t for t in considered if str(t.get("token", "")).strip() == letter),
            None,
        )
        logprob_by_letter[letter] = float(match["logprob"]) if match else _LOGPROB_FLOOR

    return _softmax_from_letter_logprobs(logprob_by_letter)


def _softmax_from_letter_logprobs(logprob_by_letter: Dict[str, float]) -> Dict[str, float]:
    return _softmax_from_logprobs(logprob_by_letter)


# --------------------------------------------------------------------------- #
# Information-theoretic aggregation                                            #
# --------------------------------------------------------------------------- #

def _entropy(dist: Sequence[float]) -> float:
    """Shannon entropy in nats, robust to zeros."""
    h = 0.0
    for p in dist:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def epistemic_disagreement(opinions: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Given the 4 agents' opinions (each a dict over A..D), compute the
    uncertainty decomposition.

    Returns:
        {
          "total":      H(mean opinion),          # total uncertainty
          "mean_per_agent":  mean per-agent entropy,    # shared confusion / ambiguity
          "epistemic":  total - mean_per_agent,         # genuine cross-agent disagreement
        }

    `epistemic` is the number to rank aspects by. It equals the generalized
    Jensen-Shannon divergence across the four opinion distributions, and the
    BALD mutual-information score with agents playing the role of model samples.
    """
    letters = LETTERS

    # Mean (mixture) opinion across agents.
    mean_opinion = [
        sum(op[letter] for op in opinions) / len(opinions) for letter in letters
    ]
    total = _entropy(mean_opinion)

    # Average individual entropy.
    aleatoric = sum(_entropy([op[letter] for letter in letters]) for op in opinions)
    aleatoric /= len(opinions)

    epistemic = total - aleatoric
    # Numerical guard: epistemic (JSD/MI) is >= 0 in exact arithmetic.
    if epistemic < 0.0:
        epistemic = 0.0

    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


def aspect_logprob_report(
    aspect: str,
    schemas: Dict[str, str],
    lens_names: Dict[str, str],
    candidates_by_agent: Dict[str, str],
    user_prompt: str,
    model: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    """
    Full logprob pass for ONE aspect.

    Inputs are keyed by agent name (AGENT_ORDER):
        schemas[agent]            -> that agent's schema text
        lens_names[agent]         -> "care" / "justice" / ...
        candidates_by_agent[agent]-> that agent's hypothesis text for this aspect

    Returns a report dict including the 4x4 scorecard and the
    total/aleatoric/epistemic decomposition.
    """
    # Fixed candidate order = AGENT_ORDER, so letter i belongs to AGENT_ORDER[i].
    candidates = [candidates_by_agent[a] for a in AGENT_ORDER]

    opinions: List[Dict[str, float]] = []
    scorecard: Dict[str, Dict[str, float]] = {}
    model = "qwen3.5"
    for rater in AGENT_ORDER:
        op = get_agent_opinion(
            schema=schemas[rater],
            lens_name=lens_names.get(rater, rater),
            user_prompt=user_prompt,
            aspect=aspect,
            candidates=candidates,
            model=model,
            ollama_url=ollama_url,
        )
        opinions.append(op)
        # Re-key the letters back to the agent each candidate came from, for readability.
        scorecard[rater] = {
            AGENT_ORDER[i]: op[LETTERS[i]] for i in range(len(AGENT_ORDER))
        }

    decomp = epistemic_disagreement(opinions)

    return {
        "aspect": aspect,
        "candidate_owner_by_letter": {LETTERS[i]: AGENT_ORDER[i] for i in range(4)},
        "scorecard": scorecard,          # rater -> {candidate_owner -> prob}
        "total_uncertainty": decomp["total"],
        "aleatoric_uncertainty": decomp["aleatoric"],
        "epistemic_disagreement": decomp["epistemic"],
    }