from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from similarity import get_scorer


ASPECTS = ["belief", "emotion", "knowledge", "motive"]

ASPECT_ALIASES = {
    "belief": "belief",
    "beliefs": "belief",
    "emotion": "emotion",
    "emotions": "emotion",
    "knowledge": "knowledge",
    "motive": "motive",
    "motives": "motive",
}


HypothesisInput = Union[str, Dict[str, Any]]


def _round_float(x: Any, ndigits: int = 4) -> float:
    try:
        return round(float(x), ndigits)
    except Exception:
        return 0.0

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _try_parse_json_string(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _safe_literal_decode(value: str, quote: str) -> str:
    """
    Try to decode Python-style escaped strings such as:
        'Kate\\u2019s problem'
        "The user's motive..."

    If decoding fails, return the raw value.
    """

    value = value.strip()

    try:
        return str(ast.literal_eval(f"{quote}{value}{quote}"))
    except Exception:
        return value


def _try_parse_json_string(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_key_value_aspects(text: str) -> Dict[str, str]:
    """
    Extract aspects from strings like:

        beliefs='...'
        emotions='...'
        motives="..."
        knowledge='...'

    This is designed for Pydantic-style or LangChain structured outputs.
    """

    extracted = {aspect: "" for aspect in ASPECTS}

    key_pattern = re.compile(
        r"\b(beliefs?|emotions?|knowledge|motives?)\s*=\s*(['\"])",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = list(key_pattern.finditer(text))

    if not matches:
        return extracted

    for idx, match in enumerate(matches):
        raw_key = match.group(1).strip().lower()
        quote = match.group(2)
        aspect = ASPECT_ALIASES.get(raw_key)

        if aspect not in extracted:
            continue

        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        raw_value = text[value_start:value_end].strip()

        # Remove trailing comma if present.
        if raw_value.endswith(","):
            raw_value = raw_value[:-1].strip()

        # Remove trailing quote if present.
        if raw_value.endswith(quote):
            raw_value = raw_value[:-1].strip()

        decoded_value = _safe_literal_decode(raw_value, quote)
        extracted[aspect] = _clean_text(decoded_value)

    return extracted

def _extract_heading_aspects(text: str) -> Dict[str, str]:
    """
    Extract aspects from heading-style strings like:

        Belief: ...
        Emotion: ...
        Knowledge: ...
        Motive: ...
    """

    extracted = {aspect: "" for aspect in ASPECTS}

    heading_pattern = re.compile(
        r"(?im)^\s*(beliefs?|emotions?|knowledge|motives?)\s*[:\-]\s*"
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return extracted

    for idx, match in enumerate(matches):
        raw_heading = match.group(1).strip().lower()
        aspect = ASPECT_ALIASES.get(raw_heading)

        if aspect not in extracted:
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        content = text[start:end].strip()
        extracted[aspect] = _clean_text(content)

    return extracted

def extract_tom_aspects(hypothesis: HypothesisInput) -> Dict[str, str]:
    """
    Extract belief, emotion, knowledge, and motive from one agent's full ToM hypothesis.

    Supports:

    1. Dictionary:
       {
           "belief": "...",
           "emotion": "...",
           "knowledge": "...",
           "motive": "..."
       }

    2. JSON string:
       '{"belief": "...", "emotion": "...", "knowledge": "...", "motive": "..."}'

    3. Heading format:
       Belief: ...
       Emotion: ...
       Knowledge: ...
       Motive: ...

    4. Pydantic-style / LangChain-style output:
       beliefs='...' emotions='...' motives="..." knowledge='...'

    Returns:
        {
            "belief": "...",
            "emotion": "...",
            "knowledge": "...",
            "motive": "..."
        }
    """

    extracted = {aspect: "" for aspect in ASPECTS}

    # Case 1: already a dictionary
    if isinstance(hypothesis, dict):
        # Common wrapper format: {"response": "..."}
        if "response" in hypothesis and isinstance(hypothesis["response"], str):
            return extract_tom_aspects(hypothesis["response"])

        for key, value in hypothesis.items():
            normalized_key = ASPECT_ALIASES.get(str(key).strip().lower())
            if normalized_key in extracted:
                extracted[normalized_key] = _clean_text(value)

        return extracted

    # Case 2: string
    text = _clean_text(hypothesis)

    if not text:
        return extracted

    # Case 3: JSON string
    parsed = _try_parse_json_string(text)
    if isinstance(parsed, dict):
        return extract_tom_aspects(parsed)

    # Case 4: Pydantic-style key=value format
    key_value_result = _extract_key_value_aspects(text)
    if any(key_value_result.values()):
        return key_value_result

    # Case 5: Heading-style format
    heading_result = _extract_heading_aspects(text)
    if any(heading_result.values()):
        return heading_result

    return extracted


def _normalize_agent_inputs(
    agent_hypotheses: Union[Dict[str, HypothesisInput], Sequence[HypothesisInput]],
    agent_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[HypothesisInput]]:
    """
    Normalize input into:
        agent_names: ["agent0", "agent1", ...]
        hypotheses: [hyp0, hyp1, ...]

    Supports either:
        {
            "agent0": hypothesis0,
            "agent1": hypothesis1,
            ...
        }

    or:
        [hypothesis0, hypothesis1, hypothesis2, hypothesis3]
    """

    if isinstance(agent_hypotheses, dict):
        names = list(agent_hypotheses.keys())
        hypotheses = [agent_hypotheses[name] for name in names]
        return names, hypotheses

    hypotheses = list(agent_hypotheses)

    if agent_names is None:
        names = [f"agent{i}" for i in range(len(hypotheses))]
    else:
        names = list(agent_names)

    if len(names) != len(hypotheses):
        raise ValueError("agent_names length must match number of hypotheses.")

    return names, hypotheses


def _normalize_weights(
    aspect_weights: Optional[Dict[str, float]],
    aspects: Sequence[str],
) -> Dict[str, float]:
    """
    Normalize aspect weights so they sum to 1.
    If no weights are supplied, use equal weights.
    """

    if aspect_weights is None:
        return {aspect: 1.0 / len(aspects) for aspect in aspects}

    weights = {}

    for aspect in aspects:
        weights[aspect] = float(aspect_weights.get(aspect, 0.0))

    total = sum(weights.values())

    if total <= 0:
        return {aspect: 1.0 / len(aspects) for aspect in aspects}

    return {aspect: value / total for aspect, value in weights.items()}


def compute_aspect_similarity_report(
    agent_hypotheses: Union[Dict[str, HypothesisInput], Sequence[HypothesisInput]],
    agent_names: Optional[Sequence[str]] = None,
    aspect_weights: Optional[Dict[str, float]] = None,
    model_name: str = "qwen3-embedding:latest",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function.

    Takes four full ToM hypotheses from four agents, extracts:
        belief, emotion, knowledge, motive

    Then computes:
        - pairwise similarity per aspect
        - pairwise disagreement per aspect
        - average similarity per aspect
        - average disagreement per aspect
        - weighted pairwise full-ToM similarity using aspect weights
        - final average similarity/disagreement across all aspects

    Similarity:
        1.0 = highly similar
        0.0 = highly different

    Disagreement:
        disagreement = 1.0 - similarity
    """

    names, hypotheses = _normalize_agent_inputs(agent_hypotheses, agent_names)

    if len(hypotheses) != 4:
        raise ValueError(
            f"Expected exactly 4 agent hypotheses, but received {len(hypotheses)}."
        )

    weights = _normalize_weights(aspect_weights, ASPECTS)
    scorer = get_scorer(model_name=model_name, device=device)

    extracted_by_agent: Dict[str, Dict[str, str]] = {}

    for name, hypothesis in zip(names, hypotheses):
        extracted_by_agent[name] = extract_tom_aspects(hypothesis)

    report: Dict[str, Any] = {
        "agents": names,
        "aspect_weights": weights,
        "extracted_aspects": extracted_by_agent,
        "aspects": {},
        "weighted_pairwise": [],
        "overall": {},
    }

    # Store pairwise similarities per pair across aspects.
    # Example:
    # pair_aspect_scores[("agent0", "agent1")]["belief"] = 0.77
    pair_aspect_scores: Dict[Tuple[str, str], Dict[str, float]] = {}

    for aspect in ASPECTS:
        texts = [extracted_by_agent[name].get(aspect, "") for name in names]
        nonempty_agents = [name for name, text in zip(names, texts) if text.strip()]
        missing_agents = [name for name, text in zip(names, texts) if not text.strip()]

        matrix = scorer.similarity_matrix(texts)

        pairwise_records = []
        valid_pair_sims = []

        for i, j in combinations(range(len(names)), 2):
            name_i = names[i]
            name_j = names[j]
            text_i = texts[i]
            text_j = texts[j]

            sim = float(matrix[i, j])
            disagreement = 1.0 - sim

            both_nonempty = bool(text_i.strip()) and bool(text_j.strip())

            if both_nonempty:
                valid_pair_sims.append(sim)

            pair_key = (name_i, name_j)
            pair_aspect_scores.setdefault(pair_key, {})
            pair_aspect_scores[pair_key][aspect] = sim

            pairwise_records.append(
                {
                    "agent_i": name_i,
                    "agent_j": name_j,
                    "similarity": _round_float(sim),
                    "disagreement": _round_float(disagreement),
                    "both_nonempty": both_nonempty,
                }
            )

        if valid_pair_sims:
            avg_similarity = float(np.mean(valid_pair_sims))
            min_similarity = float(np.min(valid_pair_sims))
            max_similarity = float(np.max(valid_pair_sims))
            std_similarity = float(np.std(valid_pair_sims))
        else:
            avg_similarity = 0.0
            min_similarity = 0.0
            max_similarity = 0.0
            std_similarity = 0.0

        avg_disagreement = 1.0 - avg_similarity

        report["aspects"][aspect] = {
            "texts": {name: extracted_by_agent[name].get(aspect, "") for name in names},
            "missing_agents": missing_agents,
            "num_nonempty": len(nonempty_agents),
            "similarity_matrix": [
                [_round_float(value) for value in row]
                for row in matrix.tolist()
            ],
            "pairwise": pairwise_records,
            "avg_similarity": _round_float(avg_similarity),
            "avg_disagreement": _round_float(avg_disagreement),
            "min_similarity": _round_float(min_similarity),
            "max_similarity": _round_float(max_similarity),
            "std_similarity": _round_float(std_similarity),
        }

    # Compute weighted full-ToM pairwise similarity across aspects.
    weighted_pairwise_sims = []

    for (name_i, name_j), aspect_scores in pair_aspect_scores.items():
        weighted_similarity = 0.0

        for aspect in ASPECTS:
            weighted_similarity += weights[aspect] * aspect_scores.get(aspect, 0.0)

        weighted_disagreement = 1.0 - weighted_similarity
        weighted_pairwise_sims.append(weighted_similarity)

        report["weighted_pairwise"].append(
            {
                "agent_i": name_i,
                "agent_j": name_j,
                "weighted_similarity": _round_float(weighted_similarity),
                "weighted_disagreement": _round_float(weighted_disagreement),
                "aspect_similarities": {
                    aspect: _round_float(aspect_scores.get(aspect, 0.0))
                    for aspect in ASPECTS
                },
            }
        )

    # Overall aspect averages
    aspect_avg_similarities = {
        aspect: report["aspects"][aspect]["avg_similarity"]
        for aspect in ASPECTS
    }

    aspect_avg_disagreements = {
        aspect: report["aspects"][aspect]["avg_disagreement"]
        for aspect in ASPECTS
    }

    overall_avg_similarity = float(np.mean(list(aspect_avg_similarities.values())))
    overall_avg_disagreement = 1.0 - overall_avg_similarity

    weighted_overall_similarity = sum(
        weights[aspect] * aspect_avg_similarities[aspect]
        for aspect in ASPECTS
    )
    weighted_overall_disagreement = 1.0 - weighted_overall_similarity

    most_disagreed_aspect = max(
        aspect_avg_disagreements,
        key=aspect_avg_disagreements.get,
    )

    most_agreed_aspect = max(
        aspect_avg_similarities,
        key=aspect_avg_similarities.get,
    )

    report["overall"] = {
        "aspect_avg_similarities": {
            aspect: _round_float(value)
            for aspect, value in aspect_avg_similarities.items()
        },
        "aspect_avg_disagreements": {
            aspect: _round_float(value)
            for aspect, value in aspect_avg_disagreements.items()
        },
        "overall_avg_similarity": _round_float(overall_avg_similarity),
        "overall_avg_disagreement": _round_float(overall_avg_disagreement),
        "weighted_overall_similarity": _round_float(weighted_overall_similarity),
        "weighted_overall_disagreement": _round_float(weighted_overall_disagreement),
        "most_disagreed_aspect": most_disagreed_aspect,
        "most_agreed_aspect": most_agreed_aspect,
    }

    return report