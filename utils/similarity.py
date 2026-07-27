"""
similarity.py

embedding-level semantic text-similarity utilities for the RL debate orchestrator.

use this module inside your state_updater.py to compute:
- similarity between two hypotheses
- redundancy among multiple agent hypotheses
- redundancy among feedback comments
- revision delta between old and revised hypotheses
- diversity score among agents

Install:
    pip install langchain-ollama numpy

Recommended usage:
    from similarity import text_similarity, redundancy_score, revision_delta

    sim = text_similarity(text_a, text_b)
    red = redundancy_score([agent0_hypothesis, agent1_hypothesis, agent2_hypothesis])
    delta = revision_delta(old_hypothesis, revised_hypothesis)
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"

TextInput = Union[str, None]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class SimilarityScorer:
    """
    Reusable semantic similarity scorer.

    Important:
    - Use ONE fixed embedding model across all experiments.
    - Do not compare embeddings produced by different embedding models.
    - This class returns similarity values in [0, 1].
    - Ollama controls model placement; device is kept only for API compatibility.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: Optional[str] = None
    cache_embeddings: bool = True

    _model: Any = field(default=None, init=False, repr=False)
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _embedding_dim: Optional[int] = field(default=None, init=False, repr=False)

    def _load_model(self) -> Any:
        """
        Lazily create the Ollama embedding client only when needed.
        """
        if self._model is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "langchain-ollama is required. Install it with:\n"
                    "    pip install langchain-ollama"
                ) from exc

            self._model = OllamaEmbeddings(model=self.model_name)

        return self._model

    @staticmethod
    def _normalize_embedding(embedding: Sequence[float]) -> np.ndarray:
        """
        Convert an Ollama embedding into a normalized float32 vector.
        """
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.ndim != 1:
            vector = vector.reshape(-1)

        if vector.size == 0:
            raise RuntimeError("Ollama returned an empty embedding.")

        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector = vector / norm

        return vector.astype(np.float32)

    def _embed_cleaned_text(self, cleaned: str) -> np.ndarray:
        """
        Embed already-cleaned non-empty text with Ollama.
        """
        model = self._load_model()
        embedding = self._normalize_embedding(model.embed_query(cleaned))
        self._embedding_dim = int(embedding.shape[0])
        return embedding

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            probe = self._embed_cleaned_text("embedding dimension probe")
            self._embedding_dim = int(probe.shape[0])

        if self._embedding_dim <= 0:
            raise RuntimeError("Could not determine embedding dimension.")

        return self._embedding_dim

    @staticmethod
    def clean_text(text: TextInput) -> str:
        """
        Normalize common output formats from your pipeline.

        Handles:
        - None
        - raw strings
        - JSON strings like {"response": "..."}
        - Pydantic-style strings like response='...'
        - Pydantic-style strings like response="..."

        Returns:
            Cleaned plain text.
        """
        if text is None:
            return ""

        s = str(text).strip()
        if not s:
            return ""

        # Try JSON: {"response": "..."}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and "response" in parsed:
                s = str(parsed["response"])
        except Exception:
            pass

        s = s.strip()

        # Try Pydantic-style: response='...' or response="..."
        if s.startswith("response="):
            inner = s[len("response=") :].strip()
            try:
                parsed_inner = ast.literal_eval(inner)
                s = str(parsed_inner)
            except Exception:
                # Fallback: strip one layer of quotes if present.
                if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in {"'", '"'}:
                    s = inner[1:-1]
                else:
                    s = inner

        # Remove null-like outputs.
        if s.lower() in {"none", "null", "nan"}:
            return ""

        # Normalize whitespace.
        s = " ".join(s.split())

        return s.strip()

    def embed_text(self, text: TextInput) -> np.ndarray:
        """
        Convert a single text into a normalized embedding vector.

        Empty text returns a zero vector.
        """
        cleaned = self.clean_text(text)

        if not cleaned:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self.cache_embeddings and cleaned in self._cache:
            return self._cache[cleaned].copy()

        embedding = self._embed_cleaned_text(cleaned)

        if self.cache_embeddings:
            self._cache[cleaned] = embedding.copy()

        return embedding

    def embed_texts(self, texts: Sequence[TextInput]) -> np.ndarray:
        """
        Convert multiple texts into a 2D embedding matrix.

        Shape:
            (num_texts, embedding_dim)
        """
        if texts is None:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cleaned_texts = [self.clean_text(text) for text in texts]

        if not cleaned_texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings: List[Optional[np.ndarray]] = [None] * len(cleaned_texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, cleaned in enumerate(cleaned_texts):
            if not cleaned:
                continue
            if self.cache_embeddings and cleaned in self._cache:
                embeddings[idx] = self._cache[cleaned].copy()
            else:
                uncached_indices.append(idx)
                uncached_texts.append(cleaned)

        if uncached_texts:
            model = self._load_model()
            raw_embeddings = model.embed_documents(uncached_texts)

            if len(raw_embeddings) != len(uncached_texts):
                raise RuntimeError(
                    "Ollama returned a different number of embeddings than requested."
                )

            for idx, cleaned, raw_embedding in zip(
                uncached_indices,
                uncached_texts,
                raw_embeddings,
            ):
                embedding = self._normalize_embedding(raw_embedding)
                self._embedding_dim = int(embedding.shape[0])
                embeddings[idx] = embedding

                if self.cache_embeddings:
                    self._cache[cleaned] = embedding.copy()

        zero_embedding: Optional[np.ndarray] = None

        for idx, embedding in enumerate(embeddings):
            if embedding is not None:
                continue

            if zero_embedding is None:
                zero_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

            embeddings[idx] = zero_embedding.copy()

        return np.vstack(embeddings).astype(np.float32)

    @staticmethod
    def _cosine_0_to_1(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity and clamp it into [0, 1].

        Since embeddings are normalized, cosine is usually already meaningful.
        We clamp negative values to 0 because RL state features are easier to
        work with in a 0-to-1 range.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        if cosine < 0.0:
            return 0.0
        if cosine > 1.0:
            return 1.0

        return cosine

    def text_similarity(self, text_a: TextInput, text_b: TextInput) -> float:
        """
        Semantic similarity between two texts.

        Returns:
            float in [0, 1]
            1.0 = very similar
            0.0 = very different or empty input
        """
        cleaned_a = self.clean_text(text_a)
        cleaned_b = self.clean_text(text_b)

        if not cleaned_a or not cleaned_b:
            return 0.0

        emb_a = self.embed_text(cleaned_a)
        emb_b = self.embed_text(cleaned_b)

        return self._cosine_0_to_1(emb_a, emb_b)

    def similarity_matrix(self, texts: Sequence[TextInput]) -> np.ndarray:
        """
        Pairwise semantic similarity matrix.

        Diagonal is 1.0 for non-empty texts and 0.0 for empty texts.
        """
        if texts is None:
            return np.zeros((0, 0), dtype=np.float32)

        cleaned = [self.clean_text(t) for t in texts]
        n = len(cleaned)

        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        matrix = np.zeros((n, n), dtype=np.float32)

        if not any(cleaned):
            return matrix

        embeddings = self.embed_texts(cleaned)

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0 if cleaned[i] else 0.0
                else:
                    matrix[i, j] = self._cosine_0_to_1(embeddings[i], embeddings[j])

        return matrix

    def pairwise_similarities(
        self,
        texts: Sequence[TextInput],
        names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return pairwise similarity records.

        Useful for debugging which agents are redundant.
        """
        if texts is None:
            return []

        n = len(texts)

        if names is None:
            names = [f"text_{i}" for i in range(n)]

        if len(names) != n:
            raise ValueError("Length of names must match length of texts.")

        results: List[Dict[str, Any]] = []

        for i, j in combinations(range(n), 2):
            sim = self.text_similarity(texts[i], texts[j])
            results.append(
                {
                    "i": i,
                    "j": j,
                    "name_i": names[i],
                    "name_j": names[j],
                    "similarity": sim,
                }
            )

        return results

    def average_pairwise_similarity(self, texts: Sequence[TextInput]) -> float:
        """
        Average pairwise similarity across all non-empty text pairs.

        Use this as a redundancy score.
        """
        if texts is None:
            return 0.0

        valid_texts = [self.clean_text(t) for t in texts if self.clean_text(t)]

        if len(valid_texts) < 2:
            return 0.0

        sims = []

        for i, j in combinations(range(len(valid_texts)), 2):
            sims.append(self.text_similarity(valid_texts[i], valid_texts[j]))

        if not sims:
            return 0.0

        return float(np.mean(sims))

    def redundancy_score(self, texts: Sequence[TextInput]) -> float:
        """
        Redundancy among a group of texts.

        Returns:
            High value = texts are semantically similar/redundant.
            Low value = texts are semantically diverse.
        """
        return self.average_pairwise_similarity(texts)

    def diversity_score(self, texts: Sequence[TextInput]) -> float:
        """
        Semantic diversity among a group of texts.

        Returns:
            1 - redundancy_score
        """
        return 1.0 - self.redundancy_score(texts)

    def revision_delta(self, old_text: TextInput, new_text: TextInput) -> float:
        """
        How much a revised text changed from the old text.

        Returns:
            0.0 = no meaningful change
            1.0 = very different
        """
        return 1.0 - self.text_similarity(old_text, new_text)

    def group_features(
        self,
        texts: Sequence[TextInput],
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute compact group-level features for RL state.

        Useful for:
        - agent hypotheses
        - agent feedback
        - agent votes if converted to text
        """
        if texts is None:
            texts = []

        cleaned = [self.clean_text(t) for t in texts]
        nonempty_count = sum(1 for t in cleaned if t)

        pairs = self.pairwise_similarities(cleaned, names=names)

        scores = [_safe_float(p["similarity"]) for p in pairs]

        if scores:
            avg_sim = float(np.mean(scores))
            min_sim = float(np.min(scores))
            max_sim = float(np.max(scores))
            std_sim = float(np.std(scores))
        else:
            avg_sim = 0.0
            min_sim = 0.0
            max_sim = 0.0
            std_sim = 0.0

        most_similar_pair = None
        least_similar_pair = None

        if pairs:
            most_similar_pair = max(pairs, key=lambda p: p["similarity"])
            least_similar_pair = min(pairs, key=lambda p: p["similarity"])

        return {
            "num_texts": len(cleaned),
            "num_nonempty_texts": nonempty_count,
            "completion_ratio": nonempty_count / len(cleaned) if cleaned else 0.0,
            "avg_pairwise_similarity": avg_sim,
            "min_pairwise_similarity": min_sim,
            "max_pairwise_similarity": max_sim,
            "std_pairwise_similarity": std_sim,
            "redundancy_score": avg_sim,
            "diversity_score": 1.0 - avg_sim,
            "most_similar_pair": most_similar_pair,
            "least_similar_pair": least_similar_pair,
        }

    def clear_cache(self) -> None:
        """
        Clear stored embeddings.
        """
        self._cache.clear()


_SCORERS: Dict[Tuple[str, Optional[str]], SimilarityScorer] = {}


def get_scorer(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> SimilarityScorer:
    """
    Get a cached SimilarityScorer instance.

    Use this to avoid reloading the embedding model repeatedly.
    """
    key = (model_name, device)

    if key not in _SCORERS:
        _SCORERS[key] = SimilarityScorer(model_name=model_name, device=device)

    return _SCORERS[key]


def text_similarity(
    text_a: TextInput,
    text_b: TextInput,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function for comparing two texts.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.text_similarity(text_a, text_b)


def average_pairwise_similarity(
    texts: Sequence[TextInput],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function for average pairwise similarity.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.average_pairwise_similarity(texts)


def redundancy_score(
    texts: Sequence[TextInput],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function for group redundancy.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.redundancy_score(texts)


def diversity_score(
    texts: Sequence[TextInput],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function for group diversity.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.diversity_score(texts)


def revision_delta(
    old_text: TextInput,
    new_text: TextInput,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function for measuring how much a text changed after revision.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.revision_delta(old_text, new_text)


def similarity_matrix(
    texts: Sequence[TextInput],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Convenience function for a pairwise similarity matrix.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.similarity_matrix(texts)


def group_features(
    texts: Sequence[TextInput],
    names: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for compact RL-state features from a group of texts.
    """
    scorer = get_scorer(model_name=model_name, device=device)
    return scorer.group_features(texts=texts, names=names)


if __name__ == "__main__":
    # Quick smoke test.
    a = "I eat rice."
    b = "Rice is eaten by me."
    c = "The sky is blue."

    print("Similarity A-B:", text_similarity(a, b))
    print("Similarity A-C:", text_similarity(a, c))

    texts = [a, b, c]
    print("Redundancy:", redundancy_score(texts))
    print("Diversity:", diversity_score(texts))
    print("Group features:", group_features(texts, names=["a", "b", "c"]))
