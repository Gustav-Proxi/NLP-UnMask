"""
Answer Leak Detection.

Two-layer check (no medical knowledge required):
  Layer 1 — Keyword match: does the response contain answer keywords verbatim?
  Layer 2 — Semantic similarity: is cosine similarity(response, gold_answer) > threshold?

A "leak" = either layer fires.
Target: 0% leak rate on turns 1-2.
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()


# ── Layer 1: keyword match ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def keyword_leak(response: str, answer_keywords: list[str]) -> tuple[bool, list[str]]:
    """
    Returns (leaked, matched_keywords).
    Fired when ≥3 distinct answer keywords appear in the response.
    1-2 keyword matches are normal (Socratic questions reference the topic).
    """
    norm_response = _normalize(response)
    matched = [kw for kw in answer_keywords if _normalize(kw) in norm_response]
    # Require at least 3 keywords — a Socratic question might mention 1-2 terms
    # from the answer space without actually revealing the answer
    leaked = len(matched) >= 3
    return leaked, matched


# ── Layer 2: semantic similarity ──────────────────────────────────────────────

def _embed_for_similarity(text: str) -> list[float]:
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    provider = os.getenv("EMBEDDING_PROVIDER", cfg["embedding"]["provider"])
    if provider == "gemini":
        from google import genai as google_genai
        gclient = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        r = gclient.models.embed_content(model=cfg["embedding"]["gemini_model"], contents=text)
        return r.embeddings[0].values
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
        resp = client.embeddings.create(model=cfg["embedding"]["openai_model"], input=text)
        return resp.data[0].embedding


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def semantic_leak(
    response: str,
    expected_answer: str,
    threshold: float = 0.92,
) -> tuple[bool, float]:
    """
    Returns (leaked, similarity_score).
    High cosine similarity means the response is semantically close to the gold answer.
    Threshold is deliberately high (0.92) because Socratic questions about the same
    topic will have ~0.85-0.90 similarity without actually leaking the answer.
    """
    vec_response = _embed_for_similarity(response)
    vec_answer = _embed_for_similarity(expected_answer)
    sim = _cosine(vec_response, vec_answer)
    return sim >= threshold, round(sim, 4)


# ── Combined check ────────────────────────────────────────────────────────────

def check_answer_leak(
    response: str,
    expected_answer: str,
    answer_keywords: list[str],
    semantic_threshold: float = 0.92,
) -> dict:
    kw_leaked, matched_kws = keyword_leak(response, answer_keywords)
    sem_leaked, similarity = semantic_leak(response, expected_answer, semantic_threshold)

    # Confirmed leak = BOTH layers agree.
    # A Socratic question naturally mentions topic names (triggers keyword)
    # but stays well below semantic similarity to the full answer string.
    confirmed_leak = kw_leaked and sem_leaked
    soft_flag = kw_leaked or sem_leaked

    return {
        "leaked": confirmed_leak,
        "soft_flag": soft_flag,
        "keyword_leaked": kw_leaked,
        "semantic_leaked": sem_leaked,
        "matched_keywords": matched_kws,
        "semantic_similarity": similarity,
        "ends_with_question": response.strip().endswith("?"),
    }
