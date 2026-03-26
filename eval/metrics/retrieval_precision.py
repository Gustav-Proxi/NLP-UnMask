"""
Retrieval Precision — Hit Rate @ k.

For each question, check if any of the top-k retrieved chunks
belongs to the correct concept (is_answer_chunk=True AND concept matches).
Also measures MRR (Mean Reciprocal Rank).

Target: Hit Rate @ k=5 ≥ 0.75
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()


def retrieve_for_eval(question: str, concept: str, top_k: int = 5, pcr_override: str = "full_reveal") -> dict:
    """
    Retrieves chunks for a question, temporarily overriding PCR mode
    so we can measure raw retrieval quality independent of mastery gating.

    Returns: {
        "hit": bool,          # relevant chunk in top-k
        "rank": int | None,   # rank of first relevant chunk (1-indexed)
        "retrieved": list     # full list of chunks
    }
    """
    import yaml
    from qdrant_client import QdrantClient
    from google import genai as google_genai

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    gclient = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    r = gclient.models.embed_content(model=cfg["embedding"]["gemini_model"], contents=question)
    query_vec = r.embeddings[0].values

    qdrant = QdrantClient(path="./qdrant_data")
    collection = os.getenv("QDRANT_COLLECTION", cfg["qdrant"]["collection"])

    results = qdrant.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )
    chunks = [hit.payload for hit in results.points]

    hit = False
    rank = None
    for i, chunk in enumerate(chunks, 1):
        if chunk.get("concept") == concept and chunk.get("is_answer_chunk"):
            hit = True
            rank = i
            break

    return {"hit": hit, "rank": rank, "retrieved": chunks}


def compute_retrieval_metrics(results: list[dict]) -> dict:
    """
    results: list of {"hit": bool, "rank": int|None}
    Returns: {"hit_rate": float, "mrr": float}
    """
    n = len(results)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0}

    hits = sum(1 for r in results if r["hit"])
    mrr = sum(1.0 / r["rank"] for r in results if r["rank"] is not None) / n

    return {
        "hit_rate": round(hits / n, 4),
        "mrr": round(mrr, 4),
        "hits": hits,
        "total": n,
    }
