"""
Index the knowledge base into Qdrant.

Usage:
  python scripts/index_kb.py                      # index default chunks.json
  python scripts/index_kb.py --collection physics  # different subject

Every chunk gets:
  - Dense vector embedding (OpenAI or Gemini)
  - Metadata fields: topic, concept, is_answer_chunk, chunk_type
    (PCR filters on is_answer_chunk and chunk_type at query time)
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    CollectionConfig,
)
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).parent.parent
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    provider = os.getenv("EMBEDDING_PROVIDER", cfg["embedding"]["provider"])
    if provider == "gemini":
        from google import genai as google_genai
        gclient = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        results = []
        for text in texts:
            r = gclient.models.embed_content(
                model=cfg["embedding"]["gemini_model"],
                contents=text,
            )
            results.append(r.embeddings[0].values)
        return results
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.embeddings.create(
            model=cfg["embedding"]["openai_model"],
            input=texts,
        )
        return [d.embedding for d in resp.data]


# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_dimension() -> int:
    return cfg["embedding"]["dimension"]  # 3072 for gemini-embedding-2-preview


def main(collection: str, chunks_path: Path, recreate: bool) -> None:
    client = QdrantClient(path="./qdrant_data")

    dim = get_dimension()

    existing = [c.name for c in client.get_collections().collections]
    if recreate and collection in existing:
        client.delete_collection(collection)
        existing.remove(collection)

    if collection not in existing:
        print(f"Creating collection '{collection}' (dim={dim})")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Indexing {len(chunks)} chunks...")
    batch_size = 16
    points = []

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed_batch(texts)

        for chunk, vec in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),
                    vector=vec,
                    payload={
                        **chunk,
                        # Ensure these fields are always present for PCR filters
                        "is_answer_chunk": chunk.get("is_answer_chunk", False),
                        "chunk_type": chunk.get("chunk_type", "context"),
                    },
                )
            )

    client.upsert(collection_name=collection, points=points)
    print(f"✓ Indexed {len(points)} chunks into '{collection}'")

    # Sanity check: count answer vs context chunks
    answer_ct = sum(1 for c in chunks if c.get("is_answer_chunk"))
    context_ct = len(chunks) - answer_ct
    print(f"  {answer_ct} answer chunks (PCR-gated)  |  {context_ct} context/prereq chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", cfg["qdrant"]["collection"]),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--chunks",
        default=str(ROOT / "src" / "knowledge_base" / "chunks.json"),
        help="Path to chunks JSON file",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection",
    )
    args = parser.parse_args()
    main(args.collection, Path(args.chunks), args.recreate)
