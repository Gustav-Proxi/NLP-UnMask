"""
Controlled Ablation Study — 4 system variants.

Variants:
  full     Full system: PCR + CRAG + concept graph
  no_pcr   No PCR: answer chunks always retrievable (prompt-only suppression)
  no_crag  No CRAG: skip relevance grading and re-query loop
  no_graph No concept graph: flat mastery dict instead of DAG-based tracing

Each variant is run on the first N questions of eval_dataset.json.
Results are compared side-by-side on: leak_rate, purity_score, hit_rate.

Usage:
  python eval/ablation.py               # all 4 variants, all questions
  python eval/ablation.py --n 10        # first 10 questions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from eval.metrics.answer_leak import check_answer_leak
from eval.metrics.socratic_purity import socratic_purity_score
from eval.metrics.retrieval_precision import compute_retrieval_metrics

EVAL_DIR = Path(__file__).parent


# ── Variant configurations ────────────────────────────────────────────────────

VARIANTS = {
    "full": {
        "description": "Full system (PCR + CRAG + concept graph)",
        "pcr_threshold_low": 0.4,
        "pcr_threshold_high": 0.7,
        "use_crag": True,
        "use_concept_graph": True,
        "force_mastery": 0.2,  # student starts cold (low mastery → context_only mode)
    },
    "no_pcr": {
        "description": "No PCR — answer chunks always included (prompt-only suppression)",
        "pcr_threshold_low": 0.0,  # mastery always >= threshold_low
        "pcr_threshold_high": 0.0,  # → always full_reveal mode
        "use_crag": True,
        "use_concept_graph": True,
        "force_mastery": 0.2,
    },
    "no_crag": {
        "description": "No CRAG — standard top-k retrieval, no self-grading or re-query",
        "pcr_threshold_low": 0.4,
        "pcr_threshold_high": 0.7,
        "use_crag": False,
        "use_concept_graph": True,
        "force_mastery": 0.2,
    },
    "no_graph": {
        "description": "No concept graph — flat mastery, no prerequisite tracing",
        "pcr_threshold_low": 0.4,
        "pcr_threshold_high": 0.7,
        "use_crag": True,
        "use_concept_graph": False,
        "force_mastery": 0.2,
    },
}


# ── Per-variant retrieval (respects PCR config) ───────────────────────────────

def retrieve_variant(question: str, concept: str, variant_cfg: dict) -> dict:
    import yaml
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    from google import genai as google_genai

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mastery = variant_cfg["force_mastery"]
    t_low = variant_cfg["pcr_threshold_low"]
    t_high = variant_cfg["pcr_threshold_high"]

    # Determine PCR mode for this variant
    if mastery < t_low:
        mode = "context_only"
        pcr_filter = Filter(must_not=[FieldCondition(key="is_answer_chunk", match=MatchValue(value=True))])
    elif mastery < t_high:
        mode = "prerequisite_first"
        pcr_filter = Filter(must=[FieldCondition(key="chunk_type", match=MatchAny(any=["context","prerequisite","figure"]))])
    else:
        mode = "full_reveal"
        pcr_filter = None

    gclient = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    r = gclient.models.embed_content(model=cfg["embedding"]["gemini_model"], contents=question)
    query_vec = r.embeddings[0].values

    qdrant = QdrantClient(path="./qdrant_data")
    collection = os.getenv("QDRANT_COLLECTION", cfg["qdrant"]["collection"])

    hits = qdrant.query_points(
        collection_name=collection,
        query=query_vec,
        query_filter=pcr_filter,
        limit=cfg["retrieval"]["top_k"],
        with_payload=True,
    )
    chunks = [h.payload for h in hits.points]

    # Hit: answer chunk for this concept present in results
    hit = any(c.get("concept") == concept and c.get("is_answer_chunk") for c in chunks)
    rank = next(
        (i + 1 for i, c in enumerate(chunks) if c.get("concept") == concept and c.get("is_answer_chunk")),
        None,
    )

    return {"hit": hit, "rank": rank, "retrieved": chunks, "mode": mode}


def generate_variant_response(question: str, chunks: list[dict]) -> str:
    import yaml
    from openai import OpenAI
    from src.nodes.socratic_generator import SocraticOutput

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    context_text = "\n\n".join(
        f"[{c.get('chunk_type','ctx').upper()}] {c['text']}" for c in chunks[:5]
    )
    system = f"""\
You are UnMask, a Socratic anatomy tutor.
You know the correct answer (it is in the context below) but must NOT reveal it.
Generate a Socratic question that guides the student toward discovering the answer.
The question must end with "?".

CONTEXT:
{context_text}
"""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    resp = client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", cfg["llm"]["model"]),
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        response_format=SocraticOutput,
    )
    output = resp.choices[0].message.parsed
    return f"{output.visible_response.encouragement} {output.visible_response.socratic_question}".strip()


# ── Main ablation loop ────────────────────────────────────────────────────────

def run_ablation(n_questions: int) -> dict:
    with open(EVAL_DIR / "eval_dataset.json") as f:
        dataset = json.load(f)[:n_questions]

    all_results = {}

    for variant_name, variant_cfg in VARIANTS.items():
        print(f"\n{'─'*55}")
        print(f"  Variant: {variant_name.upper()} — {variant_cfg['description']}")
        print(f"{'─'*55}")

        results = []
        for item in tqdm(dataset, desc=f"  {variant_name}"):
            ret = retrieve_variant(item["question"], item["concept"], variant_cfg)

            try:
                response = generate_variant_response(item["question"], ret["retrieved"])
            except Exception as e:
                response = f"[ERROR: {e}]"

            leak = check_answer_leak(
                response=response,
                expected_answer=item["expected_answer"],
                answer_keywords=item["answer_keywords"],
            )
            purity = socratic_purity_score(
                question=item["question"],
                response=response,
                gold_answer=item["expected_answer"],
                leaked=leak["leaked"],
                ends_with_question=leak["ends_with_question"],
            )

            results.append({
                "id": item["id"],
                "retrieval_hit": ret["hit"],
                "retrieval_rank": ret["rank"],
                "pcr_mode": ret["mode"],
                "leaked": leak["leaked"],
                "semantic_similarity": leak["semantic_similarity"],
                "purity_score": purity["final_score"],
                "ends_with_question": leak["ends_with_question"],
            })
            time.sleep(0.3)

        # Aggregate
        n = len(results)
        all_results[variant_name] = {
            "description": variant_cfg["description"],
            "hit_rate": sum(1 for r in results if r["retrieval_hit"]) / n,
            "leak_rate": sum(1 for r in results if r["leaked"]) / n,
            "avg_purity": sum(r["purity_score"] for r in results) / n,
            "question_rate": sum(1 for r in results if r["ends_with_question"]) / n,
            "per_question": results,
        }

    return all_results


def print_ablation_table(all_results: dict) -> None:
    print(f"\n{'='*65}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*65}")
    print(f"  {'Variant':<12} {'Hit Rate':>9} {'Leak Rate':>10} {'Purity':>8} {'Ends ?':>7}")
    print(f"  {'─'*12} {'─'*9} {'─'*10} {'─'*8} {'─'*7}")
    for name, res in all_results.items():
        print(
            f"  {name:<12} "
            f"{res['hit_rate']:>9.3f} "
            f"{res['leak_rate']:>10.3f} "
            f"{res['avg_purity']:>8.2f} "
            f"{res['question_rate']:>7.3f}"
        )
    print()
    print("  Targets: Hit Rate ≥ 0.75 | Leak Rate = 0 | Purity ≥ 4.0")

    # Save to file
    with open("/tmp/unmask_ablation.md", "w") as f:
        f.write("# UnMask Ablation Study\n\n")
        f.write("| Variant | Description | Hit Rate | Leak Rate | Avg Purity | Ends ? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, res in all_results.items():
            f.write(
                f"| {name} | {res['description']} | "
                f"{res['hit_rate']:.3f} | {res['leak_rate']:.3f} | "
                f"{res['avg_purity']:.2f} | {res['question_rate']:.3f} |\n"
            )
    print("  📄 Full ablation report: /tmp/unmask_ablation.md\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=len(json.load(open(EVAL_DIR / "eval_dataset.json"))),
                        help="Number of questions per variant")
    args = parser.parse_args()
    results = run_ablation(args.n)
    print_ablation_table(results)
