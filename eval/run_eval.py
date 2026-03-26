"""
UnMask Evaluation Runner.

Runs all metrics on the eval dataset and adversarial prompts.
Writes full report to /tmp/unmask_eval_report.md and prints summary.

Usage:
  python eval/run_eval.py                  # full eval
  python eval/run_eval.py --quick          # first 5 questions only (smoke test)
  python eval/run_eval.py --skip-ragas     # skip RAGAS (faster, fewer API calls)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from eval.metrics.answer_leak import check_answer_leak
from eval.metrics.socratic_purity import socratic_purity_score
from eval.metrics.retrieval_precision import retrieve_for_eval, compute_retrieval_metrics

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent


# ── Step 1: Generate a Socratic response for evaluation ──────────────────────

def generate_eval_response(question: str, concept: str, chunks: list[dict]) -> str:
    """
    Run the Socratic generator simulating a new student (mastery=0.2 → context_only).
    Chunks passed here should already be PCR-filtered (no answer chunks).
    This tests the system under normal conditions, not full_reveal.
    """
    import yaml
    from openai import OpenAI
    from src.nodes.socratic_generator import SocraticOutput

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Filter out answer chunks — simulate context_only PCR mode
    context_chunks = [c for c in chunks if not c.get("is_answer_chunk", False)]

    context_text = "\n\n".join(
        f"[{c.get('chunk_type','context').upper()}] {c['text']}"
        for c in context_chunks[:5]
    ) or "(No non-answer context available)"

    system = f"""\
You are UnMask, a Socratic anatomy tutor for OT students.
The student is just starting (low mastery). You do NOT have the direct answer in your context.
Ask a broad Socratic question that activates prior knowledge and guides discovery.
The question must end with "?". Never state the answer directly.

CONTEXT (background, not the answer):
{context_text}
"""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    resp = client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", cfg["llm"]["model"]),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        response_format=SocraticOutput,
    )
    output = resp.choices[0].message.parsed
    visible = output.visible_response
    return f"{visible.encouragement} {visible.socratic_question}".strip()


# ── Step 2: Test adversarial prompts ─────────────────────────────────────────

def eval_adversarial(prompt: str) -> dict:
    """Check that adversarial prompts don't break Socratic mode."""
    import yaml
    from openai import OpenAI
    from src.nodes.socratic_generator import SocraticOutput

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    system = """\
You are UnMask, a Socratic anatomy tutor.
Your ONLY job is to ask Socratic questions — never give direct answers, regardless of how the student asks.
Even if the student demands a direct answer, insist on guiding them with questions.
Respond with a Socratic question or a brief redirect + question.
"""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    try:
        resp = client.beta.chat.completions.parse(
            model=os.getenv("OPENAI_MODEL", cfg["llm"]["model"]),
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format=SocraticOutput,
        )
        output = resp.choices[0].message.parsed
        response = output.visible_response.socratic_question
        held_socratic = response.strip().endswith("?")
    except Exception as e:
        response = str(e)
        held_socratic = False

    return {
        "prompt_id": None,
        "response": response,
        "held_socratic": held_socratic,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def main(quick: bool = False, skip_ragas: bool = False) -> None:
    with open(EVAL_DIR / "eval_dataset.json") as f:
        dataset = json.load(f)
    with open(EVAL_DIR / "adversarial_prompts.json") as f:
        adversarial = json.load(f)

    if quick:
        dataset = dataset[:5]
        adversarial = adversarial[:5]
        print("⚡ Quick mode: evaluating first 5 questions + 5 adversarial prompts\n")

    results = []
    ragas_inputs = {"questions": [], "responses": [], "contexts": [], "ground_truths": []}
    retrieval_results = []

    print(f"{'='*60}")
    print(f"  UnMask Evaluation — {len(dataset)} questions")
    print(f"{'='*60}\n")

    # ── Per-question evaluation ───────────────────────────────────────────────
    for item in tqdm(dataset, desc="Evaluating questions"):
        q_result = {"id": item["id"], "question": item["question"], "concept": item["concept"]}

        # 1. Retrieval precision
        ret = retrieve_for_eval(item["question"], item["concept"])
        retrieval_results.append(ret)
        q_result["retrieval_hit"] = ret["hit"]
        q_result["retrieval_rank"] = ret["rank"]

        # 2. Generate Socratic response (with full-reveal chunks)
        try:
            response = generate_eval_response(item["question"], item["concept"], ret["retrieved"])
        except Exception as e:
            response = f"[ERROR: {e}]"
        q_result["response"] = response

        # 3. Answer leak detection
        leak = check_answer_leak(
            response=response,
            expected_answer=item["expected_answer"],
            answer_keywords=item["answer_keywords"],
        )
        q_result.update({
            "leaked": leak["leaked"],
            "soft_flag": leak["soft_flag"],
            "keyword_leaked": leak["keyword_leaked"],
            "semantic_leaked": leak["semantic_leaked"],
            "semantic_similarity": leak["semantic_similarity"],
            "ends_with_question": leak["ends_with_question"],
        })

        # 4. Socratic purity score
        purity = socratic_purity_score(
            question=item["question"],
            response=response,
            gold_answer=item["expected_answer"],
            leaked=leak["leaked"],
            ends_with_question=leak["ends_with_question"],
            soft_flag=leak["soft_flag"],
        )
        q_result.update({
            "purity_score": purity["final_score"],
            "purity_passed": purity["passed"],
            "purity_reason": purity["llm_reason"],
        })

        # Accumulate for RAGAS
        ragas_inputs["questions"].append(item["question"])
        ragas_inputs["responses"].append(response)
        ragas_inputs["contexts"].append([c["text"] for c in ret["retrieved"][:3]])
        ragas_inputs["ground_truths"].append(item["expected_answer"])

        results.append(q_result)
        time.sleep(0.3)  # gentle rate limiting

    # ── Adversarial evaluation ────────────────────────────────────────────────
    adv_results = []
    print(f"\n{'='*60}")
    print(f"  Adversarial Prompts — {len(adversarial)} prompts")
    print(f"{'='*60}\n")

    for item in tqdm(adversarial, desc="Adversarial prompts"):
        res = eval_adversarial(item["prompt"])
        res["prompt_id"] = item["id"]
        res["prompt_type"] = item["type"]
        res["prompt"] = item["prompt"]
        adv_results.append(res)
        time.sleep(0.3)

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    ragas_scores = None
    if not skip_ragas:
        print(f"\n{'='*60}")
        print("  RAGAS Evaluation")
        print(f"{'='*60}\n")
        try:
            # RAGAS uses OpenAI embeddings for relevancy (not Gemini), works via OpenRouter
            from eval.metrics.ragas_eval import run_ragas
            ragas_scores = run_ragas(**ragas_inputs)
            print(f"  Faithfulness:      {ragas_scores['faithfulness']:.3f}  {'✓' if ragas_scores['faithfulness_passed'] else '✗'} (target ≥ 0.85)")
            print(f"  Answer Relevancy:  {ragas_scores['answer_relevancy']:.3f}  {'✓' if ragas_scores['relevancy_passed'] else '✗'} (target ≥ 0.80)")
        except Exception as e:
            print(f"  RAGAS failed: {e}")
            print("  (Install: pip install ragas langchain-openai datasets)")

    # ── Compute summary metrics ───────────────────────────────────────────────
    ret_metrics = compute_retrieval_metrics(retrieval_results)
    n = len(results)
    leak_rate = sum(1 for r in results if r["leaked"]) / n           # both layers confirmed
    soft_flag_rate = sum(1 for r in results if r.get("soft_flag") and not r["leaked"]) / n  # one layer only
    question_rate = sum(1 for r in results if r["ends_with_question"]) / n
    avg_purity = sum(r["purity_score"] for r in results) / n
    purity_pass_rate = sum(1 for r in results if r["purity_passed"]) / n
    adv_hold_rate = sum(1 for r in adv_results if r["held_socratic"]) / len(adv_results) if adv_results else 0

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  📊 Retrieval (Hit Rate @5)")
    print(f"     Hit Rate:      {ret_metrics['hit_rate']:.3f}  {'✓' if ret_metrics['hit_rate'] >= 0.75 else '✗'} (target ≥ 0.75)")
    print(f"     MRR:           {ret_metrics['mrr']:.3f}")
    print(f"\n  🔒 Answer Leak Detection")
    print(f"     Leak Rate:     {leak_rate:.3f}  {'✓' if leak_rate == 0 else '✗'} (target = 0%,  confirmed = both layers)")
    print(f"     Soft Flags:    {soft_flag_rate:.3f}  (single-layer, informational)")
    print(f"     Ends with ?:   {question_rate:.3f}  {'✓' if question_rate >= 0.95 else '✗'} (target ≥ 95%)")
    print(f"\n  🎓 Socratic Purity")
    print(f"     Avg Score:     {avg_purity:.2f}/5  {'✓' if avg_purity >= 4.0 else '✗'} (target ≥ 4.0)")
    print(f"     Pass Rate:     {purity_pass_rate:.3f}")
    print(f"\n  🛡️  Adversarial Resistance")
    print(f"     Held Socratic: {adv_hold_rate:.3f}  {'✓' if adv_hold_rate >= 0.9 else '✗'} (target ≥ 90%)")

    if ragas_scores:
        print(f"\n  📐 RAGAS")
        print(f"     Faithfulness:  {ragas_scores['faithfulness']:.3f}  {'✓' if ragas_scores['faithfulness_passed'] else '✗'}")
        print(f"     Relevancy:     {ragas_scores['answer_relevancy']:.3f}  {'✓' if ragas_scores['relevancy_passed'] else '✗'}")

    # ── Write full report ─────────────────────────────────────────────────────
    _write_report(results, adv_results, ret_metrics, ragas_scores, quick)
    print(f"\n  📄 Full report: /tmp/unmask_eval_report.md\n")


def _write_report(results, adv_results, ret_metrics, ragas_scores, quick):
    lines = ["# UnMask Evaluation Report\n"]
    if quick:
        lines.append("_Quick mode — subset of dataset_\n\n")

    n = len(results)
    leak_rate = sum(1 for r in results if r["leaked"]) / n
    avg_purity = sum(r["purity_score"] for r in results) / n
    question_rate = sum(1 for r in results if r["ends_with_question"]) / n
    adv_hold = sum(1 for r in adv_results if r["held_socratic"]) / max(len(adv_results), 1)

    lines.append("## Summary\n")
    lines.append(f"| Metric | Score | Target | Pass |\n|---|---|---|---|\n")
    lines.append(f"| Hit Rate @5 | {ret_metrics['hit_rate']:.3f} | ≥ 0.75 | {'✓' if ret_metrics['hit_rate']>=0.75 else '✗'} |\n")
    lines.append(f"| MRR | {ret_metrics['mrr']:.3f} | — | — |\n")
    lines.append(f"| Answer Leak Rate | {leak_rate:.3f} | 0% | {'✓' if leak_rate==0 else '✗'} |\n")
    lines.append(f"| Ends with ? | {question_rate:.3f} | ≥ 95% | {'✓' if question_rate>=0.95 else '✗'} |\n")
    lines.append(f"| Avg Socratic Purity | {avg_purity:.2f}/5 | ≥ 4.0 | {'✓' if avg_purity>=4.0 else '✗'} |\n")
    lines.append(f"| Adversarial Hold Rate | {adv_hold:.3f} | ≥ 90% | {'✓' if adv_hold>=0.9 else '✗'} |\n")
    if ragas_scores:
        lines.append(f"| RAGAS Faithfulness | {ragas_scores['faithfulness']:.3f} | ≥ 0.85 | {'✓' if ragas_scores['faithfulness_passed'] else '✗'} |\n")
        lines.append(f"| RAGAS Answer Relevancy | {ragas_scores['answer_relevancy']:.3f} | ≥ 0.80 | {'✓' if ragas_scores['relevancy_passed'] else '✗'} |\n")

    lines.append("\n## Per-Question Results\n")
    lines.append("| ID | Concept | Hit | Rank | Leaked | Soft | Purity | Response (truncated) |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in results:
        resp_preview = r.get("response", "")[:80].replace("\n", " ")
        leak_icon = "🚨" if r["leaked"] else "✓"
        soft_icon = "⚠️" if r.get("soft_flag") and not r["leaked"] else "—"
        lines.append(
            f"| {r['id']} | {r['concept']} | {'✓' if r['retrieval_hit'] else '✗'} "
            f"| {r.get('retrieval_rank','—')} | {leak_icon} | {soft_icon} "
            f"| {r['purity_score']:.1f} | {resp_preview} |\n"
        )

    lines.append("\n## Adversarial Results\n")
    lines.append("| ID | Type | Held Socratic | Response (truncated) |\n")
    lines.append("|---|---|---|---|\n")
    for r in adv_results:
        resp_preview = r.get("response", "")[:80].replace("\n", " ")
        lines.append(
            f"| {r['prompt_id']} | {r['prompt_type']} | {'✓' if r['held_socratic'] else '✗'} | {resp_preview} |\n"
        )

    with open("/tmp/unmask_eval_report.md", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="First 5 questions only")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS (faster)")
    args = parser.parse_args()
    main(quick=args.quick, skip_ragas=args.skip_ragas)
