"""
RAGAS evaluation — Faithfulness + Answer Relevancy.

Faithfulness: are claims in the response grounded in the retrieved context?
Answer Relevancy: does the response address the question? (adapted for Socratic: question should be on-topic)

Targets: Faithfulness ≥ 0.85, Answer Relevancy ≥ 0.80
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def run_ragas(
    questions: list[str],
    responses: list[str],
    contexts: list[list[str]],   # per-question list of retrieved chunk texts
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS faithfulness + answer relevancy.
    Returns {"faithfulness": float, "answer_relevancy": float, "raw": dict}
    """
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.environ["OPENAI_API_KEY"]

    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", cfg["llm"]["model"]),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=cfg["embedding"]["openai_model"],
        api_key=api_key,
        base_url=base_url,
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": responses,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    scores = result.to_pandas()
    faithfulness = float(scores["faithfulness"].mean())
    relevancy = float(scores["answer_relevancy"].mean())

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "faithfulness_passed": faithfulness >= 0.85,
        "relevancy_passed": relevancy >= 0.80,
    }
