# DEVNOTES.md

Developer notes and architecture reference for the UnMask project.

---

## Project Identity

**Socratic-OT** — CSE 635: NLP and Text Mining, Spring 2026, University at Buffalo.
A multimodal AI tutor for Occupational Therapy (OT) students preparing for the NBCOT exam. Core constraint: the system **never gives direct answers**; it guides via Socratic questions while holding the correct answer in a hidden `internal_analysis` field.

The Word document `UnMask v4 Update-1.docx` contains the current design spec. Full Phase 1 proposal is in `../Phase1_Complete_Submission.md`.

---

## Architecture

### System Overview

```
Student Input (Chainlit UI)
        │
        ▼
  LangGraph State Machine
  ├── Manager Agent       (pure Python, zero LLM calls, <1ms routing)
  ├── Rapport Agent       [0–2 min]
  ├── RAG Retriever       [on demand]
  ├── Socratic Agent      [2–12 min]  ← core loop
  ├── Hint Agent          [consecutive_incorrect ≥ 2]
  ├── Reveal Agent        [hint_level ≥ max_hints]
  ├── Assessment Agent    [consecutive_correct ≥ 2 OR 12 min]
  ├── Image Analyzer      [on upload]
  └── Wrapup Agent        [session end]
        │
  LLM Router
  ├── Tier 1: Llama 3.1 8B via Ollama    (65–75% of turns, $0)
  └── Tier 2: GPT-4o via OpenRouter      (25–35%, ~$0.01/call)
              └── Fallback: Claude → Gemini (auto)
```

### Knowledge Masking (Core Innovation)

The structured output schema enforces answer-withholding architecturally — not via prompt instruction:

```python
response_schema = {
    "internal_analysis": {           # Stripped by app layer before display
        "correct_answer": str,
        "student_misconception": str,
        "planned_hint_sequence": list[str],
        "relevant_textbook_section": str
    },
    "visible_response": {            # Only this is rendered in Chainlit
        "socratic_question": str,
        "encouragement": str
    }
}
```

The model *computes* the correct answer (enabling a well-aimed question) but the output schema provides no field to reveal it. This is structurally stronger than "don't give the answer" prompt instructions.

### Manager Agent State Transitions

Pure-Python, deterministic — zero LLM calls:

```
Rapport → Topic Selection :  turn_count ≥ 3  OR  student_signals_readiness()
Topic   → Socratic        :  RAG retrieval complete
Socratic → Hints          :  consecutive_incorrect ≥ 2
Hints    → Reveal         :  hint_level ≥ max_hints
Any     → Assessment      :  consecutive_correct ≥ 2
Any     → Wrapup          :  elapsed_time ≥ 12 min
Any     → Image Analysis  :  image_uploaded()
```

### RAG Pipeline

- **Index:** OpenStax Anatomy & Physiology 2e (text + labeled diagrams in unified Qdrant collection)
- **Retrieval:** Hybrid — Gemini Embedding 2 (dense) + BM25 (sparse), merged via RRF, top-5 results
- **Image optimization:** Figures annotated once at index time; JSON cached as Qdrant payload to eliminate ~80% of VLM API calls at query time

### BKT (Bayesian Knowledge Tracing)

Each topic node (e.g., `brachial_plexus.upper_trunk`) maintains `SimpleBKT(P_L0=0.3, P_T=0.1, P_G=0.2, P_S=0.1)`. Topics with `P(mastery) < 0.6` are flagged for weak-topic revisit after 8 elapsed minutes. State persisted via `LangGraph MemorySaver` — no external DB.

---

## Stack

| Component | Choice |
|-----------|--------|
| UI | Chainlit |
| Orchestration | LangGraph (`MemorySaver` checkpointer) |
| Vector DB | Qdrant (local persistent) |
| Embedding | Gemini Embedding 2 |
| Local LLM | Llama 3.1 8B via Ollama |
| API LLM | GPT-4o via OpenRouter (Claude/Gemini fallback) |
| Multimodal | GPT-4o (VLM) or MedGemma 4B (local) |
| Python | 3.11+ |
| Tests | pytest |

---

## Setup & Commands

```bash
# 1. Install
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY

# 2. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 3. Index anatomy knowledge base
python scripts/index_kb.py                        # uses chunks.json
python scripts/index_kb.py --recreate             # drop and rebuild
python scripts/index_kb.py --collection physics   # physics subject-swap demo

# 4. Run the tutor
chainlit run app.py

# 5. Tests (once written)
pytest tests/
pytest tests/test_pcr.py::test_context_only_excludes_answer_chunks -v
```

### Key env vars
```
OPENAI_API_KEY        required
OPENAI_BASE_URL       optional — set to https://openrouter.ai/api/v1 for OpenRouter
OPENAI_MODEL          optional — default gpt-4o
GOOGLE_API_KEY        optional — for Gemini Embedding 2
EMBEDDING_PROVIDER    openai (default) | gemini
QDRANT_COLLECTION     unmask_anatomy (default)
```

---

## Evaluation Targets

| Metric | Target | Tool |
|--------|--------|------|
| RAGAS Faithfulness | ≥ 0.85 | `ragas` library |
| Socratic Purity (no leaks) | ≥ 0.90 | Custom LLM-as-judge |
| Blind diagram test | ≥ 70% correct | Manual + GPT-4o rubric |
| Cross-domain (Physics) | Graceful degradation | 10 OpenStax Physics QA pairs |

---

## Key Design Decisions

- **Manager Agent = pure Python** (not LLM-based) — DiagGPT (2023) showed rule-based state controllers outperform LLM routers for deterministic transitions.
- **Structured output for masking** — Phung et al. (2023) two-model architecture validates the `internal_analysis` / `visible_response` split.
- **Unified Qdrant collection** for text + images — single hybrid search retrieves both; avoids two-pass retrieval overhead.
- **OpenRouter fallback chain** (GPT-4o → Claude → Gemini) — ensures demo resilience during live in-class evaluation.
- **`consecutive_correct ≥ 2` threshold** controls Socratic → Assessment transition to prevent premature exit from tutoring loop.
