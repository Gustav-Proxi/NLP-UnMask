# UnMask — Socratic OT Anatomy Tutor

**CSE 635: NLP and Text Mining · University at Buffalo · Spring 2026**

A multimodal AI tutor for Occupational Therapy students preparing for the **NBCOT certification exam**. UnMask never gives direct answers — it guides students toward discovering answers themselves through Socratic questioning.

*Built by Sanika Vilas Najan & Vaishak Girish Kumar*

---

## Core Idea

Instead of telling the student the answer, UnMask:

1. Retrieves the correct answer from OpenStax Anatomy & Physiology 2e
2. Hides it in a masked `internal_analysis` field (structurally absent from the student's view)
3. Asks a Socratic question calibrated to guide the student toward discovering it

This is **Progressive Context Revelation (PCR)** — the system's core novelty. Answer chunks in Qdrant are gated by student mastery score:

| Mastery | PCR Mode | What the LLM sees |
|---------|----------|-------------------|
| < 0.4 | `context_only` | Background context only — no answer chunks |
| 0.4–0.7 | `prerequisite_first` | Prerequisite and context chunks |
| > 0.7 | `full_reveal` | All chunks including answer |

Knowledge masking is enforced architecturally via structured output — not just prompt instructions:

```python
class SocraticOutput(BaseModel):
    internal_analysis: InternalAnalysis  # stripped before display
    visible_response: VisibleResponse    # only this is shown to student
```

---

## Architecture

```
Student Input (Chainlit UI)
        │
        ▼
  LangGraph State Machine
  ├── Orchestrator        (pure Python, zero LLM calls)
  ├── Retrieval Planner   (PCR filter + hybrid RAG + CRAG)
  ├── Socratic Generator  (structured output masking)
  └── Pedagogy Agent      (BKT mastery update + concept DAG)
        │
  LLM: GPT-4o via OpenRouter
  Embeddings: Gemini Embedding 2 (3072d)
  Vector DB: Qdrant (local file mode)
```

**Session phases (~15 min):**

| Phase | Duration | What happens |
|-------|----------|--------------|
| Warm-up | 0–2 min | 4 diagnostic questions to calibrate starting mastery |
| Tutoring | 2–12 min | Socratic loop — questions, hints, concept graph tracing |
| Assessment | 12–14 min | Clinical scenario — student explains reasoning |
| Wrap-up | 14–15 min | Mastery summary + weak topics flagged |

**RAG pipeline:**
- Hybrid retrieval: Gemini Embedding 2 (dense) + BM25 (sparse), merged via RRF, top-5 results
- Corrective RAG (CRAG): grades retrieved docs → reformulates query on failure (max 2 retries)
- Bayesian Knowledge Tracing (BKT) per concept node; `P(mastery) < 0.6` flags weak-topic revisit

---

## Topics Covered (MVP)

- Brachial plexus (origin → trunks → cords → terminal branches)
- Peripheral nerves: axillary, radial, median, ulnar
- Rotator cuff muscles (SITS)

---

## Setup

**Requirements:** Python 3.11+, no Docker needed (Qdrant runs in local file mode)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install google-genai  # Gemini Embedding 2

# 2. Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, OPENAI_BASE_URL, GOOGLE_API_KEY
```

**.env values:**
```
OPENAI_API_KEY=<your-openrouter-key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=<your-google-api-key>
QDRANT_COLLECTION=unmask_anatomy
```

```bash
# 3. Index the knowledge base
python scripts/index_kb.py           # initial index
python scripts/index_kb.py --recreate  # drop and rebuild

# 4. Run the tutor
chainlit run app.py
```

---

## Evaluation

```bash
python eval/run_eval.py              # full eval (30 questions + 20 adversarial)
python eval/run_eval.py --quick      # first 5 questions (smoke test)
python eval/run_eval.py --skip-ragas # skip RAGAS for speed
```

**Targets and results:**

| Metric | Target | Result |
|--------|--------|--------|
| Retrieval Hit Rate @5 | ≥ 0.75 | **1.000** ✓ |
| Answer Leak Rate | 0% | **0.000** ✓ |
| Ends with `?` | ≥ 95% | **1.000** ✓ |
| Avg Socratic Purity (1–5) | ≥ 4.0 | **4.93** ✓ |
| Adversarial Resistance | ≥ 90% | **0.950** ✓ |

Leak detection uses a dual-layer approach (keyword match + semantic similarity >0.92 must both fire) to eliminate false positives from Socratic questions that naturally reference topic vocabulary.

---

## Project Structure

```
app.py                      # Chainlit entry point
config.yaml                 # All tunable parameters
src/
  graph.py                  # LangGraph state machine
  state.py                  # TutoringState TypedDict
  nodes/
    orchestrator.py         # Phase transition logic (pure Python)
    retrieval_planner.py    # PCR filter + hybrid RAG
    socratic_generator.py   # Structured output masking
    pedagogy_agent.py       # BKT + concept DAG
  knowledge_base/
    chunks.json             # 25 anatomy chunks with PCR metadata
    concept_graph.json      # 16-node prerequisite DAG
scripts/
  index_kb.py               # Index chunks.json into Qdrant
eval/
  eval_dataset.json         # 30 QA triples
  adversarial_prompts.json  # 20 adversarial prompts (5 types)
  run_eval.py               # Main evaluation runner
  ablation.py               # 4-variant ablation study
  metrics/
    answer_leak.py          # Dual-layer leak detection
    socratic_purity.py      # LLM-as-judge purity score
    retrieval_precision.py  # Hit rate + MRR
    ragas_eval.py           # RAGAS faithfulness + relevancy
```
