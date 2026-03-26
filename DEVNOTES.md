# DEVNOTES.md

Developer notes and architecture reference for the UnMask project.

---

## Project Identity

**UnMask** — CSE 635: NLP and Text Mining, Spring 2026, University at Buffalo.
Authors: Sanika Vilas Najan (`snajan@buffalo.edu`) · Vaishak Girish Kumar (`vaishakg@buffalo.edu`)

A Socratic AI tutor for Occupational Therapy (OT) students preparing for the NBCOT exam.
Core constraint: the system **never gives direct answers** — it guides via Socratic questions while holding the correct answer in a hidden `internal_analysis` field.

---

## Architecture (as implemented)

```
Student Input (Chainlit UI)
        │
        ▼
  LangGraph State Machine (5 nodes, MemorySaver checkpointer)
  ├── orchestrator         pure Python, zero LLM calls — phase transitions + revisit trigger
  ├── retrieval_planner    PCR filter + hybrid RAG (dense+BM25+RRF) + CRAG loop
  ├── socratic_generator   structured output masking (GPT-4o / Ollama)
  └── pedagogy_agent       mastery update + concept DAG + mistake log
        │
  LLM Routing:
    Llama 3.1 8B (Ollama)  — rapport + wrapup (65–75% of turns, $0)
    GPT-4o (OpenRouter)    — tutoring + assessment (~$0.08–0.10/session)
  Vector DB: Qdrant (local file mode)
  Embeddings: Gemini Embedding 2 (3072d) + BM25 sparse, merged by RRF (k=60)
```

### Session Phases

| Phase | Window | Entry | Exit |
|-------|--------|-------|------|
| Rapport | 0–120s | start | 4 diagnostic Qs complete |
| Tutoring | 120–720s | diagnostic_complete | coverage ≥ 0.80 or t ≥ 720s |
| Assessment | 720–840s | coverage/time trigger | t ≥ 840s |
| Wrapup | 840–900s | t ≥ 840s | session end |

Proactive revisit fires at **t ≥ 480s** (8 min) within Tutoring if weak topics exist.

---

## Core Mechanisms

### 1. Progressive Context Revelation (PCR)

Every Qdrant chunk carries `is_answer_chunk: bool` and `chunk_type`. The Retrieval Planner reads mastery and applies a server-side filter:

```python
if mastery < 0.40:   # context_only  → must_not(is_answer_chunk=True)
elif mastery < 0.70: # prerequisite_first → must(chunk_type in [...])
else:                # full_reveal   → no filter
```

This is a data-plane constraint — the LLM cannot leak what it never received.

### 2. Corrective RAG (CRAG)

After retrieval, an LLM grades chunk relevance (yes/no). If all chunks fail, the query is reformulated via synonym expansion and retried (max 2 retries). Evidence of firing: ablation timing shows a 186s stall at q18 in the full variant vs. typical ~8s.

### 3. Dual Knowledge Masking

```python
class InternalAnalysis(BaseModel):
    correct_answer: str          # computed, never shown
    student_misconception: str
    planned_hint_sequence: list[str]

class VisibleResponse(BaseModel):
    socratic_question: str       # must end with "?"
    encouragement: str
```

Post-generation leak guard: ≥4 significant-word overlap between `socratic_question` and `correct_answer` triggers a retry (temperature 0).

### 4. Concept Prerequisite Graph

NetworkX DAG — e.g., `brachial_plexus.origin → brachial_plexus.trunks → peripheral_nerves.axillary`. When student struggles (consecutive_incorrect ≥ 2), `nx.ancestors()` traces prerequisite gaps. Cold-start diagnostic (4 Qs in Rapport) initializes mastery: correct → 0.5, incorrect → 0.1, skipped → 0.2.

Mastery update rule:
- Correct: `m' = m + 0.15 × (1 − m)`
- Incorrect: `m' = m − 0.05 × m`

### 5. Session Mistake Memory and Proactive Revisit

**What it stores:** Every incorrect response appends to `mistake_log` (Annotated append-only list in TutoringState):
```python
{"topic": str, "misconception": str, "turn": int, "elapsed_sec": float}
```
`misconception` is extracted from `InternalAnalysis.student_misconception` at the moment of the wrong answer.

**Trigger (orchestrator.py):** At `elapsed ≥ revisit_after_sec (480s)`, if `weak_topics` is non-empty and no revisit was triggered within the last `revisit_cooldown_sec (180s)`, the Orchestrator:
1. Picks the topic with the lowest current mastery from `weak_topics`
2. Sets `revisit_scheduled=True`, `revisit_topic=<topic>`, `current_topic=<topic>`
3. Records `_last_revisit_sec` for cooldown

**Retrieval augmentation (retrieval_planner.py):** When `revisit_scheduled`, query is augmented with the readable topic name → ensures Qdrant returns relevant chunks even if the student's latest message is off-topic.

**Prompt injection (socratic_generator.py):** A `REVISIT MODE` block is appended to the tutoring system prompt:
```
REVISIT MODE: The student previously struggled with '<topic>'.
Prior misconception: "<misconception text>"
Transition naturally to this topic with a Socratic question from a fresh angle.
```

**Cleanup (pedagogy_agent.py):** Sets `revisit_scheduled=False` after one turn so it doesn't loop.

---

## State Schema (TutoringState)

Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `mastery_scores` | dict[str, float] | Per-concept mastery [0,1] |
| `weak_topics` | list[str] | Concepts with mastery < 0.4 |
| `mistake_log` | Annotated[list[dict], operator.add] | Append-only mistake records |
| `revisit_scheduled` | bool | Set by orchestrator, cleared by pedagogy_agent |
| `revisit_topic` | Optional[str] | Which topic to revisit |
| `_last_revisit_sec` | float | Cooldown tracking |
| `conversation_history` | Annotated[list[dict], operator.add] | Full turn history |
| `_internal_analysis` | Optional[dict] | Hidden structured output |

**Important:** `conversation_history` uses `operator.add` — never re-pass accumulated history to `graph.invoke`. Always set `state["conversation_history"] = []` before invoking to prevent doubling.

---

## Evaluation Results (actual, as of March 2026)

| Metric | Score | Target | Pass |
|--------|-------|--------|------|
| Hit Rate @5 | 1.000 | ≥ 0.75 | ✓ |
| MRR | 0.917 | — | — |
| Leak Rate | 0.000 | 0% | ✓ |
| Ends with ? | 1.000 | ≥ 95% | ✓ |
| Avg Socratic Purity | 4.93/5 | ≥ 4.0 | ✓ |
| Adversarial Hold Rate | 1.000 | ≥ 90% | ✓ |
| RAGAS Faithfulness | 0.779 | ≥ 0.85 | ✗ (measurement mismatch — see §7.3) |
| RAGAS Answer Relevancy | 0.521 | ≥ 0.80 | ✗ (measurement mismatch) |

RAGAS failures are a known measurement mismatch: RAGAS penalizes Socratic questions that make no factual claims by design. Socratic Purity (4.93/5) is the appropriate groundedness metric.

### Ablation (30 questions/variant, mastery = 0.20)

| Variant | Ans. Chunk Reach | Leak Rate | Avg Purity |
|---------|-----------------|-----------|------------|
| full | 0.000 (correct) | 0.000 | 4.70 |
| no_pcr | 1.000 | 0.000 | 4.83 |
| no_crag | 0.000 | 0.000 | 4.87 |
| no_graph | 0.000 | 0.000 | 4.93 |

Key finding: zero leaks across all variants under benign conditions is the **benign-condition trap** — only adversarial testing reveals PCR's architectural advantage.

---

## Key Design Decisions

- **Manager Agent = pure Python** (not LLM-based) — DiagGPT (2023): rule-based controllers outperform LLM routers for deterministic transitions.
- **Structured output for masking** — `InternalAnalysis` / `VisibleResponse` split. Post-generation leak guard as third layer.
- **Revisit uses topic override, not just prompt** — without `current_topic` override, retrieval would be based on student message keywords, which may be irrelevant to the weak topic.
- **Mistake misconception carried forward** — using `internal_analysis.student_misconception` (LLM-generated at mistake time) gives the revisit richer context than just knowing the topic was wrong.
- **Cooldown prevents revisit spam** — without `_last_revisit_sec` + `revisit_cooldown_sec`, the orchestrator would re-trigger every turn after 8 min.
- **Unified Qdrant collection** for text + images — single hybrid search retrieves both; avoids two-pass retrieval overhead.
- **`consecutive_correct ≥ 2` threshold** — prevents premature exit from tutoring loop.

---

## Gotchas

- **`operator.add` doubling** — `conversation_history` accumulates via the checkpointer. Passing the full history to `graph.invoke` doubles it. Fix: always pass `conversation_history=[]` per turn (app.py, line ~56).
- **Qdrant concurrent access** — running `eval/run_eval.py` and `eval/ablation.py` simultaneously causes `portalocker.exceptions.AlreadyLocked`. Run sequentially.
- **Ollama fallback chain** — if Ollama is not running, rapport/wrapup fall back to GPT-4o API. No crash, just higher cost.
- **`revisit_scheduled` must be cleared** — pedagogy_agent resets it to `False`. If this clear is removed, revisit triggers every single turn after 8 min.
- **LaTeX natbib warning** — `report.tex` uses manual `\begin{thebibliography}` with numbered citations, but `acl.sty` loads natbib in author-year mode. Warning is harmless; PDF compiles correctly.

---

## TODO / Outstanding

- [ ] Task 4 (Multimodal VLM): Chainlit accepts image uploads; VLM backend (MedGemma 4B or GPT-4o Vision) not yet connected
- [ ] Cross-session persistence: `mistake_log` and mastery live in-memory (MemorySaver). For multi-session tracking, swap to SQLite checkpointer.
- [ ] Pilot study: 10 UB students (5 OT, 5 CS), 15-min sessions, pre/post quiz for learning gain
- [ ] Mistake memory evaluation: no current eval metric measures whether the revisit actually improves post-revisit performance
