"""
Socratic Generator — structured output with knowledge masking.

The model COMPUTES the correct answer (enabling a well-aimed question)
but the output schema provides no field to reveal it.
Only visible_response.socratic_question + encouragement reach the student.
"""
from __future__ import annotations

import os
from typing import Optional

import yaml
from openai import OpenAI
from pydantic import BaseModel

from src.state import TutoringState, Phase

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)


# ── Structured output schema (knowledge masking via schema) ───────────────────

class InternalAnalysis(BaseModel):
    """Stripped by app layer — never shown to student."""
    correct_answer: str
    student_misconception: str
    planned_hint_sequence: list[str]
    relevant_textbook_section: str


class VisibleResponse(BaseModel):
    """Only this is rendered in Chainlit."""
    socratic_question: str
    encouragement: str


class SocraticOutput(BaseModel):
    internal_analysis: InternalAnalysis
    visible_response: VisibleResponse


# ── System prompts ────────────────────────────────────────────────────────────

_RAPPORT_SYSTEM = """\
You are UnMask, a friendly Socratic tutor helping OT students prepare for the NBCOT exam.
You are currently running a short diagnostic to calibrate the student's starting level.
The diagnostic questions are provided externally — do NOT ask your own questions.
React naturally to the student's answer in 1-2 sentences: acknowledge if correct/incorrect
(without giving the full answer away), offer brief encouragement, and then stop.
Do not repeat previous encouragement phrases you have already used."""

_TUTORING_SYSTEM = """\
You are UnMask, a Socratic tutor for OT anatomy/neuroscience (NBCOT prep).

RULES (strictly enforced by output schema — you cannot violate them):
1. You KNOW the correct answer (it lives in internal_analysis.correct_answer).
2. You NEVER state the answer in visible_response. The schema has no field for it.
3. socratic_question must end with "?" and guide the student toward discovery.
4. Turns 1-2: broad conceptual questions only. No partial answers.
5. Calibrate your question to the student's demonstrated knowledge level.
6. If student misconception is detected, address it indirectly via a question.

CONTEXT CHUNKS (textbook source of truth):
{context}

STUDENT MASTERY LEVEL: {mastery:.0%} on current topic
RETRIEVAL MODE: {mode} (answer chunks {"NOT " if mode != "full_reveal" else ""}present in context)
CONVERSATION SO FAR: {history}
CURRENT TURN: {turn}
CONSECUTIVE INCORRECT: {consecutive_incorrect}
{revisit_block}"""

_ASSESSMENT_SYSTEM = """\
You are UnMask in assessment mode.
Present a clinical scenario (do NOT reveal the answer in socratic_question).
Ask the student to explain their clinical reasoning in free text.
The scenario must be grounded in the provided textbook chunks.
After the student responds, evaluate accuracy/completeness/reasoning quality.

CONTEXT CHUNKS:
{context}

MASTERY SCORES: {mastery_json}
"""

_WRAPUP_SYSTEM = """\
You are UnMask wrapping up a tutoring session.
Generate a brief, encouraging summary: what the student learned, which topics to review.
In socratic_question, ask one closing reflection question.
Keep internal_analysis brief (it is not shown to the student).

WEAK TOPICS: {weak_topics}
MASTERY SCORES: {mastery_json}
"""


# ── Client ────────────────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


def _use_local(phase: Phase) -> bool:
    """Route to Ollama for rapport/wrapup to save API budget."""
    return phase in _cfg["llm"].get("use_local_for", [])


def _call_ollama(system: str, user: str, history: list[dict] | None = None) -> str:
    """Call local Ollama as a plain text fallback (no structured output)."""
    import subprocess, json
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user})
    payload = {
        "model": _cfg["llm"]["local_model"],
        "messages": messages,
        "stream": False,
    }
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", "http://localhost:11434/api/chat",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=30,
    )
    return json.loads(result.stdout)["message"]["content"]


# ── Main node ─────────────────────────────────────────────────────────────────

def socratic_generator(state: TutoringState) -> dict:
    """
    Generate a Socratic response using structured output.
    Returns: generated_response (visible only), _internal_analysis (hidden).
    """
    phase = state["phase"]
    turn = state["turn_count"]
    mode = state.get("retrieval_mode", "context_only")
    chunks = state.get("retrieved_chunks", [])
    history = state.get("conversation_history", [])
    mastery = state.get("mastery_scores", {})
    topic = state.get("current_topic", "")

    context_text = "\n\n".join(
        f"[{c.get('chunk_type','context').upper()}] {c['text']}"
        for c in chunks
    ) or "(No context retrieved)"

    recent_history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    ) or "(Session start)"

    # ── Rapport / Wrapup: plain LLM (no structured schema) ─────────────────
    if phase in ("rapport", "wrapup"):
        if phase == "rapport":
            system = _RAPPORT_SYSTEM
            user = state["student_message"] or "Hello"
        else:
            import json
            system = _WRAPUP_SYSTEM.format(
                weak_topics=", ".join(state.get("weak_topics", [])) or "none",
                mastery_json=json.dumps(mastery, indent=2),
            )
            user = "Please give me a session summary."

        # Try local first; fall back to API — both use plain completions, no schema
        text = None
        if _use_local(phase):
            try:
                text = _call_ollama(system, user, history=history)
            except Exception:
                pass

        if text is None:
            client = _get_client()
            api_messages = [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": user}]
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", _cfg["llm"]["model"]),
                messages=api_messages,
                max_tokens=120,
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()

        return {
            "generated_response": text,
            "_internal_analysis": None,
            "conversation_history": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ],
            "turn_count": turn + 1,
        }

    # ── Tutoring / Assessment: structured output via OpenAI ────────────────
    if phase == "tutoring":
        # Build revisit block when orchestrator has scheduled a proactive revisit
        revisit_block = ""
        if state.get("revisit_scheduled") and state.get("revisit_topic"):
            rt = state["revisit_topic"]
            rt_readable = rt.replace("_", " ").replace(".", " ")
            # Pull misconception from the most recent mistake on this topic
            prior_misconception = next(
                (m["misconception"] for m in reversed(state.get("mistake_log", []))
                 if m["topic"] == rt and m["misconception"]),
                None,
            )
            misconception_hint = (
                f" The student previously showed this misconception: \"{prior_misconception}\"."
                if prior_misconception else ""
            )
            revisit_block = (
                f"\nREVISIT MODE: The student previously struggled with '{rt_readable}'."
                f"{misconception_hint}"
                f" Naturally transition the conversation back to this topic with a"
                f" Socratic question that probes the same concept from a fresh angle."
            )

        system = _TUTORING_SYSTEM.format(
            context=context_text,
            mastery=mastery.get(topic, _cfg["mastery"]["default_prior"]),
            mode=mode,
            history=recent_history,
            turn=turn,
            consecutive_incorrect=state.get("consecutive_incorrect", 0),
            revisit_block=revisit_block,
        )
    elif phase == "assessment":
        import json
        system = _ASSESSMENT_SYSTEM.format(
            context=context_text,
            mastery_json=json.dumps(mastery, indent=2),
        )
    else:
        system = _RAPPORT_SYSTEM  # fallback

    user_msg = state["student_message"]

    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        *history[-10:],
        {"role": "user", "content": user_msg},
    ]

    # ── Generate with post-generation leak guard (max 2 attempts) ──────────
    internal_analysis = None
    response_text = ""

    for attempt in range(2):
        resp = client.beta.chat.completions.parse(
            model=os.getenv("OPENAI_MODEL", _cfg["llm"]["model"]),
            temperature=_cfg["llm"]["temperature"] if attempt == 0 else 0,
            messages=messages,
            response_format=SocraticOutput,
        )
        output: SocraticOutput = resp.choices[0].message.parsed
        visible = output.visible_response
        candidate = visible.socratic_question
        if visible.encouragement:
            candidate = f"{visible.encouragement} {candidate}"

        internal_analysis = output.internal_analysis
        correct_answer = internal_analysis.correct_answer if internal_analysis else ""

        # Leak guard: check if response contains ≥3 words from the correct answer
        if correct_answer and _response_leaks_answer(candidate, correct_answer):
            if attempt == 0:
                # Inject explicit instruction and retry
                messages = [
                    {"role": "system", "content": system + "\n\nCRITICAL: Your previous response was too revealing. Do NOT mention specific anatomical names or values from the answer. Ask only a broad, open-ended guiding question."},
                    *history[-10:],
                    {"role": "user", "content": user_msg},
                ]
                continue
        break  # accept on second attempt regardless

        response_text = candidate

    response_text = candidate  # use last generated candidate

    return {
        "generated_response": response_text,
        "_internal_analysis": internal_analysis.model_dump() if internal_analysis else None,
        "conversation_history": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response_text},
        ],
        "turn_count": turn + 1,
    }


def _response_leaks_answer(response: str, correct_answer: str) -> bool:
    """
    Simple heuristic: if ≥3 significant words from the correct answer
    appear verbatim in the response, flag as a potential leak.
    """
    import re

    def significant_words(text: str) -> set[str]:
        stopwords = {"the", "a", "an", "is", "are", "of", "and", "to", "in", "for",
                     "it", "its", "that", "this", "by", "from", "with", "at", "be"}
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) >= 4 and w not in stopwords}

    answer_words = significant_words(correct_answer)
    response_words = significant_words(response)
    overlap = answer_words & response_words
    return len(overlap) >= 4  # 4+ significant words overlap signals a leak
