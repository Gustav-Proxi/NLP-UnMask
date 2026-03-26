"""
UnMask — Chainlit entry point.

Run:
  chainlit run app.py

The backend panel (visible in Chainlit's "debug" sidebar) shows:
  - Retrieved chunks with PCR mode
  - Mastery scores per concept
  - Current phase and turn count
"""
from __future__ import annotations

import time
import uuid

import chainlit as cl
import yaml

from src.graph import graph, make_initial_state
from src.nodes.pedagogy_agent import generate_diagnostic_question

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

_DIAGNOSTIC_QUESTIONS = cfg["session"]["diagnostic_questions"]


# ── Session lifecycle ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    state = make_initial_state(session_id)
    cl.user_session.set("state", state)
    cl.user_session.set("session_start", time.time())

    # Send opening rapport message directly (no LLM call for first turn)
    opening = (
        "Hey! I'm UnMask, your NBCOT anatomy tutor. "
        "I'll never just give you the answer — we'll work through everything together.\n\n"
        f"**Quick warm-up** ({_DIAGNOSTIC_QUESTIONS} questions to calibrate where we start):\n\n"
        f"{generate_diagnostic_question(0)}"
    )
    await cl.Message(content=opening).send()


@cl.on_message
async def on_message(message: cl.Message):
    state = cl.user_session.get("state")
    start_time = cl.user_session.get("session_start", time.time())

    # Update elapsed time
    state["elapsed_seconds"] = time.time() - start_time
    state["student_message"] = message.content

    # Run LangGraph
    config = {"configurable": {"thread_id": state["session_id"]}}
    result = graph.invoke(state, config=config)

    # Persist updated state
    cl.user_session.set("state", result)

    # ── Main response ────────────────────────────────────────────────────────
    response = result.get("generated_response", "")

    # During rapport/diagnostic, append next diagnostic question
    phase = result.get("phase", "rapport")
    turn = result.get("turn_count", 0)
    diagnostic_complete = result.get("diagnostic_complete", False)

    if phase == "rapport" and not diagnostic_complete:
        next_q = generate_diagnostic_question(turn)
        if next_q:
            response = response + f"\n\n{next_q}" if response else next_q

    await cl.Message(content=response).send()

    # ── Backend debug panel (instructor-visible metadata) ────────────────────
    mastery = result.get("mastery_scores", {})
    chunks = result.get("retrieved_chunks", [])
    retrieval_mode = result.get("retrieval_mode", "—")

    if chunks or mastery:
        elements = []

        if mastery:
            mastery_lines = "\n".join(
                f"  {'🟢' if v >= 0.7 else '🟡' if v >= 0.4 else '🔴'} {k}: {v:.2f}"
                for k, v in sorted(mastery.items())
            )
            elements.append(
                cl.Text(
                    name="📊 Mastery Scores",
                    content=f"```\n{mastery_lines}\n```",
                    display="side",
                )
            )

        if chunks:
            chunk_summary = f"**PCR Mode: `{retrieval_mode}`** — {len(chunks)} chunks retrieved\n\n"
            for i, c in enumerate(chunks[:5], 1):
                flag = "🔒 ANSWER" if c.get("is_answer_chunk") else f"📄 {c.get('chunk_type','ctx').upper()}"
                chunk_summary += f"**{i}. [{flag}]** `{c.get('concept','?')}`\n> {c['text'][:120]}...\n\n"
            elements.append(
                cl.Text(
                    name="🔍 Retrieved Chunks",
                    content=chunk_summary,
                    display="side",
                )
            )

        # Show session state
        session_info = (
            f"**Phase:** {phase}  |  **Turn:** {turn}  |  "
            f"**Elapsed:** {result.get('elapsed_seconds', 0):.0f}s\n"
            f"**Coverage:** {result.get('coverage_ratio', 0):.0%}  |  "
            f"**Consecutive correct:** {result.get('consecutive_correct', 0)}  |  "
            f"**Consecutive incorrect:** {result.get('consecutive_incorrect', 0)}"
        )
        elements.append(
            cl.Text(name="⚙️ Session State", content=session_info, display="side")
        )

        await cl.Message(content="", elements=elements).send()
