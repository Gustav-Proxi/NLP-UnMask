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
from src.anatomy_images import get_image_for_topic

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

_DIAGNOSTIC_QUESTIONS = cfg["session"]["diagnostic_questions"]


def _fmt_diag_q(idx: int, question: str) -> str:
    """Format a diagnostic question with a numbered header and separator."""
    return f"---\n\n**🩺 Question {idx + 1} of {_DIAGNOSTIC_QUESTIONS}:**\n\n{question}"

# Phase display names and icons
_PHASE_INFO = {
    "rapport":    ("🩺 Diagnostic",  "#3B82F6"),
    "tutoring":   ("📖 Tutoring",    "#10B981"),
    "assessment": ("🧪 Assessment",  "#F59E0B"),
    "wrapup":     ("📋 Session End", "#8B5CF6"),
}

_PHASE_TRANSITION_MSGS = {
    ("rapport",    "tutoring"):    (
        "## 🎓 Diagnostic Complete — Starting Tutoring\n\n"
        "I've calibrated your starting point based on your diagnostic answers. "
        "We'll now dive into the topics that need the most attention using the Socratic method — "
        "I'll guide you with questions rather than answers. Let's go!"
    ),
    ("tutoring",   "assessment"): (
        "## 🧪 Tutoring Complete — Moving to Assessment\n\n"
        "You've covered strong ground in the tutoring phase! "
        "Now let's put your knowledge to the test with a clinical scenario. "
        "I'll present a realistic NBCOT-style case — explain your reasoning out loud."
    ),
    ("assessment", "wrapup"): (
        "## 📋 Assessment Complete — Generating Your Report\n\n"
        "Session complete! Compiling your performance report with personalised "
        "study recommendations and follow-up questions..."
    ),
    ("tutoring",   "wrapup"): (
        "## 📋 Session Time Up — Generating Your Report\n\n"
        "Time's up for today! Compiling your session report with an honest breakdown "
        "of what you've learned and what still needs work..."
    ),
    ("rapport",    "wrapup"): (
        "## 📋 Session Ended\n\nGenerating your session report..."
    ),
}


def _parse_onboarding(text: str) -> tuple[str, str]:
    """Extract study_focus and learning_mode from the user's first greeting reply."""
    lower = text.lower()
    if any(w in lower for w in ("diagram", "visual", "image", "picture", "figure", "draw")):
        learning_mode = "visual"
    else:
        learning_mode = "text"
    if any(w in lower for w in ("revise", "revision", "review", "recap", "revisit")):
        study_focus = "revision"
    elif any(w in lower for w in ("everything", "all", "full", "complete", "entire", "whole")):
        study_focus = "everything"
    else:
        study_focus = f"specific: {text.strip()[:120]}"
    return study_focus, learning_mode


# ── Session lifecycle ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    state = make_initial_state(session_id)
    cl.user_session.set("state", state)
    cl.user_session.set("session_start", time.time())
    cl.user_session.set("warmup_done", False)    # Track casual warm-up exchange
    cl.user_session.set("diag_q_index", 0)      # Next diagnostic question to inject

    # Warm greeting — asks how they're doing, what to focus on, and learning preference
    welcome = (
        "# 👋 Hi! Welcome to UnMask\n\n"
        "I'm your personal NBCOT anatomy study companion. "
        "I use the **Socratic method** — meaning I guide you *toward* answers rather than just telling you. "
        "It's more work, but the understanding sticks way better for exams. 💪\n\n"
        "**Here's the plan for today's 15-minute session:**\n"
        "1. 🩺 **Quick Diagnostic** ({n} questions) — calibrates where we start\n"
        "2. 📖 **Targeted Tutoring** — Socratic deep-dives on your weak spots\n"
        "3. 🧪 **Clinical Assessment** — a real-world NBCOT-style scenario\n"
        "4. 📋 **Session Report** — honest feedback, flashcards, resources & diagrams\n\n"
        "---\n\n"
        "**Before we dive in — three quick things:**\n\n"
        "1. **How's studying going?** (Helps me know where your head is at)\n"
        "2. **What would you like to focus on today?**\n"
        "   - A specific concept or muscle group *(tell me which one)*\n"
        "   - Cover everything systematically\n"
        "   - Revise and reinforce what you've already covered\n"
        "3. **How do you learn best?**\n"
        "   - Written explanations & Q&A *(text)*\n"
        "   - Diagrams and visual breakdowns *(visual)*\n\n"
        "Just reply naturally — I'll pick up on your preferences and we'll get going! 😊"
    ).format(n=_DIAGNOSTIC_QUESTIONS)

    await cl.Message(content=welcome, author="UnMask").send()


@cl.on_message
async def on_message(message: cl.Message):
    state = cl.user_session.get("state")
    start_time = cl.user_session.get("session_start", time.time())

    prev_phase = state.get("phase", "rapport")

    # Update elapsed time
    state["elapsed_seconds"] = time.time() - start_time
    state["student_message"] = message.content
    # IMPORTANT: Never re-pass accumulated conversation_history to graph.invoke.
    # TutoringState.conversation_history uses operator.add, so passing the full
    # history again would double it on every turn (checkpointer already holds it).
    state["conversation_history"] = []

    # ── Capture study preferences from first message (before graph call) ────
    warmup_already_done = cl.user_session.get("warmup_done", False)
    if not warmup_already_done:
        study_focus, learning_mode = _parse_onboarding(message.content)
        state["study_focus"] = study_focus
        state["learning_mode"] = learning_mode

    # ── Show loading indicator immediately ──────────────────────────────────
    thinking_msg = cl.Message(content="⏳ *Thinking...*", author="UnMask")
    await thinking_msg.send()

    # Run LangGraph
    config = {"configurable": {"thread_id": state["session_id"]}}
    result = graph.invoke(state, config=config)

    # ── If diagnostic just completed, immediately re-invoke to get first tutoring Q ─
    # (Orchestrator transitions rapport→tutoring only on the *next* invocation,
    #  so we fire a second silent invoke here to avoid leaving the user with a blank.)
    prev_diag_complete = state.get("diagnostic_complete", False)
    just_finished_diag = (not prev_diag_complete) and result.get("diagnostic_complete", False)

    if just_finished_diag:
        # Start on the weakest diagnosed topic so the first tutoring Q is relevant
        mastery = result.get("mastery_scores", {})
        if mastery:
            weakest = min(mastery, key=lambda k: mastery[k])
            trigger_msg = f"Let's work on {weakest.replace('_', ' ').replace('.', ' ')}"
        else:
            trigger_msg = "Let's start tutoring on upper limb anatomy"
        result["student_message"] = trigger_msg
        result["conversation_history"] = []
        result["elapsed_seconds"] = time.time() - start_time
        result = graph.invoke(result, config=config)

    # Persist updated state
    cl.user_session.set("state", result)

    phase = result.get("phase", "rapport")
    turn = result.get("turn_count", 0)
    diagnostic_complete = result.get("diagnostic_complete", False)

    # ── Phase transition banner ──────────────────────────────────────────────
    if prev_phase != phase:
        transition_key = (prev_phase, phase)
        transition_msg = _PHASE_TRANSITION_MSGS.get(transition_key)
        if transition_msg:
            # Replace the thinking placeholder with the transition banner
            thinking_msg.content = transition_msg
            thinking_msg.author = "🔄 Phase Transition"
            await thinking_msg.update()
            thinking_msg = None  # send response as a new message below

    # ── Main response ────────────────────────────────────────────────────────
    response = result.get("generated_response", "")
    warmup_done = cl.user_session.get("warmup_done", False)

    # During rapport: first turn = warmup response, then start injecting diagnostic Qs
    if phase == "rapport" and not diagnostic_complete:
        diag_idx = cl.user_session.get("diag_q_index", 0)
        if not warmup_done:
            q0 = generate_diagnostic_question(0)
            q0_block = _fmt_diag_q(0, q0)
            response = (response + f"\n\n{q0_block}") if response else q0_block
            cl.user_session.set("warmup_done", True)
            cl.user_session.set("diag_q_index", 1)
        else:
            next_q = generate_diagnostic_question(diag_idx)
            if next_q:
                q_block = _fmt_diag_q(diag_idx, next_q)
                response = (response + f"\n\n{q_block}") if response else q_block
                cl.user_session.set("diag_q_index", diag_idx + 1)

    # Determine author label
    author_map = {
        "wrapup":     "📋 Session Report",
        "assessment": "🧪 Assessment",
        "tutoring":   "📖 Tutor",
    }
    author = author_map.get(phase, "UnMask")

    # Update thinking placeholder with the tutor response first
    if thinking_msg is not None:
        thinking_msg.content = response
        thinking_msg.author = author
        await thinking_msg.update()
    else:
        await cl.Message(content=response, author=author).send()

    # ── Visual hint card — always a fresh send() so cl.Image renders correctly ──
    visual_hint = result.get("visual_hint")
    if visual_hint and phase == "tutoring":
        # Extract concept id embedded in hint text: "__concept__:id\ntext"
        hint_text = visual_hint
        hint_concept = result.get("current_topic") or ""
        if visual_hint.startswith("__concept__:"):
            first_newline = visual_hint.index("\n")
            hint_concept = visual_hint[len("__concept__:"):first_newline].strip()
            hint_text = visual_hint[first_newline + 1:].strip()

        img_data = get_image_for_topic(hint_concept) or get_image_for_topic(result.get("current_topic") or "")

        concept_label = hint_concept.replace("_", " ").replace(".", " › ").title()

        if img_data:
            await cl.Message(
                content=(
                    f"### 🖼️ Visual Reference — {concept_label}\n\n"
                    f"📌 *{img_data['caption']}*\n\n"
                    f"```\n{img_data['diagram']}\n```\n\n"
                    f"---\n*Study this, then try the question below.*"
                ),
                author="🖼️ Visual Aid",
            ).send()
        else:
            await cl.Message(
                content=f"### 🖼️ Reference — {concept_label}\n\n```\n{hint_text}\n```\n\n---\n*Study this, then try again.*",
                author="🖼️ Visual Aid",
            ).send()

    if phase == "wrapup":
        await _send_followup_resources(result)

    # ── Assessment feedback (separate styled message) ────────────────────────
    assessment_feedback = result.get("assessment_feedback")
    if assessment_feedback and phase == "assessment":
        await cl.Message(
            content=assessment_feedback,
            author="📝 Assessment Feedback",
        ).send()

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

        # Session state + mistake log + revisit status
        mistake_log = result.get("mistake_log", [])
        revisit_info = ""
        if result.get("revisit_scheduled"):
            rt = result.get("revisit_topic", "")
            revisit_info = f"\n**Revisit scheduled:** {rt.replace('_',' ').replace('.',' › ')}"
        session_info = (
            f"**Phase:** {phase}  |  **Turn:** {turn}  |  "
            f"**Elapsed:** {result.get('elapsed_seconds', 0):.0f}s\n"
            f"**Coverage:** {result.get('coverage_ratio', 0):.0%}  |  "
            f"**Consecutive correct:** {result.get('consecutive_correct', 0)}  |  "
            f"**Consecutive incorrect:** {result.get('consecutive_incorrect', 0)}"
            f"{revisit_info}"
        )
        elements.append(
            cl.Text(name="⚙️ Session State", content=session_info, display="side")
        )

        if mistake_log:
            mistake_text = "\n".join(
                f"Turn {m['turn']} | {m['topic'].replace('_',' ').replace('.',' › ')}: "
                f"{m.get('misconception','—') or '—'}"
                for m in mistake_log
            )
            elements.append(
                cl.Text(
                    name="⚠️ Mistake Log",
                    content=f"```\n{mistake_text}\n```",
                    display="side",
                )
            )

        await cl.Message(content="", elements=elements).send()


async def _send_followup_resources(result: dict) -> None:
    """
    After a wrapup, send a separate flashcard + diagram message from the structured
    SessionSummary stored in _internal_analysis.
    """
    internal = result.get("_internal_analysis")
    if not internal:
        return

    flashcards = internal.get("flashcards", [])
    diagrams = internal.get("diagram_suggestions", [])

    # ── Flashcard message ────────────────────────────────────────────────────
    if flashcards:
        lines = ["## 🃏 Your Session Flashcards\n"]
        lines.append("*Cover the answer, read the question aloud — then reveal. Repeat daily!*\n")
        lines.append("---\n")
        for i, fc in enumerate(flashcards, 1):
            if isinstance(fc, dict):
                concept = fc.get("concept", "").replace("_", " ").replace(".", " › ")
                front = fc.get("front", "")
                back = fc.get("back", "")
            else:
                concept = getattr(fc, "concept", "").replace("_", " ").replace(".", " › ")
                front = getattr(fc, "front", "")
                back = getattr(fc, "back", "")
            lines.append(f"**Card {i}** — `{concept}`")
            lines.append(f"**Q:** {front}")
            lines.append(f"**A:** {back}\n")

        await cl.Message(
            content="\n".join(lines),
            author="🃏 Flashcards",
        ).send()

    # ── Diagram suggestions message ──────────────────────────────────────────
    if diagrams:
        from src.anatomy_images import get_image_for_topic as _get_img
        lines = ["## 🖼️ Diagrams to Study\n"]
        lines.append("*Study each diagram — then try drawing it from memory:*\n")
        for d in diagrams:
            lines.append(f"**→** {d}\n")
            for key in ["brachial_plexus", "rotator_cuff", "peripheral_nerves", "shoulder_joint",
                        "median", "ulnar", "radial", "axillary", "subscapularis", "supraspinatus"]:
                if key.replace("_", " ") in d.lower() or key in d.lower():
                    img = _get_img(key)
                    if img:
                        lines.append(f"```\n{img['diagram']}\n```\n")
                    break
        lines.append("---\n*Tip: Cover the diagram, redraw from memory, then check.*")

        await cl.Message(
            content="\n".join(lines),
            author="🖼️ Study Diagrams",
        ).send()
