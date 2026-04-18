"""
Pedagogy Agent — mastery updates, concept prerequisite graph, diagnostic probe.

After each student response, this node:
  1. Evaluates if the response was correct (LLM judge using internal_analysis)
  2. Updates mastery scores via simple Bayesian update
  3. Traces prerequisite gaps in the concept graph
  4. Flags weak topics for proactive revisit
  5. Updates coverage_ratio and diagnostic_complete
"""
from __future__ import annotations

import json
import os
from typing import Optional

import networkx as nx
import yaml
from openai import OpenAI

from src.state import TutoringState

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_M = _cfg["mastery"]

# ── Concept graph (loaded once) ───────────────────────────────────────────────

_concept_graph: Optional[nx.DiGraph] = None


def _load_concept_graph() -> nx.DiGraph:
    global _concept_graph
    if _concept_graph is not None:
        return _concept_graph
    path = os.path.join(
        os.path.dirname(__file__), "..", "knowledge_base", "concept_graph.json"
    )
    with open(path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for concept_id, info in data["concepts"].items():
        G.add_node(concept_id, **{k: v for k, v in info.items() if k != "prerequisites"})
        for prereq in info.get("prerequisites", []):
            G.add_edge(prereq, concept_id)  # prereq → concept
    _concept_graph = G
    return G


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


# ── Correctness evaluation ────────────────────────────────────────────────────

def _evaluate_response(
    student_answer: str,
    internal: Optional[dict],
) -> tuple[bool, str]:
    """
    Returns (is_correct, feedback_reason).
    Uses internal_analysis.correct_answer as the gold standard.
    Falls back to heuristic if internal_analysis is not available.
    """
    if internal is None:
        # Rapport phase — no correctness judgment
        return True, "rapport"

    correct_answer = internal.get("correct_answer", "")
    if not correct_answer:
        return True, "no gold answer"

    client = _get_client()
    prompt = (
        f"Gold answer: {correct_answer}\n"
        f"Student's response: {student_answer}\n\n"
        "Is the student's response substantially correct? "
        "Reply with 'correct' or 'incorrect' followed by one sentence of reason."
    )
    resp = client.chat.completions.create(
        model=_cfg["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().lower()
    is_correct = text.startswith("correct")
    return is_correct, text


# ── Mastery update ────────────────────────────────────────────────────────────

def _update_mastery(current: float, is_correct: bool) -> float:
    if is_correct:
        updated = current + _M["correct_gain"] * (1 - current)
    else:
        updated = current - _M["incorrect_loss"] * current
    return max(0.0, min(1.0, updated))


# ── Coverage ratio ────────────────────────────────────────────────────────────

def _compute_coverage(mastery: dict[str, float], G: nx.DiGraph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    mastered = sum(
        1 for node in G.nodes
        if mastery.get(node, _M["default_prior"]) >= _cfg["pcr"]["threshold_high"]
    )
    return mastered / G.number_of_nodes()


# ── Prerequisite gap tracing ──────────────────────────────────────────────────

def _find_prerequisite_gaps(topic: str, mastery: dict, G: nx.DiGraph) -> list[str]:
    """
    Given a topic that the student struggled with, trace back to find
    prerequisite concepts with low mastery — the ROOT CAUSE of the failure.
    """
    if topic not in G:
        return []
    gaps = []
    for prereq in nx.ancestors(G, topic):
        if mastery.get(prereq, _M["default_prior"]) < _M["weak_threshold"]:
            gaps.append(prereq)
    return gaps


# ── Diagnostic probe ──────────────────────────────────────────────────────────

def get_diagnostic_order(study_focus: str) -> list[int]:
    """Return question indices [0-3] reordered by relevance to study_focus.
    0=brachial_plexus, 1=rotator_cuff, 2=axillary_nerve, 3=supraspinatus
    """
    focus = (study_focus or "").lower()
    if any(k in focus for k in ["brachial", "plexus", "nerve", "axillary", "radial", "median", "ulnar"]):
        return [0, 2, 3, 1]   # nerve-focused: lead with plexus, push rotator cuff last
    if any(k in focus for k in ["rotator", "cuff", "shoulder", "supraspinatus", "infraspinatus", "teres"]):
        return [1, 3, 2, 0]   # shoulder-focused: lead with rotator cuff
    return [0, 1, 2, 3]       # default order


_DIAGNOSTIC_PROMPTS = [
    "What spinal cord levels make up the brachial plexus?",
    "Name the four rotator cuff muscles.",
    "Which nerve innervates the deltoid muscle?",
    "What is the function of the supraspinatus?",
]

_DIAGNOSTIC_ANSWERS = {
    0: ["c5", "c6", "c7", "c8", "t1"],
    1: ["supraspinatus", "infraspinatus", "teres minor", "subscapularis"],
    2: ["axillary"],
    3: ["abduction", "abduct"],
}


def _init_mastery_from_diagnostic(
    answer: str, question_idx: int, mastery: dict
) -> dict:
    """Initialize mastery from a diagnostic probe answer."""
    ans = answer.lower()
    expected = _DIAGNOSTIC_ANSWERS.get(question_idx, [])
    score = sum(1 for e in expected if e in ans) / max(len(expected), 1)
    updated = dict(mastery)

    # Map diagnostic questions to concept IDs
    concept_map = {
        0: "brachial_plexus.origin",
        1: "rotator_cuff.muscles",
        2: "peripheral_nerves.axillary",
        3: "rotator_cuff.supraspinatus",
    }
    concept = concept_map.get(question_idx)
    if concept:
        updated[concept] = 0.5 if score > 0.5 else 0.1
    return updated


# ── Main node ─────────────────────────────────────────────────────────────────

def pedagogy_agent(state: TutoringState) -> dict:
    """
    Evaluate student response, update mastery, identify weak topics.
    """
    phase = state["phase"]
    topic = state.get("current_topic", "")
    mastery = dict(state.get("mastery_scores", {}))
    internal = state.get("_internal_analysis")
    student_msg = state["student_message"]
    turn = state["turn_count"]
    diagnostic_complete = state.get("diagnostic_complete", False)
    consecutive_correct = state.get("consecutive_correct", 0)
    consecutive_incorrect = state.get("consecutive_incorrect", 0)

    G = _load_concept_graph()

    # ── Rapport: handle diagnostic probe ──────────────────────────────────
    if phase == "rapport":
        # turn 1 = warmup exchange (no anatomy answer), turn 2+ = diagnostic answers
        # display_idx is 0-based position in the diagnostic sequence (0 = first Q shown)
        display_idx = turn - 2
        if 0 <= display_idx < len(_DIAGNOSTIC_PROMPTS):
            # Map display position back to the actual question ID via study_focus order
            order = get_diagnostic_order(state.get("study_focus") or "")
            actual_q_id = order[min(display_idx, len(order) - 1)]
            mastery = _init_mastery_from_diagnostic(student_msg, actual_q_id, mastery)

        complete = (turn - 1) >= _cfg["session"]["diagnostic_questions"]
        return {
            "mastery_scores": mastery,
            "diagnostic_complete": complete,
        }

    # ── Tutoring / Assessment: evaluate and update ─────────────────────────
    if not topic:
        # Try to extract topic from conversation context
        topic = _extract_topic_from_message(student_msg)

    is_correct, _ = _evaluate_response(student_msg, internal)

    if topic:
        current_mastery = mastery.get(topic, _M["default_prior"])
        mastery[topic] = _update_mastery(current_mastery, is_correct)

    if is_correct:
        consecutive_correct += 1
        consecutive_incorrect = 0
    else:
        consecutive_incorrect += 1
        consecutive_correct = 0

    coverage = _compute_coverage(mastery, G)

    # Find weak topics (for proactive revisit after 8 min)
    weak = [
        c for c, m in mastery.items()
        if m < _M["weak_threshold"]
    ]

    # Trace prerequisite gaps if student is struggling
    prereq_gaps = []
    if consecutive_incorrect >= _cfg["mastery"]["consecutive_incorrect_for_hint"] and topic:
        prereq_gaps = _find_prerequisite_gaps(topic, mastery, G)

    # Append to mistake log when student answers incorrectly
    new_mistakes = []
    if not is_correct and topic:
        misconception = (internal or {}).get("student_misconception", "")
        new_mistakes = [{
            "topic": topic,
            "misconception": misconception,
            "turn": turn,
            "elapsed_sec": round(state.get("elapsed_seconds", 0.0), 1),
        }]

    # ── Topic cycling: once current topic is mastered, move to next weakest ──
    next_topic = topic
    if topic and mastery.get(topic, 0) >= _cfg["pcr"]["threshold_high"]:
        # Find weakest concept not yet mastered
        candidates = [
            (c, m) for c, m in mastery.items()
            if m < _cfg["pcr"]["threshold_high"] and c != topic
        ]
        if candidates:
            next_topic = min(candidates, key=lambda x: x[1])[0]

    return {
        "mastery_scores": mastery,
        "consecutive_correct": consecutive_correct,
        "consecutive_incorrect": consecutive_incorrect,
        "coverage_ratio": coverage,
        "weak_topics": weak + prereq_gaps,
        "mistake_log": new_mistakes,   # Annotated[list, operator.add] — appends
        # Clear revisit_scheduled after one use so it doesn't loop indefinitely
        "revisit_scheduled": False,
        # Cycle to next weakest topic once current is mastered
        "current_topic": next_topic,
    }


def _extract_topic_from_message(msg: str) -> str:
    """Heuristic: match message to known concept IDs."""
    msg_lower = msg.lower()
    topic_keywords = {
        "brachial_plexus.origin": ["brachial plexus", "c5", "c6", "t1"],
        "brachial_plexus.trunks": ["trunk", "upper trunk", "lower trunk"],
        "brachial_plexus.cords": ["cord", "posterior cord", "lateral cord"],
        "peripheral_nerves.axillary": ["axillary nerve", "deltoid"],
        "peripheral_nerves.radial": ["radial nerve", "wrist drop"],
        "peripheral_nerves.median": ["median nerve", "carpal tunnel"],
        "peripheral_nerves.ulnar": ["ulnar nerve", "claw hand"],
        "rotator_cuff.muscles": ["rotator cuff", "supraspinatus", "infraspinatus"],
        "rotator_cuff.supraspinatus": ["supraspinatus", "abduction"],
    }
    for concept, keywords in topic_keywords.items():
        if any(kw in msg_lower for kw in keywords):
            return concept
    return ""


def generate_diagnostic_question(question_idx: int) -> str:
    """Return the next diagnostic probe question."""
    if question_idx < len(_DIAGNOSTIC_PROMPTS):
        return _DIAGNOSTIC_PROMPTS[question_idx]
    return ""
