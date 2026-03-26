"""
Orchestrator node — pure Python, zero LLM calls.

Determines phase transitions and next routing target based on session state.
Implements the state machine from the UnMask v4 spec.
"""
import time
from typing import Literal

import yaml

from src.state import TutoringState, Phase

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_S = _cfg["session"]
_M = _cfg["mastery"]


def orchestrator(state: TutoringState) -> dict:
    """
    Route the session to the next phase.
    Returns partial state updates only (phase, next routing signal).
    """
    phase = state["phase"]
    elapsed = state["elapsed_seconds"]
    consecutive_correct = state["consecutive_correct"]
    consecutive_incorrect = state["consecutive_incorrect"]
    coverage = state["coverage_ratio"]
    diagnostic_complete = state["diagnostic_complete"]

    # ── Time-based ceiling overrides (highest priority) ──────────────────────
    if elapsed >= _S["wrapup_cutoff_sec"] and phase not in ("wrapup",):
        return {"phase": "wrapup"}

    if elapsed >= _S["assessment_cutoff_sec"] and phase not in ("assessment", "wrapup"):
        return {"phase": "assessment"}

    # ── Event-based transitions ───────────────────────────────────────────────
    if phase == "rapport":
        if diagnostic_complete:
            return {"phase": "tutoring"}
        # Stay in rapport until diagnostic is done

    elif phase == "tutoring":
        # Advance if student has demonstrated sufficient mastery
        if consecutive_correct >= _M["consecutive_correct_for_advance"]:
            if coverage >= 0.8:
                return {"phase": "assessment", "consecutive_correct": 0}
        # Otherwise stay in tutoring

    elif phase == "assessment":
        # Assessment always runs to completion (wrapup triggered by time ceiling)
        pass

    # No phase change — return current phase unchanged
    return {"phase": phase}


def should_retrieve(state: TutoringState) -> Literal["retrieval_planner", "socratic_generator"]:
    """Conditional edge: skip retrieval for rapport/wrapup phases."""
    if state["phase"] in ("rapport", "wrapup"):
        return "socratic_generator"
    return "retrieval_planner"
