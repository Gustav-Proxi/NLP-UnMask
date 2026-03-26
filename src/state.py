"""LangGraph state schema for UnMask tutoring sessions."""
from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict
import operator


Phase = Literal["rapport", "tutoring", "assessment", "wrapup"]
RetrievalMode = Literal["context_only", "prerequisite_first", "full_reveal"]


class TutoringState(TypedDict):
    # Identity
    session_id: str

    # Current turn
    student_message: str
    turn_count: int

    # Session phase
    phase: Phase
    elapsed_seconds: float
    diagnostic_complete: bool

    # Current topic being tutored
    current_topic: Optional[str]

    # Per-concept mastery: concept_id -> P(mastery) in [0, 1]
    mastery_scores: dict[str, float]

    # PCR retrieval mode for this turn (set by retrieval_planner)
    retrieval_mode: RetrievalMode

    # Retrieved chunks (list of dicts with text, concept, is_answer_chunk, etc.)
    retrieved_chunks: list[dict]

    # Generated response — only visible_response.socratic_question + encouragement
    generated_response: str

    # Full structured output (internal_analysis stripped before delivery)
    _internal_analysis: Optional[dict]

    # Conversation history: [{role, content}]
    # Uses operator.add so nodes can append without overwriting
    conversation_history: Annotated[list[dict], operator.add]

    # Adaptive counters
    consecutive_correct: int
    consecutive_incorrect: int
    hints_used: int

    # Coverage ratio: proportion of concept graph nodes with mastery > threshold_high
    coverage_ratio: float

    # Weak topics for proactive revisit (set by pedagogy_agent)
    weak_topics: list[str]
