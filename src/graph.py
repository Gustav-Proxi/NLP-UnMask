"""
LangGraph state machine for UnMask.

Graph topology:
  START
    └─► orchestrator
          └─► [rapport/wrapup] socratic_generator
          └─► [tutoring/assessment] retrieval_planner → socratic_generator
                                                             └─► pedagogy_agent
                                                                      └─► END
"""
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import TutoringState
from src.nodes.orchestrator import orchestrator, should_retrieve
from src.nodes.retrieval_planner import retrieval_planner
from src.nodes.socratic_generator import socratic_generator
from src.nodes.pedagogy_agent import pedagogy_agent


def build_graph() -> StateGraph:
    builder = StateGraph(TutoringState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("retrieval_planner", retrieval_planner)
    builder.add_node("socratic_generator", socratic_generator)
    builder.add_node("pedagogy_agent", pedagogy_agent)

    builder.add_edge(START, "orchestrator")

    # After orchestrator: skip retrieval for rapport/wrapup
    builder.add_conditional_edges(
        "orchestrator",
        should_retrieve,
        {
            "retrieval_planner": "retrieval_planner",
            "socratic_generator": "socratic_generator",
        },
    )

    builder.add_edge("retrieval_planner", "socratic_generator")
    builder.add_edge("socratic_generator", "pedagogy_agent")
    builder.add_edge("pedagogy_agent", END)

    return builder


# Compiled graph with in-memory checkpointer (MemorySaver maintains session state)
checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)


def make_initial_state(session_id: str | None = None) -> TutoringState:
    """Create a fresh session state."""
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    return TutoringState(
        session_id=session_id or str(uuid.uuid4()),
        student_message="",
        turn_count=0,
        phase="rapport",
        elapsed_seconds=0.0,
        diagnostic_complete=False,
        current_topic=None,
        mastery_scores={},
        retrieval_mode="context_only",
        retrieved_chunks=[],
        generated_response="",
        _internal_analysis=None,
        conversation_history=[],
        consecutive_correct=0,
        consecutive_incorrect=0,
        hints_used=0,
        coverage_ratio=0.0,
        weak_topics=[],
        mistake_log=[],
        revisit_scheduled=False,
        revisit_topic=None,
        _last_revisit_sec=0.0,
    )
