"""LangGraph workflow. Phase 4 will replace the rule-based node with an LLM node.

For now the graph wires up state flow so the structure is testable end to end.
"""
from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.text.signals import burstiness


def text_node(state: AgentState) -> AgentState:
    return {"burstiness": burstiness(state.get("text", ""))}


def decide_node(state: AgentState) -> AgentState:
    # TODO(phase 4): LLM-based reasoning over vision + text signals.
    img = state.get("image_ai_probability")
    if img is None:
        return {"verdict": "uncertain", "confidence": 0.0,
                "reasoning": "No image signal available; text-only fusion not implemented yet."}
    verdict = "likely_ai" if img >= 0.7 else "likely_human"
    return {"verdict": verdict, "confidence": img if img >= 0.5 else 1 - img,
            "reasoning": "Placeholder rule on image probability."}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("text_signals", text_node)
    g.add_node("decide", decide_node)
    g.set_entry_point("text_signals")
    g.add_edge("text_signals", "decide")
    g.add_edge("decide", END)
    return g.compile()
