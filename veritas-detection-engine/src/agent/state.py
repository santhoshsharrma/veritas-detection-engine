from typing import TypedDict


class AgentState(TypedDict, total=False):
    text: str
    image_ai_probability: float | None
    burstiness: float | None
    perplexity: float | None
    verdict: str
    confidence: float
    reasoning: str
