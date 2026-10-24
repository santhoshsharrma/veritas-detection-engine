from fastapi import FastAPI

from src.agent.graph import build_graph
from src.api.schemas import AnalyzeRequest, AuthenticityReport, TextSignals

app = FastAPI(title="Veritas", version="0.1.0")
_graph = build_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/analyze/text", response_model=AuthenticityReport)
def analyze_text(req: AnalyzeRequest) -> AuthenticityReport:
    out = _graph.invoke({"text": req.text})
    return AuthenticityReport(
        verdict=out["verdict"],
        confidence=out["confidence"],
        text_signals=TextSignals(burstiness=out["burstiness"]),
        reasoning=out["reasoning"],
    )
