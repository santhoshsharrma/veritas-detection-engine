from pydantic import BaseModel, Field


class TextSignals(BaseModel):
    burstiness: float = Field(ge=0)
    perplexity: float | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class AuthenticityReport(BaseModel):
    verdict: str  # "likely_ai" | "likely_human" | "uncertain"
    confidence: float = Field(ge=0, le=1)
    image_ai_probability: float | None = None
    text_signals: TextSignals | None = None
    reasoning: str
