# Veritas: Agentic Multi-Modal AI-Content Detection

A compound AI system that combines CNN-based visual artifact detection with text signals
and a LangGraph agent that produces an explainable authenticity report.

> **Status:** early development (Phase 0). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Architecture (planned)
Input -> image CNN + text signals -> LangGraph agent -> Authenticity Report
(with a pgvector fingerprint cache for known content)

## Quickstart
```bash
cp .env.example .env
docker compose up --build
curl -X POST localhost:8000/v1/analyze/text -H "Content-Type: application/json" \
  -d '{"text": "Some text to analyze. It has two sentences."}'
```
Interactive docs: http://localhost:8000/docs

## Results
_Not yet measured. Benchmarks will be published here with methodology and
held-out-generator evaluation._

## Development
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
ruff check . && pytest
```
