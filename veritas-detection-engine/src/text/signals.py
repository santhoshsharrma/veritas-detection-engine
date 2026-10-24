"""Lightweight text signals used as inputs to the agent."""
import re
import statistics


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def burstiness(text: str) -> float:
    """Coefficient of variation of sentence length. Human text tends to vary more.

    This is a heuristic signal, not a verdict. Calibrate it on real data.
    """
    lengths = [len(s.split()) for s in split_sentences(text)]
    if len(lengths) < 2:
        return 0.0
    mean = statistics.mean(lengths)
    return statistics.pstdev(lengths) / mean if mean else 0.0
