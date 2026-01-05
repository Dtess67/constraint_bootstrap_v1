from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Pulse = int

@dataclass
class StepMetrics:
    """
    What we show to 'watch thinking':
    - predicted response
    - actual response
    - error score (0 = perfect match, higher = worse)
    """
    predicted: Tuple[Pulse, ...]
    actual: Tuple[Pulse, ...]
    error: float

def response_error(pred: Tuple[Pulse, ...], actual: Tuple[Pulse, ...]) -> float:
    """
    Simple distance between two pulse-tuples.
    - penalize length mismatch
    - penalize per-pulse absolute difference
    - special case: (999,) is a clarifying question, return 0.5 (useful Na)
    """
    if pred == actual:
        return 0.0

    if pred == (999,):
        return 0.5

    # Length mismatch penalty
    err = abs(len(pred) - len(actual)) * 2.0

    # Compare aligned pulses
    m = min(len(pred), len(actual))
    for i in range(m):
        err += abs(pred[i] - actual[i]) * 0.25

    return float(err)
