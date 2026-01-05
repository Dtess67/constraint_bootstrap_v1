from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

Pulse = int # duration units, integer to keep it simple

@dataclass(frozen=True)
class Exchange:
    """A single turn: agent sends pulses, partner responds with pulses."""
    sent: Tuple[Pulse, ...]
    received: Tuple[Pulse, ...]

class ChannelV1:
    """
    A minimal shared channel.
    - Agent sends a tuple of pulse durations.
    - Partner returns a tuple of pulse durations.
    Optional noise can perturb pulses.
    """

    def __init__(self, noise_prob: float = 0.0, noise_jitter: int = 0, seed: int | None = None):
        self._noise_prob = float(noise_prob)
        self._noise_jitter = int(noise_jitter)
        self._rng = random.Random(seed)

    def transmit(self, sent: Tuple[Pulse, ...], received: Tuple[Pulse, ...]) -> Exchange:
        """Apply optional noise to both sent and received and return an Exchange."""
        sent_n = self._apply_noise(sent)
        recv_n = self._apply_noise(received)
        return Exchange(sent=sent_n, received=recv_n)

    def _apply_noise(self, pulses: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        if self._noise_prob <= 0.0 or self._noise_jitter <= 0:
            return pulses

        out: List[Pulse] = []
        for p in pulses:
            if self._rng.random() < self._noise_prob:
                jitter = self._rng.randint(-self._noise_jitter, self._noise_jitter)
                p2 = max(1, p + jitter)
                out.append(p2)
            else:
                out.append(p)
        return tuple(out)
