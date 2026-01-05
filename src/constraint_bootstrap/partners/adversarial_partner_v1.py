from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple, List

from . import Pulse, register_partner
from .mixed_v1 import MixedPartnerV1

@dataclass
class AdversarialPartnerV1:
    """
    Adversarial partner with seasons and concept drift.
    - Seasonal ambiguity: target_sig flips dominant response between [5] and [7].
    - Concept drift: drift_sig flips mapping after drift_step.
    - Fallback: uses MixedPartnerV1 logic.
    """
    name: str = "adversarial_v1"
    seed: int = 42
    season_len: int = 200
    drift_step: int | None = 500
    p_major: float = 0.75
    target_sig: str = "8,4"
    drift_sig: str = "2,2"
    
    _rng: random.Random = field(init=False)
    _mixed: MixedPartnerV1 = field(init=False)
    _total_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._mixed = MixedPartnerV1()

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        t = self._total_steps
        self._total_steps += 1
        
        # Determine signature
        sent_s = ",".join(map(str, sent)) if sent else "0"
        
        # Seasonal ambiguity for target_sig
        if sent_s == self.target_sig:
            season = (t // self.season_len) % 2
            roll = self._rng.random()
            if season == 0:
                # Season 0: favors [5]
                return (5,) if roll < self.p_major else (7,)
            else:
                # Season 1: favors [7]
                return (7,) if roll < self.p_major else (5,)

        # Concept drift for drift_sig
        if self.drift_step is not None and t >= self.drift_step and sent_s == self.drift_sig:
            # Swap what MixedPartnerV1 would have returned
            # MixedPartnerV1 for [3]: len=1 (not prime), sum=3 (prime) -> (7,)
            # After drift, we swap it to (5,) if it was (7,) or vice versa?
            # Spec: "swap what it used to output"
            # Let's see what MixedPartnerV1 does for [3]
            base_resp = self._mixed.respond(sent)
            if base_resp == (7,):
                return (5,)
            elif base_resp == (5,):
                return (7,)
            elif base_resp == (5, 7):
                return ()
            else:
                return (5, 7)

        # Fallback to mixed_v1
        return self._mixed.respond(sent)

register_partner("adversarial", AdversarialPartnerV1)
register_partner("adversarial_v1", AdversarialPartnerV1)
