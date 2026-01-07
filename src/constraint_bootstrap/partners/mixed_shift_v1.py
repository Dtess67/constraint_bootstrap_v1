from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from . import Pulse, register_partner

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for k in range(3, r + 1, 2):
        if n % k == 0:
            return False
    return True

@dataclass
class MixedShiftPartnerV1:
    """
    For steps 0..499 use 'mixed' rule.
    For steps 500..999 use 'sum_prime' rule.
    """
    name: str = "mixed_shift_v1"
    step_count: int = field(default=0, init=False)

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        phase = 0 if self.step_count < 500 else 1
        self.step_count += 1
        
        if not sent:
            return ()
        
        if phase == 0:
            # Mixed rule
            results = []
            if is_prime(len(sent)):
                results.append(5)
            if is_prime(sum(sent)):
                results.append(7)
            return tuple(sorted(results))
        else:
            # Rule B: Parity rule (switch to something very different)
            # If sum(sent) is even, respond with (2,). Otherwise (1,).
            s = sum(sent)
            return (2,) if s % 2 == 0 else (1,)

register_partner("mixed_shift", MixedShiftPartnerV1)
