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
class MixedShiftLargePartnerV1:
    """
    For steps 0..4999 use 'mixed' rule.
    For steps 5000..9999 use 'parity' rule.
    """
    name: str = "mixed_shift_large_v1"
    step_count: int = field(default=0, init=False)
    split_point: int = 5000

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        phase = 0 if self.step_count < self.split_point else 1
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
            # Parity rule
            s = sum(sent)
            return (2,) if s % 2 == 0 else (1,)

register_partner("mixed_shift_large", MixedShiftLargePartnerV1)
