from __future__ import annotations

import math
from dataclasses import dataclass
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
class MixedPartnerV1:
    """
    If len(sent) is prime: include 5
    If sum(sent) is prime: include 7
    Return the tuple of outputs in sorted order.
    """
    name: str = "mixed_v1"

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        if not sent:
            return ()
        
        results = []
        if is_prime(len(sent)):
            results.append(5)
        if is_prime(sum(sent)):
            results.append(7)
            
        return tuple(sorted(results))

register_partner("mixed", MixedPartnerV1)
