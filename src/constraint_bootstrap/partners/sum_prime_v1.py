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
class SumPrimePartnerV1:
    """
    Responds with (7,) if sum(sent) is prime.
    Otherwise responds with silence (empty tuple).
    """
    name: str = "sum_prime_v1"

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        if not sent:
            return ()
        s = sum(sent)
        return (7,) if is_prime(s) else ()

register_partner("sumprime", SumPrimePartnerV1)
