from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from . import Pulse, register_partner

def _is_prime(n: int) -> bool:
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
class PrimeCountPartnerV1:
    """
    Responds with a single pulse if the number of sent pulses is prime.
    Otherwise responds with silence (empty tuple).
    """
    name: str = "prime_count_v1"

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        n = len(sent)
        return (5,) if _is_prime(n) else ()

@dataclass
class RatioPartnerV1:
    """
    Responds if it detects a 2:1 ratio between the first two pulses (within exact integer ratio).
    Example: (2,1,...) or (6,3,...) triggers.
    Response echoes a simple signature pulse length equal to gcd(a,b).
    """
    name: str = "ratio_2_to_1_v1"

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        if len(sent) < 2:
            return ()
        a, b = sent[0], sent[1]
        if a == 2 * b or b == 2 * a:
            g = math.gcd(a, b)
            return (g, g)  # a stable, repeatable signature
        return ()

@dataclass
class SymmetryPartnerV1:
    """
    Responds with a pulse if the sent sequence is a palindrome (perfect symmetry).
    Response length encodes the middle pulse if odd length, else 4.
    """
    name: str = "symmetry_palindrome_v1"

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        if len(sent) < 3:
            return ()
        if sent == tuple(reversed(sent)):
            if len(sent) % 2 == 1:
                mid = sent[len(sent)//2]
                return (max(1, mid),)
            return (4,)
        return ()

register_partner("prime", PrimeCountPartnerV1)
register_partner("prime_count", PrimeCountPartnerV1)
register_partner("ratio", RatioPartnerV1)
register_partner("2to1", RatioPartnerV1)
register_partner("symmetry", SymmetryPartnerV1)
register_partner("palindrome", SymmetryPartnerV1)
