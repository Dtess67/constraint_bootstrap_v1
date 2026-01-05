from __future__ import annotations

from typing import Tuple, Protocol, Type

Pulse = int

class Partner(Protocol):
    name: str

    def respond(self, sent: Tuple[Pulse, ...]) -> Tuple[Pulse, ...]:
        ...

_REGISTRY: dict[str, Type[Partner]] = {}

def register_partner(kind: str, cls: Type[Partner]):
    _REGISTRY[kind.lower()] = cls

def make_partner(kind: str) -> Partner:
    # Ensure all partners are registered
    from . import legacy_v1, sum_prime_v1, mixed_v1, adversarial_partner_v1
    
    kind = kind.strip().lower()
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown partner kind: {kind!r}. Registered: {list(_REGISTRY.keys())}")
    return _REGISTRY[kind]()
