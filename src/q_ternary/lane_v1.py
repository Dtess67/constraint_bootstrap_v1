from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple

class Lane(Enum):
    SPEAK = "SPEAK"
    QUESTION = "QUESTION"
    NA = "NA"
    SILENT = "SILENT"

@dataclass(frozen=True)
class Decision:
    lane: Lane
    act: Optional[Tuple[int, ...]] = None
    question: Optional[str] = None
    meta: dict = field(default_factory=dict)
