from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
