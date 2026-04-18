from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    page: int
    modality: str  # text | table | image
    text: str
    citation: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
