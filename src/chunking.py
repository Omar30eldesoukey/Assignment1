from __future__ import annotations

from typing import Iterable

from .schema import ChunkRecord


def split_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_records(records: Iterable[ChunkRecord]) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    for record in records:
        if record.modality == "text":
            parts = split_text(record.text)
            if len(parts) <= 1:
                out.append(record)
                continue
            for idx, part in enumerate(parts, start=1):
                out.append(
                    ChunkRecord(
                        chunk_id=f"{record.chunk_id}::part{idx}",
                        source_file=record.source_file,
                        page=record.page,
                        modality=record.modality,
                        text=part,
                        citation=f"{record.citation}, chunk {idx}",
                        metadata={**record.metadata, "chunk_part": idx},
                    )
                )
        else:
            out.append(record)
    return out
