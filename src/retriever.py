from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import UnifiedClipEmbedder
from .indexer import FaissChunkIndex
from .schema import ChunkRecord


@dataclass
class RetrievalResult:
    score: float
    chunk: ChunkRecord


def retrieve(
    query: str,
    embedder: UnifiedClipEmbedder,
    index: FaissChunkIndex,
    chunks: list[ChunkRecord],
    top_k: int = 6,
) -> list[RetrievalResult]:
    query_vec = embedder.embed_texts([query])
    scores, ids = index.search(query_vec, top_k=top_k)
    results: list[RetrievalResult] = []

    for score, idx in zip(scores[0], ids[0], strict=False):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(RetrievalResult(score=float(score), chunk=chunks[int(idx)]))

    return results


def group_context(results: list[RetrievalResult], max_chars: int = 5000) -> str:
    parts: list[str] = []
    current_len = 0
    for item in results:
        block = (
            f"[Source: {item.chunk.citation}]\n"
            f"Modality: {item.chunk.modality}\n"
            f"{item.chunk.text.strip()}\n"
        )
        if current_len + len(block) > max_chars:
            break
        parts.append(block)
        current_len += len(block)
    return "\n".join(parts)
