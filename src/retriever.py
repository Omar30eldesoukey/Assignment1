from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .embeddings import UnifiedClipEmbedder
from .indexer import FaissChunkIndex
from .schema import ChunkRecord


@dataclass
class RetrievalResult:
    score: float
    chunk: ChunkRecord


def _is_informative_chunk(chunk: ChunkRecord) -> bool:
    text = chunk.text.strip()
    if not text:
        return False

    # Some extracted tables contain only separators like "|".
    if chunk.modality == "table" and text.replace("|", "").strip() == "":
        return False

    alnum_count = sum(ch.isalnum() for ch in text)
    if alnum_count < 25:
        return False

    # Filter chunks that are mostly punctuation or boilerplate symbols.
    cleaned = re.sub(r"[A-Za-z0-9\s]", "", text)
    return len(cleaned) < max(40, len(text) * 0.8)


def _is_image_intent(query: str) -> bool:
    q = query.lower()
    image_terms = (
        "image",
        "figure",
        "chart",
        "graph",
        "diagram",
        "map",
        "photo",
        "visual",
        "picture",
        "screenshot",
    )
    return any(term in q for term in image_terms)


def retrieve(
    query: str,
    embedder: UnifiedClipEmbedder,
    index: FaissChunkIndex,
    chunks: list[ChunkRecord],
    top_k: int = 6,
) -> list[RetrievalResult]:
    query_vec = embedder.embed_texts([query])
    search_k = max(top_k, min(len(chunks), top_k * 8))
    scores, ids = index.search(query_vec, top_k=search_k)
    image_intent = _is_image_intent(query)

    informative: list[RetrievalResult] = []
    fallback: list[RetrievalResult] = []
    seen_indices: set[int] = set()

    for score, idx in zip(scores[0], ids[0], strict=False):
        if idx < 0 or idx >= len(chunks):
            continue
        idx_int = int(idx)
        if idx_int in seen_indices:
            continue
        seen_indices.add(idx_int)

        result = RetrievalResult(score=float(score), chunk=chunks[idx_int])
        if _is_informative_chunk(result.chunk):
            informative.append(result)
        else:
            fallback.append(result)

    informative_images = [r for r in informative if r.chunk.modality == "image"]
    informative_non_images = [r for r in informative if r.chunk.modality != "image"]
    fallback_images = [r for r in fallback if r.chunk.modality == "image"]
    fallback_non_images = [r for r in fallback if r.chunk.modality != "image"]

    results: list[RetrievalResult] = []
    used_chunk_ids: set[str] = set()

    def add_result(item: RetrievalResult) -> None:
        if item.chunk.chunk_id in used_chunk_ids:
            return
        if len(results) >= top_k:
            return
        results.append(item)
        used_chunk_ids.add(item.chunk.chunk_id)

    if image_intent:
        target_image_count = max(1, top_k // 2)
        for item in informative_images[:target_image_count]:
            add_result(item)
        if len(results) < target_image_count:
            for item in fallback_images[: target_image_count - len(results)]:
                add_result(item)

    for item in informative_non_images:
        add_result(item)
    for item in informative_images:
        add_result(item)
    if len(results) < top_k:
        for item in fallback_non_images:
            add_result(item)
        for item in fallback_images:
            add_result(item)

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
