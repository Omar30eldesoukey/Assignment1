from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .schema import ChunkRecord


class FaissChunkIndex:
    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.size:
            self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_embedding.astype(np.float32), top_k)


def save_index(index: FaissChunkIndex, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.index, str(output_dir / "index.faiss"))


def load_index(index_dir: Path) -> FaissChunkIndex:
    idx = faiss.read_index(str(index_dir / "index.faiss"))
    wrapper = FaissChunkIndex(dimension=idx.d)
    wrapper.index = idx
    return wrapper


def save_chunk_store(chunks: list[ChunkRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunk_store(index_dir: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return chunks
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunks.append(ChunkRecord(**item))
    return chunks
