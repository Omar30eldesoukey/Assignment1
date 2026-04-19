from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import UnifiedClipEmbedder
from src.indexer import load_chunk_store, load_index
from src.retriever import retrieve


def run_eval(index_dir: Path, benchmark_file: Path, top_k: int = 6) -> None:
    print("Loading CLIP embedder for evaluation...", flush=True)
    embedder = UnifiedClipEmbedder()
    print("Loading index and chunk store...", flush=True)
    index = load_index(index_dir)
    chunks = load_chunk_store(index_dir)

    print(f"Reading benchmark queries from: {benchmark_file}", flush=True)
    benchmarks = json.loads(benchmark_file.read_text(encoding="utf-8"))
    if not benchmarks:
        print("No benchmark queries found.")
        return

    hit_modality = 0
    for item in benchmarks:
        qid = item["id"]
        question = item["question"]
        expected_modality = item["expected_modality"]

        results = retrieve(question, embedder, index, chunks, top_k=top_k)
        modalities = [r.chunk.modality for r in results]
        match = expected_modality in modalities
        hit_modality += int(match)

        print(f"[{qid}] expected={expected_modality}, match={match}, retrieved={modalities}")

    total = len(benchmarks)
    score = hit_modality / total
    print(f"ModalityHit@{top_k}: {score:.2%} ({hit_modality}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simple retrieval evaluation")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark_queries.json"))
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    run_eval(index_dir=args.index_dir, benchmark_file=args.benchmark, top_k=args.top_k)
