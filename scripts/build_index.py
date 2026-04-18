from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_records
from src.embeddings import UnifiedClipEmbedder
from src.indexer import FaissChunkIndex, save_chunk_store, save_index
from src.ingestion import extract_pdf_elements, iter_pdf_paths


def build_index(raw_pdf_dir: Path, image_dir: Path, index_dir: Path, max_pdfs: int | None = None) -> None:
    print(f"Ingesting PDFs from: {raw_pdf_dir}")
    pdf_paths = list(iter_pdf_paths(raw_pdf_dir))
    if not pdf_paths:
        print("No PDF files found.")
        print("Place your PDFs in data/raw/ and rerun this command.")
        return

    if max_pdfs is not None and max_pdfs > 0:
        pdf_paths = pdf_paths[:max_pdfs]

    print(f"Found {len(pdf_paths)} PDF files.")
    raw_chunks = []
    failed_pdfs = 0
    for pdf_path in pdf_paths:
        try:
            raw_chunks.extend(extract_pdf_elements(pdf_path, image_output_dir=image_dir))
        except Exception as exc:  # noqa: BLE001
            failed_pdfs += 1
            print(f"Failed to extract {pdf_path.name}: {exc}")

    chunked = chunk_records(raw_chunks)
    print(f"Total chunks: {len(chunked)}")
    if failed_pdfs:
        print(f"Skipped PDFs due to extraction errors: {failed_pdfs}")

    if not chunked:
        print("No chunks were extracted from the PDFs.")
        print("This usually means PDFs are image-only scans or extraction failed.")
        return

    embedder = UnifiedClipEmbedder()

    text_like = [c for c in chunked if c.modality in {"text", "table", "image"}]
    if not text_like:
        print("No indexable chunks found after filtering modalities.")
        return

    text_embeddings = embedder.embed_texts([c.text for c in text_like])

    if len(text_like) != text_embeddings.shape[0]:
        raise RuntimeError("Embedding count mismatch.")

    index = FaissChunkIndex(dimension=text_embeddings.shape[1])
    index.add(text_embeddings)

    save_index(index, index_dir)
    save_chunk_store(text_like, index_dir)

    print(f"Saved FAISS index to: {index_dir}")
    print(f"Indexed chunks: {len(text_like)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build multimodal RAG index from PDFs")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--max-pdfs", type=int, default=None)
    args = parser.parse_args()

    build_index(
        raw_pdf_dir=args.pdf_dir,
        image_dir=args.image_dir,
        index_dir=args.index_dir,
        max_pdfs=args.max_pdfs,
    )
