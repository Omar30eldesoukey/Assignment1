from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import fitz
import pdfplumber

from .schema import ChunkRecord


def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def extract_pdf_elements(pdf_path: Path, image_output_dir: Path) -> list[ChunkRecord]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkRecord] = []

    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber_doc:
        for page_idx, page in enumerate(doc, start=1):
            citation = f"{pdf_path.name}, page {page_idx}"
            page_text = page.get_text("text") or ""
            if page_text.strip():
                chunks.append(
                    ChunkRecord(
                        chunk_id=_stable_id(f"text::{pdf_path.name}::{page_idx}::{page_text[:500]}"),
                        source_file=pdf_path.name,
                        page=page_idx,
                        modality="text",
                        text=page_text,
                        citation=citation,
                        metadata={"source_path": str(pdf_path)},
                    )
                )

            try:
                plumber_page = plumber_doc.pages[page_idx - 1]
                tables = plumber_page.extract_tables() or []
            except Exception:  # noqa: BLE001
                tables = []
            for table_idx, table in enumerate(tables, start=1):
                lines = []
                for row in table:
                    safe_cells = [(cell or "").strip().replace("\n", " ") for cell in row]
                    lines.append(" | ".join(safe_cells))
                table_text = "\n".join(lines).strip()
                if not table_text:
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=_stable_id(
                            f"table::{pdf_path.name}::{page_idx}::{table_idx}::{table_text[:300]}"
                        ),
                        source_file=pdf_path.name,
                        page=page_idx,
                        modality="table",
                        text=table_text,
                        citation=f"{citation}, table {table_idx}",
                        metadata={"table_index": table_idx, "source_path": str(pdf_path)},
                    )
                )

            for image_idx, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                image_dict = doc.extract_image(xref)
                if not image_dict:
                    continue
                ext = image_dict.get("ext", "png")
                image_bytes = image_dict["image"]
                image_name = f"{pdf_path.stem}_p{page_idx}_img{image_idx}.{ext}"
                image_path = image_output_dir / image_name
                image_path.write_bytes(image_bytes)

                # Use nearby figure/chart lines from page text as lightweight metadata.
                caption_lines = [
                    ln.strip()
                    for ln in page_text.splitlines()
                    if ln.strip().lower().startswith(("figure", "chart", "graph"))
                ]
                caption_text = (
                    caption_lines[min(image_idx - 1, len(caption_lines) - 1)]
                    if caption_lines
                    else f"Image {image_idx} extracted from {citation}"
                )

                chunks.append(
                    ChunkRecord(
                        chunk_id=_stable_id(
                            f"image::{pdf_path.name}::{page_idx}::{image_idx}::{caption_text}"
                        ),
                        source_file=pdf_path.name,
                        page=page_idx,
                        modality="image",
                        text=caption_text,
                        citation=f"{citation}, image {image_idx}",
                        metadata={
                            "image_index": image_idx,
                            "image_path": str(image_path),
                            "source_path": str(pdf_path),
                        },
                    )
                )

    return chunks


def ingest_pdf_directory(pdf_dir: Path, image_output_dir: Path) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    for pdf_path in pdf_paths:
        all_chunks.extend(extract_pdf_elements(pdf_path, image_output_dir=image_output_dir))
    return all_chunks


def iter_pdf_paths(pdf_dir: Path) -> Iterable[Path]:
    yield from sorted(pdf_dir.rglob("*.pdf"))
