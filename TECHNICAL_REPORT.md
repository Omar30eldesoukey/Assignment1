# Technical Report: Multi-Modal RAG for PDF Reports

## 1. Project Overview
This project implements a simple multi-modal Retrieval-Augmented Generation (RAG) system for financial/policy PDF reports. The system extracts content from PDFs, indexes it in a unified vector space, and answers user questions in a Streamlit app with citations.

Main supported modalities:
- Text
- Tables
- Images (with lightweight caption metadata)

Goal: return grounded answers and evidence snippets from source documents instead of unsupported free-form responses.

## 2. System Architecture
The pipeline is organized in four stages:

1. Ingestion
- Input PDFs are read from `data/raw/`.
- `src/ingestion.py` extracts:
  - page text (PyMuPDF)
  - tables (pdfplumber)
  - images (PyMuPDF image extraction)
- Each extracted item is converted into a `ChunkRecord` with source metadata and citation text.

2. Chunking and Normalization
- `src/chunking.py` splits long text into overlapping chunks (`max_chars=1200`, overlap `150`).
- Table and image records are kept as single chunks.

3. Embedding and Indexing
- `src/embeddings.py` uses CLIP (`openai/clip-vit-base-patch32`) to embed content.
- In this implementation, text, table text, and image caption text are embedded through the text encoder into one shared embedding space.
- `src/indexer.py` builds a FAISS `IndexFlatIP` index for similarity search and stores:
  - vectors in `data/index/index.faiss`
  - chunk metadata in `data/index/chunks.jsonl`

4. Retrieval and QA
- `src/retriever.py` embeds a query, retrieves top candidates, filters low-information chunks, and applies light modality-aware reranking (for image-intent queries).
- `src/qa.py` generates final answers using an LLM when API key is available; otherwise it returns extractive evidence snippets.
- `app.py` provides the Streamlit UI and shows citations plus retrieved evidence.

## 3. Key Design Choices
1. Unified embedding model (CLIP)
- A single embedding model simplifies cross-modal retrieval and reduces architecture complexity.
- Trade-off: table-specific semantics may be weaker than with specialized table-aware models.

2. Simple chunking strategy
- Character-based chunking is easy to implement and robust.
- Trade-off: it does not always preserve sentence/section boundaries.

3. Lightweight image handling
- Images are represented by nearby figure/chart text when available.
- Trade-off: true visual reasoning is limited because extracted image pixels are not used at query time in this baseline.

4. Fast exact vector search (FAISS IndexFlatIP)
- Good baseline behavior and simple setup.
- Trade-off: exact search can be slower than approximate indexes for very large datasets.

5. Evidence-first fallback behavior
- If LLM key is missing/fails, the app still returns grounded snippets with citations.
- This improves reliability for demo and grading scenarios.

## 4. Benchmark Summary
Evaluation command:
- `python scripts/run_evaluation.py --index-dir data/index --benchmark evaluation/benchmark_queries.json --top-k 6`

Benchmark file:
- `evaluation/benchmark_queries.json`
- 3 queries, each with expected dominant modality (`text`, `table`, `image`).

Metric:
- `ModalityHit@K`: checks whether expected modality appears in top-K retrieved chunks.

Observed run (Top-K = 6):
- `q_text_1`: match=True
- `q_table_1`: match=False
- `q_chart_1`: match=True
- Final: **ModalityHit@6 = 66.67% (2/3)**

Interpretation:
- Text and image-intent retrieval worked in this small benchmark.
- Table retrieval is currently the weakest area.

## 5. Key Observations and Limitations
1. Strong points
- End-to-end flow is complete: ingestion -> indexing -> retrieval -> cited answers.
- System is practical and easy to run with `scripts/run_all.ps1`.
- UI exposes retrieved evidence for transparency.

2. Weak points
- Very small benchmark size (3 queries) limits confidence in general performance.
- Table understanding is weaker than text/image in current setup.
- Some image answers rely on caption-like text, not deep visual interpretation.

3. Risks
- PDF extraction quality varies across document formats.
- Retrieval quality depends heavily on chunk quality and extracted text cleanliness.

## 6. Suggested Next Improvements
- Expand benchmark to at least 30-50 queries with per-modality balance.
- Improve table representation (row/column-aware serialization or table-specific embeddings).
- Add reranking stage (cross-encoder or modality-aware reranker).
- Track additional metrics: Recall@K, MRR, and answer faithfulness.

## 7. Conclusion
This MVP demonstrates a clear and usable multi-modal RAG architecture with citation-backed QA. The benchmark result (66.67% ModalityHit@6 on current small set) shows promising behavior for text/image questions, with table retrieval as the main improvement target.
