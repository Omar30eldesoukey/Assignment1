# Multi-Modal RAG (Streamlit)

This project is an MVP multi-modal RAG pipeline for PDF reports (financial/policy style docs).
It supports:
- Text extraction
- Table extraction
- Image extraction with lightweight chart/figure caption metadata
- Unified embedding space via CLIP
- FAISS vector retrieval
- Streamlit QA app with citation-backed responses
- Basic retrieval evaluation suite

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Add PDF data

Option A: place PDFs manually in `data/raw/`.

Option B: put direct PDF links (one per line) into `data/imf_urls.txt`, then run:

```bash
python scripts/download_public_pdfs.py --url-file data/imf_urls.txt --output-dir data/raw
```

## 3) Build index

```bash
python scripts/build_index.py --pdf-dir data/raw --image-dir data/processed/images --index-dir data/index
```

Quick test run (recommended first):

```bash
python scripts/build_index.py --pdf-dir data/raw --max-pdfs 50
```

## 4) Run app

```bash
streamlit run app.py
```

Ask questions in the UI and review citations under "Top Retrieved Evidence".

## 5) Evaluate retrieval

```bash
python scripts/run_evaluation.py --index-dir data/index --benchmark evaluation/benchmark_queries.json --top-k 6
```

## Notes
- For fully generative answers, set environment variable `OPENAI_API_KEY`.
- Without API key, app falls back to extractive evidence snippets with citations.
- You can adapt `evaluation/benchmark_queries.json` to your own benchmark questions.
