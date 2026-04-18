from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.embeddings import UnifiedClipEmbedder
from src.indexer import load_chunk_store, load_index
from src.qa import answer_with_context
from src.retriever import retrieve

st.set_page_config(page_title="Multi-Modal RAG for PDF Reports", layout="wide")

st.title("Multi-Modal RAG QA (Text + Tables + Images)")
st.caption("Grounded answers with source citations from policy/financial PDFs")

INDEX_DIR = Path("data/index")
RAW_DIR = Path("data/raw")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieval", min_value=3, max_value=15, value=6, step=1)
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
    st.markdown("### Build Index")
    st.write("Run this in terminal:")
    st.code("python scripts/build_index.py")

if not (INDEX_DIR / "index.faiss").exists():
    st.warning("Index not found.")
    st.info("1) Put PDFs in data/raw/\n2) Run: python scripts/build_index.py\n3) Reopen this app")
    st.stop()

@st.cache_resource
def get_runtime_objects() -> tuple[UnifiedClipEmbedder, object, list]:
    embedder = UnifiedClipEmbedder()
    index = load_index(INDEX_DIR)
    chunks = load_chunk_store(INDEX_DIR)
    return embedder, index, chunks

embedder, index, chunks = get_runtime_objects()

if not chunks:
    st.warning("Index exists but chunk store is empty.")
    st.info("Rebuild index after verifying PDFs in data/raw/: python scripts/build_index.py")
    st.stop()

question = st.text_input("Ask a question about your reports")

if question:
    with st.spinner("Retrieving evidence..."):
        results = retrieve(question=question, embedder=embedder, index=index, chunks=chunks, top_k=top_k)

    with st.spinner("Generating answer..."):
        answer = answer_with_context(question=question, retrieved=results, model=model_name)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Top Retrieved Evidence")
    for rank, item in enumerate(results, start=1):
        with st.expander(f"#{rank} | score={item.score:.3f} | {item.chunk.modality} | {item.chunk.citation}"):
            st.write(item.chunk.text[:1200])
            img_path = item.chunk.metadata.get("image_path") if item.chunk.metadata else None
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=item.chunk.citation, use_column_width=True)
