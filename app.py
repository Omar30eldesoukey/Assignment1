from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.answer_cache import build_cache_key, load_cached_answer, save_cached_answer
from src.embeddings import UnifiedClipEmbedder
from src.indexer import load_chunk_store, load_index
from src.qa import answer_with_context
from src.retriever import retrieve


def _find_open_port(start_port: int = 8501, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free local port found for Streamlit.")


if __name__ == "__main__" and get_script_run_ctx() is None and os.environ.get("APP_PY_STREAMLIT_LAUNCHED") != "1":
    launch_port = _find_open_port(8501)
    print(f"Starting Streamlit app at http://localhost:{launch_port}", flush=True)
    launch_env = os.environ.copy()
    launch_env["APP_PY_STREAMLIT_LAUNCHED"] = "1"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                "--server.port",
                str(launch_port),
                "--server.headless",
                "true",
            ],
            env=launch_env,
        )
    )

st.set_page_config(page_title="Multi-Modal RAG for PDF Reports", layout="wide")

st.title("Multi-Modal RAG QA (Text + Tables + Images)")
st.caption("Grounded answers with source citations from policy/financial PDFs")

# NOTE: Hidden from UI per user request.
DEFAULT_OPENAI_API_KEY = "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"

INDEX_DIR = Path("data/index")
RAW_DIR = Path("data/raw")
CACHE_PATH = Path("data/cache/qa_cache.jsonl")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieval", min_value=3, max_value=15, value=6, step=1)
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
    use_answer_cache = st.checkbox("Reuse cached answers for repeated questions", value=True)
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
    resolved_api_key = DEFAULT_OPENAI_API_KEY

    with st.spinner("Retrieving evidence..."):
        results = retrieve(query=question, embedder=embedder, index=index, chunks=chunks, top_k=top_k)

    cache_key = build_cache_key(
        question=question,
        model=model_name,
        top_k=top_k,
        used_api=bool(resolved_api_key),
    )
    cached = load_cached_answer(CACHE_PATH, cache_key) if use_answer_cache else None

    if cached:
        answer = cached.answer
        st.info("Loaded cached answer for this question.")
    else:
        with st.spinner("Generating answer..."):
            answer = answer_with_context(
                question=question,
                retrieved=results,
                model=model_name,
                api_key=resolved_api_key,
            )
        citations = [item.chunk.citation for item in results]
        save_cached_answer(
            cache_path=CACHE_PATH,
            key=cache_key,
            question=question,
            answer=answer,
            model=model_name,
            used_api=bool(resolved_api_key),
            citations=citations,
        )

    st.subheader("Answer")
    st.write(answer)

    image_results = [item for item in results if item.chunk.modality == "image"]
    if image_results:
        st.subheader("Retrieved Images")
        for img_item in image_results:
            img_path = img_item.chunk.metadata.get("image_path") if img_item.chunk.metadata else None
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=img_item.chunk.citation, width="stretch")

    st.subheader("Top Retrieved Evidence")
    for rank, item in enumerate(results, start=1):
        with st.expander(f"#{rank} | score={item.score:.3f} | {item.chunk.modality} | {item.chunk.citation}"):
            st.write(item.chunk.text[:1200])
            img_path = item.chunk.metadata.get("image_path") if item.chunk.metadata else None
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=item.chunk.citation, width="stretch")
