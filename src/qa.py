from __future__ import annotations

import os
import re
from typing import Sequence

from openai import OpenAI

from .retriever import RetrievalResult, group_context


SYSTEM_PROMPT = """You are a precise policy and financial analyst assistant.
Answer only using the provided context.
If evidence is insufficient, say so clearly.
Always add citations in brackets using the provided Source labels.
"""


def _build_extractive_fallback(
    retrieved: Sequence[RetrievalResult],
    *,
    reason: str | None = None,
) -> str:
    top = list(retrieved)[:3]
    if reason:
        lines = [f"LLM generation unavailable ({reason}). Showing evidence-grounded extractive answer:"]
    else:
        lines = ["No LLM API key found. Showing evidence-grounded extractive answer:"]
    for item in top:
        snippet = _best_snippet(item.chunk.text)
        if not snippet:
            snippet = f"{item.chunk.modality.title()} evidence found; see source details"
        lines.append(f"- {snippet} [{item.chunk.citation}]")
    if not reason:
        lines.append("Set OPENAI_API_KEY to enable generative answers.")
    return "\n".join(lines)


def _best_snippet(text: str, max_chars: int = 350) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""

    # Ignore table separator placeholders and symbol-only fragments.
    if compact.replace("|", "").strip() == "":
        return ""

    if sum(ch.isalnum() for ch in compact) < 18:
        return ""

    sentence_like = re.split(r"(?<=[.!?])\s+", compact)
    for candidate in sentence_like:
        cleaned = candidate.strip(" |\t")
        if len(cleaned) >= 40 and sum(ch.isalnum() for ch in cleaned) >= 24:
            return cleaned[:max_chars]

    return compact[:max_chars]


def answer_with_context(
    question: str,
    retrieved: Sequence[RetrievalResult],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> str:
    if not retrieved:
        return "I could not find relevant evidence in the indexed documents."

    context = group_context(list(retrieved))
    resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()

    if not resolved_api_key:
        return _build_extractive_fallback(retrieved)

    use_openrouter = resolved_api_key.startswith("sk-or-v1-")
    model_to_use = model
    if use_openrouter:
        client = OpenAI(api_key=resolved_api_key, base_url="https://openrouter.ai/api/v1")
        if "/" not in model_to_use:
            model_to_use = f"openai/{model_to_use}"
    else:
        client = OpenAI(api_key=resolved_api_key)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Write a concise, direct answer with source citations."
    )

    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or "No answer generated."
    except Exception as exc:  # noqa: BLE001
        return _build_extractive_fallback(retrieved, reason=f"{type(exc).__name__}")
