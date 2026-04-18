from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI

from .retriever import RetrievalResult, group_context


SYSTEM_PROMPT = """You are a precise policy and financial analyst assistant.
Answer only using the provided context.
If evidence is insufficient, say so clearly.
Always add citations in brackets using the provided Source labels.
"""


def answer_with_context(question: str, retrieved: Sequence[RetrievalResult], model: str = "gpt-4o-mini") -> str:
    if not retrieved:
        return "I could not find relevant evidence in the indexed documents."

    context = group_context(list(retrieved))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        # Fallback mode keeps the app usable without a paid API key.
        top = list(retrieved)[:3]
        lines = ["No LLM API key found. Showing evidence-grounded extractive answer:"]
        for item in top:
            snippet = item.chunk.text.strip().replace("\n", " ")[:350]
            lines.append(f"- {snippet} [{item.chunk.citation}]")
        lines.append("Set OPENAI_API_KEY to enable generative answers.")
        return "\n".join(lines)

    client = OpenAI(api_key=api_key)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Write a concise, direct answer with source citations."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or "No answer generated."
