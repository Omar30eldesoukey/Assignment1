from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedAnswer:
    key: str
    question: str
    answer: str
    model: str
    used_api: bool
    citations: list[str]


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def build_cache_key(question: str, model: str, top_k: int, used_api: bool) -> str:
    normalized = _normalize_question(question)
    raw = f"q={normalized}|model={model}|top_k={top_k}|used_api={used_api}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_answer(cache_path: Path, key: str) -> CachedAnswer | None:
    if not cache_path.exists():
        return None

    latest: dict[str, Any] | None = None
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key") == key:
                latest = row

    if not latest:
        return None

    return CachedAnswer(
        key=str(latest.get("key", "")),
        question=str(latest.get("question", "")),
        answer=str(latest.get("answer", "")),
        model=str(latest.get("model", "")),
        used_api=bool(latest.get("used_api", False)),
        citations=[str(x) for x in latest.get("citations", [])],
    )


def save_cached_answer(
    cache_path: Path,
    key: str,
    question: str,
    answer: str,
    model: str,
    used_api: bool,
    citations: list[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "question": question,
        "answer": answer,
        "model": model,
        "used_api": used_api,
        "citations": citations,
    }
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
