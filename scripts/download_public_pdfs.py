from __future__ import annotations

import argparse
from pathlib import Path

import requests


def download_pdfs(url_file: Path, output_dir: Path, timeout: int = 60) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = [ln.strip() for ln in url_file.read_text(encoding="utf-8").splitlines()]
    urls = [u for u in urls if u and not u.startswith("#")]

    if not urls:
        print("No URLs found in URL file.")
        return

    for idx, url in enumerate(urls, start=1):
        name = url.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0] or f"document_{idx}.pdf"
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        out_path = output_dir / name

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            ctype = response.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower() and not out_path.suffix.lower() == ".pdf":
                print(f"Skipping non-PDF URL: {url}")
                continue
            out_path.write_bytes(response.content)
            print(f"[{idx}/{len(urls)}] Downloaded: {out_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(urls)}] Failed: {url} -> {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public PDF dataset from URL list")
    parser.add_argument("--url-file", type=Path, default=Path("data/imf_urls.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    download_pdfs(url_file=args.url_file, output_dir=args.output_dir)
