from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Client for PDF2ZH FastAPI server")
    p.add_argument("--server", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--pdf", required=True, help="Path to local PDF file")
    p.add_argument("--src", required=True, help="Source language code, e.g. en, auto")
    p.add_argument("--tgt", required=True, help="Target language code, e.g. zh")
    p.add_argument("--out", required=True, help="Output directory on server")
    p.add_argument("--pages", default=None, help="Optional page ranges, e.g. 1-3,5")
    p.add_argument("--chunk-pages", type=int, default=None, help="Override pages per chunk for large PDFs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    payload = {
        "pdf_path": str(pdf_path),
        "source_lang": args.src,
        "target_lang": args.tgt,
        "output_dir": str(Path(args.out)),
        "pages": args.pages,
        "chunk_pages": args.chunk_pages,
    }
    url = args.server.rstrip("/") + "/translate"

    # Long-running translation: set a generous timeout (None = wait forever)
    resp = requests.post(url, json=payload, timeout=None)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        print(f"Request failed: {resp.status_code} {detail}", file=sys.stderr)
        return 1

    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

