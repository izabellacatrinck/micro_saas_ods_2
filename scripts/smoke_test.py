"""Smoke test against the live HF Space URL.

Usage:
    HF_SPACE_URL=https://username-rag-pt-backend.hf.space \
        .venv/Scripts/python.exe scripts/smoke_test.py

The first call may take ~30s if the Space is cold. This script uses a 90s timeout.
"""
import os
import sys

import httpx


def main() -> None:
    base_url = os.environ.get("HF_SPACE_URL", "").rstrip("/")
    if not base_url:
        print("ERROR: HF_SPACE_URL must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Smoke testing: {base_url}")

    # 1. Health check
    print("  GET /health ...", end=" ", flush=True)
    r = httpx.get(f"{base_url}/health", timeout=90)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("models_loaded") is True, f"/health: models_loaded not True: {body}"
    print(f"OK (chroma_count={body.get('chroma_count')})")

    # 2. Ask endpoint
    print("  POST /ask ...", end=" ", flush=True)
    r = httpx.post(
        f"{base_url}/ask",
        json={"question": "Como fazer merge entre dois DataFrames no pandas?"},
        timeout=90,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("answer"), f"/ask: empty answer: {body}"
    assert isinstance(body.get("citations"), list), f"/ask: citations not a list: {body}"
    print(f"OK")
    print(f"    Answer preview: {body['answer'][:120]}...")
    print(f"    Citations: {len(body['citations'])} returned")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
