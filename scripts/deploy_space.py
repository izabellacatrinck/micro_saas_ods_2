"""Upload backend to Hugging Face Space via huggingface_hub.

Usage:
    HF_SPACE_ID=username/rag-pt-backend HF_TOKEN=hf_... \
        .venv/Scripts/python.exe scripts/deploy_space.py

Environment variables:
    HF_SPACE_ID   — Space repo ID, e.g. "myuser/rag-pt-backend"
    HF_TOKEN      — HF write-access token (never commit this)
"""
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    space_id = os.environ.get("HF_SPACE_ID")
    token = os.environ.get("HF_TOKEN")

    if not space_id or not token:
        print("ERROR: HF_SPACE_ID and HF_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Deploying to Space: {space_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=".",
        repo_id=space_id,
        repo_type="space",
        token=token,
        allow_patterns=["backend/**", "src/**", "data/chroma_db/**"],
        ignore_patterns=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".venv/**",
            "**/.git/**",
        ],
    )
    print("Deploy concluído. O Space vai fazer rebuild automaticamente.")
    print(f"Monitor build logs at: https://huggingface.co/spaces/{space_id}/logs")


if __name__ == "__main__":
    main()
