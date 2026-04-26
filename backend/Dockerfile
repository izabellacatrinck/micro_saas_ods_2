FROM python:3.12-slim

# Install uv for fast dependency resolution
# Pin to a specific tag for reproducibility once the target version is known;
# for now, `latest` is used to avoid a non-existent tag breaking the build.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install torch CPU-only first (separate index — uv does not support inline --index-url in requirements.txt)
RUN uv pip install --system --no-cache torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps (torch already installed above)
COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
# data/chroma_db/ must be pre-populated locally and uploaded by scripts/deploy_space.py
# before this Dockerfile is built on HF Space. See docs/superpowers/SETUP_HF_SPACE.md.
COPY data/chroma_db/ ./data/chroma_db/

# Run as non-root user for security
RUN useradd --create-home --no-log-init appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
