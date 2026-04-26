FROM python:3.12-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps before copying code (layer cache)
COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/chroma_db/ ./data/chroma_db/

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
