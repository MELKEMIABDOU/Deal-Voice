FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/huggingface

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra pdf --extra audio --extra local-ai

COPY src ./src
RUN uv sync --frozen --no-dev \
    --extra pdf --extra audio --extra local-ai

RUN groupadd --gid 10001 dealvoice \
    && useradd --uid 10001 --gid dealvoice --create-home dealvoice \
    && mkdir -p /app/data/raw /app/data/processed \
        /data/parent_docstore /models/huggingface \
    && chown -R dealvoice:dealvoice /app/data /data /models

USER dealvoice
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "deal_voice.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
