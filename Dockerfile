
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /srv

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Install dependencies from the lockfile
RUN uv sync --frozen --no-dev --no-install-project

# Copy application files
COPY app ./app
COPY examples ./examples

# Ensure installed virtual environment is used
ENV PATH="/srv/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
