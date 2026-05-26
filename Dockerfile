FROM python:3.10.19-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY README.md .
COPY src/ src/
COPY tests/ tests/
RUN uv sync

CMD ["uv", "run", "pytest", "-v"]
