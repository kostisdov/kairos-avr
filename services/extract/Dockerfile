# kairos-extract: FastAPI service (rules first, hosted model second, flag-gated).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 KAIROS_REPO_ROOT=/app
WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY src ./src
COPY config ./config
COPY data/reference ./data/reference
COPY services/extract ./services/extract
RUN pip install --no-cache-dir --no-deps .
ARG GIT_SHA=nogit
ENV KAIROS_GIT_SHA=$GIT_SHA
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "services/extract"]
