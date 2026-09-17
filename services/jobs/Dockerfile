# kairos-jobs: batch CLI (scenarios, train, evaluate, extraction-eval, upload-reference, migrate).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 KAIROS_REPO_ROOT=/app MPLBACKEND=Agg
WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY src ./src
COPY config ./config
COPY data/reference ./data/reference
COPY data/derived/aggregates ./data/derived/aggregates
COPY tests/fixtures/synthetic_notes ./tests/fixtures/synthetic_notes
COPY services/jobs ./services/jobs
RUN pip install --no-cache-dir --no-deps .
ARG GIT_SHA=nogit
ENV KAIROS_GIT_SHA=$GIT_SHA
ENTRYPOINT ["python", "services/jobs/cli.py"]
CMD ["--help"]
