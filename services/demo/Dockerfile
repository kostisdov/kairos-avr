# kairos-demo: Streamlit one-patient demonstration (behind Entra ID built-in authentication).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 KAIROS_REPO_ROOT=/app
WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY src ./src
COPY config ./config
COPY data/reference ./data/reference
COPY services/demo ./services/demo
RUN pip install --no-cache-dir --no-deps .
ARG GIT_SHA=nogit
ENV KAIROS_GIT_SHA=$GIT_SHA
EXPOSE 8501
CMD ["streamlit", "run", "services/demo/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
