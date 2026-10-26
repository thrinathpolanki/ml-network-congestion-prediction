# One image, two roles (API or dashboard): the command is chosen in docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel drags in several GB of CUDA libraries we do not use).
# Override with --build-arg TORCH_INDEX=https://pypi.org/simple if you want the default build.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} "torch>=2.1"

# Dependencies before code, so editing source does not reinstall everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY models ./models
COPY reports ./reports
COPY data/sample ./data/sample

# Run as an unprivileged user; /app/logs is the only place the app writes (log file, alert state).
RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs && chown -R app:app /app/logs
USER app

EXPOSE 8000 8501

# Default role: the API. Single worker on purpose: latest results and alert state live in process memory.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"
