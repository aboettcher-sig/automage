# syntax=docker/dockerfile:1
FROM python:3.12-slim AS package
WORKDIR /source
COPY pyproject.toml README.md MANIFEST.in ./
COPY automage ./automage
RUN python -m pip wheel --no-cache-dir --no-deps . --wheel-dir /wheels

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
RUN --mount=from=package,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir /wheels/*.whl google-cloud-storage==3.16.0
COPY deploy/cloud-run/worker.py /worker.py
ENTRYPOINT ["python", "/worker.py"]
