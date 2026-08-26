# syntax=docker/dockerfile:1
# 3.11 is required: the ensemble pickles were saved with scikit-learn 1.2.0,
# which has no cp312 wheels. See pyproject.toml for the full rationale.
ARG PYTHON_VERSION=3.11

# Stage 1: Build wheel using uv
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv build --wheel

# Stage 2: Minimal runtime image
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app
COPY --from=builder /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl

# Model artifacts are NOT baked into the image — they are declared as `assets`
# in schema/model.json and downloaded at run time by the runner to their paths
# under the WORKDIR (./artifacts/…). See src/model/artifacts/README.md.
RUN mkdir -p /data/output
ENTRYPOINT ["model-run"]
