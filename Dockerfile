FROM --platform=linux/amd64 python:3.8-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FPL_LOGIN_HEADLESS=1 \
    PYTHONPATH=/app/api \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY requirements.txt requirements-pipeline.txt ./
# Original pins predate pip's strict resolver. TensorFlow 2.4.1 needs numpy~=1.19.2 and h5py~=2.10.0.
RUN sed -i \
        -e 's/^h5py==3.2.1/h5py==2.10.0/' \
        -e 's/^numpy==1.20.1/numpy==1.19.5/' \
        requirements.txt \
    && pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements-pipeline.txt \
    && python -m playwright install-deps chromium \
    && python -m playwright install chromium

# LightGBM (--learning-v2 --lightgbm), the same version as the evaluation venv. Its own
# layer so the TensorFlow / Playwright layer above stays cached.
RUN pip install --no-cache-dir lightgbm==4.6.0

# Schema SQL only — sqlite files are excluded by .dockerignore and filled at runtime.
COPY database ./database
COPY api ./api

WORKDIR /app/api

ENTRYPOINT ["python", "run_pipeline.py"]
