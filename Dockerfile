
FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openslide-tools \
        libopenslide0 \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY db /app/db
COPY README.md /app/README.md

RUN mkdir -p /app/data /app/experiments /app/mlruns

ENV PYTHONPATH=/app

CMD ["bash", "-c", "sleep infinity"]
