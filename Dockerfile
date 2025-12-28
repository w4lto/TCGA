FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências de sistema (openslide etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openslide-tools \
        libopenslide0 \
        git \
        curl \
        ca-certificates \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Cria venv isolado para não brigar com pacotes do sistema (distutils/apt)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copia requirements primeiro para cache
COPY requirements.txt /app/requirements.txt

# Instala deps no venv (não mexe no site-packages do sistema)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Copia o código
COPY src /app/src
COPY db /app/db
COPY README.md /app/README.md
COPY config.yaml /app/config.yaml
COPY compose.yaml /app/compose.yaml

# Diretórios úteis
RUN mkdir -p /app/data /app/experiments /app/mlruns

ENV PYTHONPATH=/app

CMD ["bash", "-c", "sleep infinity"]
