FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    libsox-dev \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv python pin 3.12

RUN uv sync --extra cu129 --frozen --no-install-project

COPY . .

RUN uv sync --extra cu129 --frozen

RUN mkdir -p \
    /app/input \
    /app/output \
    /app/checkpoints

ENTRYPOINT ["/bin/bash"]