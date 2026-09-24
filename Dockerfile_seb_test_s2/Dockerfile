# FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 \
#     python3-dev \
#     python3-pip \
#     git \
#     curl \
#     ca-certificates \
#     build-essential \
#     cmake \
#     libsox-dev \
#     libasound-dev \
#     portaudio19-dev \
#     libportaudio2 \
#     libportaudiocpp0 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# WORKDIR /app

# COPY pyproject.toml uv.lock README.md ./

# RUN uv python pin 3.12

# pour s'assurer que ça prenne bien le cache ! Car la première fois ça n'a pas prit le cache...Certainement du a cet environnement UV
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --extra cu129 --frozen --no-install-project

# COPY . .

# pour s'assurer que ça prenne bien le cache ! Car la première fois ça n'a pas prit le cache...Certainement du a cet environnement UV
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --extra cu129 --frozen

# RUN mkdir -p \
#     /app/input \
#     /app/output \
#     /app/checkpoints

# CMD ["bash"]

# LA VERSION CI DESSUS PREND TROP DE TEMPS A SE CONSTRUIRE... CAR INSTALLE ABSOLUMENT TOUT ET ON N'A PAS FORCEMENT BESOIN DE TOUT
# DE PLUS CONTIENT DES .VENV, DE UV , ETC ... C'EST VITE COMPLIQUE..
# DONC ON INSTALL LE MIMINUM, ON FAIT TOUNER L'INFERENCES SOUS RUNPOD ET ON AJOUTER LES DEPENDANCES PAS A PAS !

FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# TORCHINDUCTOR CACHE
# ============================================================

# local test with docker 
ENV TORCHINDUCTOR_CACHE_DIR=/app/torchinductor-cache


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3.12-dev \
    git \
    ca-certificates \
    ffmpeg \
    libsox-dev \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTORCH
# ============================================================

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129


# ============================================================
# FISH SPEECH DEPENDENCIES
# ============================================================

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    hydra-core \
    numpy \
    pyrootutils \
    soundfile \
    loguru \
    natsort \
    lightning \
    rich \
    descript-audiotools \
    descript-audio-codec \
    "transformers<=4.57.3" \
    loralib \
    click \
    runpod \
    huggingface_hub


# ============================================================
# WORKSPACE
# ============================================================

WORKDIR /app

RUN mkdir -p /app/torchinductor-cache

# ============================================================
# SERVERLESS HANDLER
# ============================================================

COPY Dockerfile_seb_test_s2/handler.py /app/handler.py


## je le mets spécialemement ici pour ne pas reconstruire toute l'image
# test in serveless to get in un cache
ENV TORCHINDUCTOR_CACHE_DIR=/runpod-volume/torchinductor-cache



# ============================================================
# DEFAULT COMMAND (DOCKER)
# ============================================================

# CMD ["bash"]

# ============================================================
# START SERVERLESS WORKER
# ============================================================

CMD ["python3", "/app/handler.py"]