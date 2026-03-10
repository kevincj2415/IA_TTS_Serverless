# Imagen base fija — nunca usar :latest en producción
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Dependencias del sistema requeridas por NeMo y audio
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Forzar torch, torchvision y torchaudio compatibles antes de instalar NeMo
# Esto evita el RuntimeError: operator torchvision::nms does not exist y errores de ABI
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Dependencias base del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-requisitos para compilación de audio C++ en Python
RUN pip install --no-cache-dir Cython packaging

# NeMo en tag de release fijo — nunca usar @main en producción
RUN pip install --no-cache-dir \
    "nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@r2.2.0" \
    kaldialign

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]