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

# Qwen3-TTS y generador de dependencias extra si se requieren
RUN pip install --no-cache-dir qwen-tts

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]