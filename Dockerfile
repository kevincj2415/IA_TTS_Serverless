FROM pytorch/pytorch:latest

WORKDIR /app

# Evitar prompts interactivos durante la instalación (ej. tzdata)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Instalar dependencias del sistema requeridas por NeMo, compilación y audios
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias base y optimizaciones
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias pre-requisito para compilación de audio C++ en Python
RUN pip install --no-cache-dir Cython packaging

# Instalar NeMo toolkit específicamente con soporte TTS y kaldialign
RUN pip install "nemo_toolkit[tts] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@main" kaldialign --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
