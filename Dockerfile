FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Evitar prompts interactivos durante la instalación (ej. tzdata)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Instalar dependencias del sistema requeridas por NeMo y libsndfile para audios
RUN apt-get update && apt-get install -y git libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# Instalar dependencias base y optimizaciones
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar NeMo toolkit específicamente con soporte TTS y kaldialign
RUN pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[tts]" kaldialign --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
