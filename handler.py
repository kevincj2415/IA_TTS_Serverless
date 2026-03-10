import os
import traceback
import runpod
import torch
import base64
from io import BytesIO
import soundfile as sf
import numpy as np

from qwen_tts import Qwen3TTSModel

# Default to Qwen3-TTS VoiceDesign
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
print(f"Cargando {model_name}...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = None

try:
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa", # Usando sdpa en lugar de flash_attention_2 para máxima compatibilidad out-of-the-box sin aislar entorno de build
    )
    print("Modelo cargado exitosamente.")
except Exception as e:
    print("ERROR CRÍTICO: El modelo no pudo ser cargado.")
    traceback.print_exc()

# Qwen3-TTS Idiomas Soportados
SUPPORTED_LANGUAGES = {"Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"}
MAX_TEXT_LENGTH = 1000

def handler(job):
    if model is None:
        return {"error": "El modelo no pudo ser cargado. Revisa los logs del contenedor para más detalles."}

    job_input = job.get("input", {})

    text = job_input.get("text")
    language = job_input.get("language", "English")
    # For VoiceDesign, 'instruct' represents the natural language description of the voice
    instruct = job_input.get("instruct", "A person speaking clearly.")

    if not text:
        return {"error": "El parámetro 'text' es obligatorio."}

    if len(text) > MAX_TEXT_LENGTH:
        return {"error": f"El texto excede el límite de {MAX_TEXT_LENGTH} caracteres ({len(text)} recibidos)."}

    # Asegurar formato válido de idioma
    lang_formatted = str(language).capitalize()
    if lang_formatted not in SUPPORTED_LANGUAGES:
        return {"error": f"Idioma '{language}' no soportado. Opciones: {sorted(SUPPORTED_LANGUAGES)}"}

    try:
        with torch.no_grad():
            wav_list, sr = model.generate_voice_design(
                text=text,
                language=lang_formatted,
                instruct=instruct,
            )

        if not wav_list or len(wav_list) == 0:
            return {"error": "El modelo no devolvió ningún audio."}

        audio_data = wav_list[0]
        
        # Guardar audio en memoria como WAV
        buffer = BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)

        # Convertir a Base64 para enviarlo al cliente
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        real_audio_length = len(audio_data)

        return {
            "audio_base64": audio_base64,
            "audio_length": real_audio_length,
            "sample_rate": sr,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Error durante la inferencia: {str(e)}"}

runpod.serverless.start({"handler": handler})