import os
import traceback
import runpod
import torch
import base64
from io import BytesIO
import scipy.io.wavfile as wavf

from nemo.collections.tts.models import MagpieTTSModel

model_name = os.environ.get("MODEL_NAME", "nvidia/magpie_tts_multilingual_357m")
print(f"Cargando {model_name}...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- FIX 5: Loggear el traceback completo si el modelo falla ---
model = None
sample_rate = 22050  # fallback

try:
    model = MagpieTTSModel.from_pretrained(model_name)
    model.eval()
    if device == "cuda":
        model = model.to(device)

    # --- FIX 3: Obtener sample rate real del modelo en lugar de hardcodear ---
    sample_rate = int(model.cfg.get("sample_rate", 22050))

    print(f"Modelo cargado exitosamente. Sample rate: {sample_rate} Hz.")
except Exception as e:
    # FIX 5: Imprimir traceback completo para facilitar diagnóstico en producción
    print("ERROR CRÍTICO: El modelo no pudo ser cargado.")
    traceback.print_exc()

# Mapeo de voces disponibles
SPEAKER_MAP = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}

# --- FIX 6 (adelantado): Idiomas soportados explícitamente ---
SUPPORTED_LANGUAGES = {"en", "es", "de", "fr", "vi", "it", "zh", "hi", "ja"}

# --- FIX 7 (adelantado): Límite de caracteres para evitar OOM/timeouts ---
MAX_TEXT_LENGTH = 500


def handler(job):
    # FIX 5: El error de carga es ahora visible en logs; aquí solo lo reportamos al cliente
    if model is None:
        return {"error": "El modelo no pudo ser cargado. Revisa los logs del contenedor para más detalles."}

    job_input = job.get("input", {})

    text = job_input.get("text")
    language = job_input.get("language", "en")
    speaker = job_input.get("speaker", "Sofia")
    apply_TN = job_input.get("apply_TN", False)

    # Validaciones
    if not text:
        return {"error": "El parámetro 'text' es obligatorio."}

    if len(text) > MAX_TEXT_LENGTH:
        return {"error": f"El texto excede el límite de {MAX_TEXT_LENGTH} caracteres ({len(text)} recibidos)."}

    if language not in SUPPORTED_LANGUAGES:
        return {"error": f"Idioma '{language}' no soportado. Opciones: {sorted(SUPPORTED_LANGUAGES)}"}

    if speaker not in SPEAKER_MAP:
        return {"error": f"Voz '{speaker}' no válida. Opciones: {list(SPEAKER_MAP.keys())}"}

    speaker_idx = SPEAKER_MAP[speaker]

    try:
        with torch.no_grad():
            audio_tensor, audio_len = model.do_tts(
                text,
                language=language,
                apply_TN=apply_TN,
                speaker_index=speaker_idx,
            )

        audio_data = audio_tensor.squeeze().cpu().numpy()

        # FIX 4: Usar la longitud real del array en lugar del valor raw de audio_len,
        # que puede representar frames/tokens y no muestras.
        real_audio_length = len(audio_data)

        # FIX 3: Usar el sample_rate obtenido del modelo al inicio
        buffer = BytesIO()
        wavf.write(buffer, sample_rate, audio_data)
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "audio_length": real_audio_length,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Error durante la inferencia: {str(e)}"}


runpod.serverless.start({"handler": handler})