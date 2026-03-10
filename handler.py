import os
import runpod
import torch
import base64
from io import BytesIO
import scipy.io.wavfile as wavf

# Importar el modelo MagpieTTS desde NeMo
from nemo.collections.tts.models import MagpieTTSModel

model_name = os.environ.get("MODEL_NAME", "nvidia/magpie_tts_multilingual_357m")
print(f"Cargando {model_name}...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargamos el modelo
try:
    model = MagpieTTSModel.from_pretrained(model_name)
    model.eval()
    if device == "cuda":
        model = model.to(device)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error cargando el modelo: {str(e)}")
    model = None

# Mapeo de voces públicas e internas disponibles
speaker_map = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4
}

def handler(job):
    if model is None:
        return {"error": "El modelo no pudo ser cargado correctamente."}

    job_input = job.get("input", {})
    
    # Parámetros requeridos
    text = job_input.get("text")
    language = job_input.get("language", "en")
    
    # Opcionales
    speaker = job_input.get("speaker", "Sofia")
    apply_TN = job_input.get("apply_TN", False)

    if not text:
        return {"error": "El parámetro 'text' es obligatorio."}

    # Validar el speaker
    if speaker not in speaker_map:
        return {"error": f"Voz '{speaker}' no válida. Opciones: {list(speaker_map.keys())}"}
    
    speaker_idx = speaker_map[speaker]

    try:
        # Ejecutar inferencia
        with torch.no_grad():
            audio_tensor, audio_len = model.do_tts(
                text, 
                language=language, 
                apply_TN=apply_TN, 
                speaker_index=speaker_idx
            )
            
        # Convertir a numpy, mover al CPU y asegurar una sola dimensión para wav
        audio_data = audio_tensor.squeeze().cpu().numpy()
        
        # El sample rate típicamente es 22050 para estos modelos, ajustable si cambia el codec
        sample_rate = 22050 

        # Escribir el audio en un buffer en memoria
        buffer = BytesIO()
        wavf.write(buffer, sample_rate, audio_data)
        buffer.seek(0)
        
        # Codificar a base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        return {"audio_base64": audio_base64, "audio_length": int(audio_len.item()) if hasattr(audio_len, 'item') else audio_len}
        
    except Exception as e:
        return {"error": f"Error durante la inferencia: {str(e)}"}

runpod.serverless.start({"handler": handler})
