# MagpieTTS Multilingual Serverless API

Este proyecto proporciona una API serverless para **RunPod** que permite generar voz a partir de texto (Text-to-Speech) de alta fidelidad, utilizando el modelo **MagpieTTS Multilingual 357M** de NVIDIA a través de su framework NeMo.

## Características

* **Multilenguaje**: Soporte para Español (es), Inglés (en), Alemán (de), Francés (fr), Vietnamita (vi), Italiano (it), Mandarín (zh), Hindi (hi) y Japonés (ja).
* **Voces Disponibles**: Incluye 5 opciones de voces expresivas: `Sofia`, `Aria`, `Jason`, `Leo`, y `John`.
* **Aceleración**: Ejecución nativa por GPU de los modelos Transformer Encoder/Decoder con códecs neuronales (NanoCodec).
* **Despliegue Serverless**: Arquitectura compatible al 100% con endpoints `runpod.serverless`.
* **Respuesta Base64**: Devuelve directamente un archivo `.wav` codificado en cadena Base64 listo para ser reproducido o descargado por un cliente frontend.

## Requisitos Previos

* Cuenta en [RunPod](https://www.runpod.io/).
* Docker instalado localmente (para construir la imagen antes del despliegue en un registro público).
* Se recomienda un contenedor/endpoint de RunPod basado en GPUs NVIDIA (A10, A30, A100, H100, RTX A6000 o superiores) debido a la naturaleza optimizada del framework NeMo.

## Estructura de Archivos

* `handler.py`: Script principal que carga dinámicamente el checkpoint de HuggingFace `nvidia/magpie_tts_multilingual_357m` y define el manejador del job.
* `Dockerfile`: Entorno de sistema operativo y dependencias requeridas (incluyendo libsndfile, toolkit NeMo especializado y PyTorch acelerado por CUDA).
* `requirements.txt`: Módulos de Python suplementarios ligeros.

## Configuración y Despliegue

### Variables de Entorno (Opcionales)

El modelo en este caso es un modelo de peso abierto de NVIDIA y no requiere intrínsecamente un token de acceso, pero el endpoint admite anulación del modelo local a través de variables:

| Variable | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `MODEL_NAME` | El identificador o path del modelo. | `nvidia/magpie_tts_multilingual_357m` |

### Construcción de la Imagen Docker

Construye la imagen Docker localmente, especificando una etiqueta acorde a tu registro actual (ej. Docker Hub o Github Container Registry):

```bash
docker build -t tu_usuario/magpietts-multilingual-api:latest .
```

Haz push de la imagen:

```bash
docker push tu_usuario/magpietts-multilingual-api:latest
```

### Crear el Endpoint en RunPod

1. Dirígete a **RunPod** -> **Serverless** -> **Templates**.
2. Presiona "New Template", ingresa el nombre de tu imagen base `tu_usuario/magpietts-multilingual-api:latest`.
3. Expande tu contenedor y asocia el Endpoint creado. La carga inicial del modelo (357M parámetros) tomará algunos segundos de instanciación ("warmup").

## Uso de la API

La carga de entrada JSON (`input`) requiere mandatoriamente la propiedad `text`.

### Payload de Entrada

```json
{
  "input": {
    "text": "Hola mundo, esta es la voz de Sofia en español generada desde RunPod.",
    "language": "es",
    "speaker": "Sofia",
    "apply_TN": false
  }
}
```

#### Parámetros

* `text` (string, **Requerido**): El guión a sintetizar. (Límite sugerido por estándar: ~20 segundos o sentencias cortas-medianas).
* `language` (string, opcional): El código ISO de dos letras del idioma. Funciona con `es`, `en`, `de`, `fr`, `vi`, `it`, `zh`, `hi`, `ja`. (Por defecto: `en`).
* `speaker` (string, opcional): La voz a utilizar. Están disponibles `John`, `Sofia`, `Aria`, `Jason` y `Leo`. (Por defecto: `Sofia`).
* `apply_TN` (booleano, opcional): Si se debe aplicar Normalización de Texto (útil para expandir números, abreviaturas, etc.). Soportada en todos los idiomas excepto `vi`. (Por defecto: `false`).

### Respuesta (Ejemplo)

La API te entregará en formato JSON un string Base64 convertible a `.wav` (22050Hz) y la longitud estimada.

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "audio_length": 860432
}
```
