
import os
os.environ["HF_HOME"] = r"E:\huggingface"
os.environ["HF_HUB_CACHE"] = r"E:\huggingface\hub"
os.environ["HF_XET_CACHE"] = r"E:\huggingface\xet"

import re
import torch
import h5py
import pandas as pd

from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig
)

MODELO = "Qwen/Qwen3-VL-4B-Instruct"
DATASET = "data/Galaxy10_DECals.h5"
VALIDACION = "data/validacion_1000.csv"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no disponible. Revisa PyTorch y el driver NVIDIA.")

# 1. Leer una imagen de validación
df = pd.read_csv(VALIDACION)
registro = df.iloc[0]

image_id = int(registro["image_id"])
etiqueta_real = int(registro["label"])

with h5py.File(DATASET, "r") as f:
    imagen = Image.fromarray(f["images"][image_id][:]).convert("RGB")

print("Imagen seleccionada:", image_id)

# 2. Configurar cuantización
configuracion = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 3. Cargar Qwen
print("Cargando Qwen3-VL-4B...")

processor = AutoProcessor.from_pretrained(
    MODELO,
    local_files_only=True
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODELO,
    quantization_config=configuracion,
    device_map="auto",
    local_files_only=True
)

model.eval()

print("Modelo cargado correctamente.")

# 4. Prompt de clasificación
prompt = """
Clasifica la morfología de la galaxia principal, situada
cerca del centro de la imagen.

Opciones:
0: Disturbed
1: Merging
2: Round Smooth
3: In-between Smooth
4: Cigar-shaped Smooth
5: Barred Spiral
6: Unbarred Tight Spiral
7: Unbarred Loose Spiral
8: Edge-on without Bulge
9: Edge-on with Bulge

Responde únicamente con un número entero del 0 al 9.
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": imagen},
            {"type": "text", "text": prompt}
        ]
    }
]

# 5. Preparar entrada
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 6. Inferencia
print("Clasificando galaxia...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False
    )

tokens_generados = output[0][inputs["input_ids"].shape[-1]:]

respuesta = processor.decode(
    tokens_generados,
    skip_special_tokens=True
).strip()

# 7. Evaluación
coincidencia = re.fullmatch(r"[0-9]", respuesta)
prediccion = int(respuesta) if coincidencia else None

print("\nRESULTADOS")
print("Image ID:", image_id)
print("Etiqueta real:", etiqueta_real)
print("Respuesta Qwen:", respuesta)
print("Predicción:", prediccion)
print("¿Correcto?:", prediccion == etiqueta_real)
print(
    "VRAM reservada:",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "GB"
)
