
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# Configuración
DATASET = Path("data/Galaxy10_DECals.h5")
SALIDA = Path("data/muestra_2000")
SEMILLA = 42
IMAGENES_POR_CLASE = 200

CLASES = [
    "0_Disturbed",
    "1_Merging",
    "2_Round_Smooth",
    "3_Inbetween_Smooth",
    "4_Cigar_Shaped",
    "5_Barred_Spiral",
    "6_Unbarred_Tight_Spiral",
    "7_Unbarred_Loose_Spiral",
    "8_Edgeon_No_Bulge",
    "9_Edgeon_With_Bulge",
]

rng = np.random.default_rng(SEMILLA)
SALIDA.mkdir(parents=True, exist_ok=True)

registros = []

with h5py.File(DATASET, "r") as f:
    imagenes = f["images"]
    etiquetas = f["ans"][:].astype(int)

    for clase in range(10):
        # Encontrar imágenes de esta clase
        indices = np.where(etiquetas == clase)[0]

        # Seleccionar 100 sin repetir
        seleccionados = rng.choice(
            indices,
            size=IMAGENES_POR_CLASE,
            replace=False
        )

        carpeta = SALIDA / CLASES[clase]
        carpeta.mkdir(parents=True, exist_ok=True)

        for indice in seleccionados:
            imagen = imagenes[int(indice)]

            nombre = f"galaxia_{int(indice):05d}.png"
            ruta = carpeta / nombre

            Image.fromarray(imagen.astype(np.uint8)).save(ruta)

            registros.append({
                "image_id": int(indice),
                "label": clase,
                "class_name": CLASES[clase],
                "filepath": str(ruta)
            })

        print(f"Clase {clase}: {len(seleccionados)} imágenes")

# Guardar etiquetas y rutas
df = pd.DataFrame(registros)
df.to_csv(SALIDA / "etiquetas.csv", index=False)

print("\nTotal:", len(df))
print("Guardado en:", SALIDA)
