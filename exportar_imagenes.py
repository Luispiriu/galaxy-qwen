
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

DATASET = Path("data/Galaxy10_DECals.h5")
SALIDA = Path("data/galaxias_png")

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

SALIDA.mkdir(parents=True, exist_ok=True)
registros = []

with h5py.File(DATASET, "r") as f:
    imagenes = f["images"]
    etiquetas = f["ans"][:].astype(int)

    for i in range(len(etiquetas)):
        clase = int(etiquetas[i])
        carpeta = SALIDA / CLASES[clase]
        carpeta.mkdir(parents=True, exist_ok=True)

        imagen = np.asarray(imagenes[i])
        archivo = carpeta / f"galaxia_{i:05d}.png"

        Image.fromarray(imagen.astype(np.uint8)).save(archivo)

        registros.append({
            "image_id": i,
            "clase": clase,
            "archivo": str(archivo)
        })

        if (i + 1) % 1000 == 0:
            print(f"Exportadas: {i + 1}/{len(etiquetas)}")

pd.DataFrame(registros).to_csv(
    SALIDA / "etiquetas.csv",
    index=False
)

print("Exportación terminada.")

