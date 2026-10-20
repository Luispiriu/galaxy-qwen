
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RUTA = Path("data/Galaxy10_DECals.h5")

CLASES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Smooth",
    "Cigar-shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

# 1. Verificar integridad y estructura
with h5py.File(RUTA, "r") as f:
    print("Campos disponibles:", list(f.keys()))

    images = f["images"]
    labels = f["ans"][:].astype(int)

    print("Dimensiones:", images.shape)
    print("Número de etiquetas:", len(labels))
    print("Tipo de datos:", images.dtype)

    # 2. Distribución de clases
    valores, conteos = np.unique(labels, return_counts=True)

    print("\nDistribución de clases:")
    for valor, cantidad in zip(valores, conteos):
        print(f"{valor}: {CLASES[valor]} — {cantidad} imágenes")

    # 3. Visualizar una imagen por clase
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for clase, ax in enumerate(axes.flat):
        indices = np.where(labels == clase)[0]

        if len(indices) == 0:
            ax.axis("off")
            continue

        imagen = images[indices[0]]

        ax.imshow(imagen)
        ax.set_title(f"{clase}: {CLASES[clase]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

print("\nDataset leído correctamente.")

