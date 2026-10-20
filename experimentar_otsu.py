
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Cargar las rutas de imágenes seleccionadas
df = pd.read_csv("data/muestra_2000/etiquetas.csv")

# Una galaxia por clase
muestra = df.groupby("label").first().reset_index()

def aplicar_tratamientos(rgb):
    # Imagen original
    original = rgb.copy()

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(
        rgb, (5, 5), sigmaX=1
    )

    # CLAHE sobre luminancia
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_mejorada = clahe.apply(l)
    lab_mejorada = cv2.merge((l_mejorada, a, b))
    imagen_clahe = cv2.cvtColor(
        lab_mejorada, cv2.COLOR_LAB2RGB
    )

    # Gaussian + Otsu
    gris_gaussian = cv2.cvtColor(
        gaussian, cv2.COLOR_RGB2GRAY
    )

    umbral_g, mascara_g = cv2.threshold(
        gris_gaussian,
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # CLAHE + Otsu
    gris_clahe = cv2.cvtColor(
        imagen_clahe, cv2.COLOR_RGB2GRAY
    )

    umbral_c, mascara_c = cv2.threshold(
        gris_clahe,
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return {
        "Original": original,
        "Gaussian": gaussian,
        "CLAHE": imagen_clahe,
        "Gaussian + Otsu": mascara_g,
        "CLAHE + Otsu": mascara_c,
    }, umbral_g, umbral_c


# Visualizar los cinco tratamientos por clase
fig, axes = plt.subplots(10, 5, figsize=(15, 28))

resultados = []

for fila, (_, registro) in enumerate(muestra.iterrows()):
    imagen = np.array(
        Image.open(registro["filepath"]).convert("RGB")
    )

    tratamientos, t_g, t_c = aplicar_tratamientos(imagen)

    for columna, (nombre, resultado) in enumerate(
        tratamientos.items()
    ):
        ax = axes[fila, columna]

        if resultado.ndim == 2:
            ax.imshow(resultado, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(resultado)

        if fila == 0:
            ax.set_title(nombre, fontsize=11)

        ax.axis("off")

    axes[fila, 0].set_ylabel(
        f"Clase {registro['label']}",
        fontsize=11
    )

    resultados.append({
        "clase": int(registro["label"]),
        "umbral_gaussian": t_g,
        "umbral_clahe": t_c,
    })

plt.tight_layout()
plt.savefig("comparacion_otsu.png", dpi=160)
plt.show()

print(pd.DataFrame(resultados))
