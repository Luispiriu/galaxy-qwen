
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

DATASET = Path("data/Galaxy10_DECals.h5")
PRUEBA = Path("data/muestra_2000/etiquetas.csv")
SALIDA = Path("data/validacion_1000.csv")

SEED = 43
POR_CLASE = 100
rng = np.random.default_rng(SEED)

test = pd.read_csv(PRUEBA)
test_ids = set(test["image_id"].astype(int))

with h5py.File(DATASET, "r") as f:
    labels = f["ans"][:].astype(int)

registros = []

for clase in range(10):
    indices = np.where(labels == clase)[0]

    disponibles = np.array([
        int(i) for i in indices
        if int(i) not in test_ids
    ])

    if len(disponibles) < POR_CLASE:
        raise ValueError(f"No hay suficientes imágenes en clase {clase}")

    seleccion = rng.choice(
        disponibles,
        size=POR_CLASE,
        replace=False
    )

    for image_id in seleccion:
        registros.append({
            "image_id": int(image_id),
            "label": clase
        })

df = pd.DataFrame(registros)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

assert len(df) == 1000
assert not set(df["image_id"]).intersection(test_ids)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SALIDA, index=False)

print("Validación:", len(df))
print("Prueba:", len(test_ids))
print("Solapamiento:", len(set(df["image_id"]) & test_ids))
print(df["label"].value_counts().sort_index())
print("Guardado:", SALIDA)
