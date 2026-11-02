"""Carga galaxias de validación del HDF5 y guarda su segmentación."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from segmentacion import segmentar_split_merge


BASE = Path(__file__).resolve().parent


def preprocesar(rgb, metodo):
    if metodo == "gaussian":
        return cv2.GaussianBlur(rgb, (5, 5), sigmaX=1)
    if metodo == "clahe":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return rgb.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BASE / "data/Galaxy10_DECals.h5")
    parser.add_argument("--validation", type=Path, default=BASE / "data/validacion_1000.csv")
    parser.add_argument("--test", type=Path, default=BASE / "data/muestra_2000/etiquetas.csv")
    parser.add_argument("--output", type=Path, default=BASE / "resultados_split_merge")
    parser.add_argument("--preprocesamiento", choices=["original", "gaussian", "clahe"], default="gaussian")
    parser.add_argument("--por-clase", type=int, default=0,
                        help="0 (predeterminado): toda la validación. Un valor positivo crea una muestra de inspección.")
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--fondo-k", type=float, default=3.0)
    args = parser.parse_args()
    if args.por_clase < 0:
        parser.error("--por-clase debe ser >= 0")

    df = pd.read_csv(args.validation)
    if not {"image_id", "label"}.issubset(df.columns):
        raise ValueError("El CSV debe contener image_id y label.")
    if df[["image_id", "label"]].isna().any().any() or df.image_id.duplicated().any():
        raise ValueError("La validación contiene valores faltantes o identificadores duplicados.")
    if not df.label.isin(range(10)).all():
        raise ValueError("Las etiquetas deben estar en el rango 0-9.")
    test_ids = set(pd.read_csv(args.test)["image_id"].astype(int))
    if set(df.image_id).intersection(test_ids):
        raise ValueError("Hay solapamiento entre validación y prueba final.")
    if args.por_clase:
        df = df.groupby("label", sort=True).head(args.por_clase)
    if df.empty:
        raise ValueError("No hay imágenes seleccionadas.")
    df = df.sort_values(["label", "image_id"])
    args.output = args.output / args.preprocesamiento
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["sample_size"] = len(df)
    config["selection_hash"] = hashlib.sha256(
        df[["image_id", "label"]].to_numpy(dtype=np.int64).tobytes()
    ).hexdigest()
    config["pipeline_version"] = "v3_split_merge"
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Configuración distinta. Utiliza otro --output para no mezclar resultados.")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Segmentación: {len(df)} imágenes de validación. Solapamiento con prueba: 0.")
    print("La comparación visual muestra una imagen por clase; las estadísticas incluyen toda la selección.")

    registros = []
    vistas = []
    clases_vistas = set()
    with h5py.File(args.dataset, "r") as dataset:
        for fila in df.itertuples(index=False):
            image_id, clase = int(fila.image_id), int(fila.label)
            if not 0 <= image_id < len(dataset["images"]):
                raise ValueError(f"image_id fuera del dataset: {image_id}")
            if int(dataset["ans"][image_id]) != clase:
                raise ValueError(f"Etiqueta inconsistente para image_id={image_id}")
            rgb = np.asarray(dataset["images"][image_id], dtype=np.uint8)
            procesada = preprocesar(rgb, args.preprocesamiento)
            segmentada, mascara, regiones, stats = segmentar_split_merge(
                procesada, tau=args.tau, min_size=args.min_size, fondo_k=args.fondo_k
            )
            carpeta = args.output / f"clase_{clase}"
            carpeta.mkdir(exist_ok=True)
            nombre = f"galaxia_{image_id:05d}"
            Image.fromarray(segmentada).save(carpeta / f"{nombre}_segmentada.png")
            Image.fromarray(mascara).save(carpeta / f"{nombre}_mascara.png")
            np.save(carpeta / f"{nombre}_regiones.npy", regiones)
            # NaN indica un dato no aplicable a este método; CSV lo deja vacío.
            registros.append({"image_id": image_id, "label": clase, **stats})
            if clase not in clases_vistas:
                vistas.append((clase, image_id, rgb, procesada, regiones, mascara, segmentada))
                clases_vistas.add(clase)
            print(f"[{len(registros)}/{len(df)}] galaxia {image_id}, clase {clase}: "
                  f"{stats['segmented_pct']:.1f}% conservado")

    pd.DataFrame(registros).to_csv(args.output / "estadisticas.csv", index=False)
    fig, axes = plt.subplots(len(vistas), 5, figsize=(15, 3 * len(vistas)), squeeze=False)
    titulos = ["Original", args.preprocesamiento, "Regiones", "Máscara", "Segmentada"]
    for i, (clase, image_id, *imagenes) in enumerate(vistas):
        for j, imagen in enumerate(imagenes):
            ax = axes[i, j]
            if j == 2:
                ax.imshow(imagen, cmap="tab20", interpolation="nearest")
            elif j == 3:
                ax.imshow(imagen, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(imagen)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titulos[j])
            if j == 0:
                ax.set_ylabel(f"Clase {clase}\nID {image_id}")
    fig.tight_layout()
    fig.savefig(args.output / "comparacion.png", dpi=120)
    plt.close(fig)
    print(f"Resultados guardados en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
