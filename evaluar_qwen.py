
import os

# Configurar Hugging Face antes de cargar Transformers
os.environ["HF_HOME"] = r"E:\huggingface"
os.environ["HF_HUB_CACHE"] = r"E:\huggingface\hub"
os.environ["HF_XET_CACHE"] = r"E:\huggingface\xet"

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from segmentacion import segmentar_split_merge

from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

TRATAMIENTOS = [
    "original",
    "gaussian",
    "clahe",
    "gaussian_otsu",
    "clahe_otsu",
    "gaussian_split_merge",
    "clahe_split_merge",
]

TRATAMIENTOS_SPLIT_MERGE = [t for t in TRATAMIENTOS if t.endswith("_split_merge")]
COLUMNAS_SPLIT_MERGE = [
    "split_regions", "merged_regions", "foreground_regions",
    "background_threshold", "preprocessing_s",
]

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

COLUMNAS = [
    "image_id",
    "label",
    "prediction",
    "raw_response",
    "valid",
    "latency_s",
    "otsu_threshold",
    "segmented_pct",
    "num_regions",
]

PROMPT = """
Classify the morphology of the main galaxy
near the center of the image.

Choose exactly one category:

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

Return ONLY one digit from 0 to 9.
Do not explain.
""".strip()


def argumentos():
    parser = argparse.ArgumentParser(
        description="Galaxy10 + Qwen3-VL + Otsu / split and merge + MLflow"
    )

    parser.add_argument(
        "--dataset",
        default="data/Galaxy10_DECals.h5"
    )
    parser.add_argument(
        "--validation",
        default="data/validacion_1000.csv"
    )
    parser.add_argument(
        "--test",
        default="data/muestra_2000/etiquetas.csv"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct"
    )
    parser.add_argument(
        "--output",
        default="resultados_qwen"
    )
    parser.add_argument(
        "--experiment",
        default="Galaxy10_Qwen3VL_Otsu"
    )
    parser.add_argument(
        "--treatment",
        choices=["all", "split_merge"] + TRATAMIENTOS,
        default="all"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16
    )

    parser.add_argument("--sm-tau", type=float, default=10.0)
    parser.add_argument("--sm-min-size", type=int, default=4)
    parser.add_argument("--sm-fondo-k", type=float, default=3.0)
    args = parser.parse_args()
    if not np.isfinite(args.sm_tau) or args.sm_tau < 0:
        parser.error("--sm-tau debe ser finito y >= 0")
    if args.sm_min_size < 1:
        parser.error("--sm-min-size debe ser >= 1")
    if not np.isfinite(args.sm_fondo_k) or args.sm_fondo_k < 0:
        parser.error("--sm-fondo-k debe ser finito y >= 0")
    return args


# =========================================================
# 2. PREPROCESAMIENTO Y SEGMENTACIÓN
# =========================================================

def aplicar_tratamiento(rgb, tratamiento, sm_tau=10.0, sm_min_size=4, sm_fondo_k=3.0):
    rgb = np.asarray(rgb, dtype=np.uint8)

    stats = {
        "otsu_threshold": np.nan,
        "segmented_pct": np.nan,
        "num_regions": np.nan,
    }

    if tratamiento == "original":
        return rgb, stats

    if tratamiento in ("gaussian", "gaussian_otsu", "gaussian_split_merge"):
        procesada = cv2.GaussianBlur(
            rgb,
            (5, 5),
            sigmaX=1
        )

    elif tratamiento in ("clahe", "clahe_otsu", "clahe_split_merge"):
        lab = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        procesada = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2RGB
        )

    else:
        raise ValueError(f"Tratamiento inválido: {tratamiento}")

    if tratamiento.endswith("_split_merge"):
        final, _, _, stats = segmentar_split_merge(
            procesada, tau=sm_tau, min_size=sm_min_size, fondo_k=sm_fondo_k
        )
        return final, stats

    if not tratamiento.endswith("_otsu"):
        return procesada, stats

    gris = cv2.cvtColor(
        procesada,
        cv2.COLOR_RGB2GRAY
    )

    umbral, mascara = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Conserva RGB del primer plano; fondo negro.
    final = cv2.bitwise_and(
        procesada,
        procesada,
        mask=mascara
    )

    n_componentes, _ = cv2.connectedComponents(
        mascara,
        connectivity=8
    )

    stats = {
        "otsu_threshold": float(umbral),
        "segmented_pct": float(
            100 * np.count_nonzero(mascara) / mascara.size
        ),
        "num_regions": int(n_componentes - 1),
    }

    return final, stats


# =========================================================
# 3. CARGAR QWEN UNA SOLA VEZ
# =========================================================

def cargar_qwen(model_id):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no disponible. Revisa tu entorno hitvision."
        )

    print("\nCargando Qwen3-VL-4B cuantizado a 4 bits...")

    cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=cuantizacion,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    print("Modelo cargado.")
    print("GPU:", torch.cuda.get_device_name(0))

    return processor, model


# =========================================================
# 4. CLASIFICACIÓN
# =========================================================

def clasificar(rgb, processor, model, max_tokens):
    imagen = Image.fromarray(rgb).convert("RGB")

    mensajes = [{
        "role": "user",
        "content": [
            {"type": "image", "image": imagen},
            {"type": "text", "text": PROMPT},
        ],
    }]

    inputs = processor.apply_chat_template(
        mensajes,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inicio = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencia = time.perf_counter() - inicio

    tokens_nuevos = output[0][
        inputs["input_ids"].shape[-1]:
    ]

    respuesta = processor.decode(
        tokens_nuevos,
        skip_special_tokens=True
    ).strip()

    valido = bool(re.fullmatch(r"[0-9]", respuesta))
    prediccion = int(respuesta) if valido else -1

    return prediccion, respuesta, valido, latencia


# =========================================================
# 5. MÉTRICAS
# =========================================================

def calcular_metricas(df):
    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["prediction"].astype(int).to_numpy()

    clases = list(range(10))

    precision, recall, f1, soporte = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=clases,
            average=None,
            zero_division=0,
        )
    )

    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(
            np.average(f1, weights=soporte)
        ),
        "invalid_responses": int(np.sum(y_pred == -1)),
        "invalid_rate": float(np.mean(y_pred == -1)),
        "mean_latency_s": float(df["latency_s"].mean()),
        "total_inferences": int(len(df)),
    }

    for clase in clases:
        metricas[f"precision_class_{clase}"] = float(
            precision[clase]
        )
        metricas[f"recall_class_{clase}"] = float(
            recall[clase]
        )
        metricas[f"f1_class_{clase}"] = float(
            f1[clase]
        )

    return metricas


def guardar_matriz(df, destino, titulo):
    # Incluye -1 para respuestas inválidas.
    etiquetas = list(range(10)) + [-1]

    matriz = confusion_matrix(
        df["label"].astype(int),
        df["prediction"].astype(int),
        labels=etiquetas,
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[str(i) for i in range(10)] + ["Inv."]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(titulo)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


# =========================================================
# 6. MLflow CON SQLITE
# =========================================================

def configurar_mlflow(experimento):
    raiz = Path.cwd()

    db = (raiz / "mlflow.db").resolve()
    artefactos = (raiz / "mlartifacts").resolve()

    artefactos.mkdir(parents=True, exist_ok=True)

    # URI SQLite absoluta, compatible con Windows.
    tracking_uri = "sqlite:///" + db.as_posix()

    mlflow.set_tracking_uri(tracking_uri)

    cliente = MlflowClient()

    exp = cliente.get_experiment_by_name(experimento)

    if exp is None:
        cliente.create_experiment(
            name=experimento,
            artifact_location=artefactos.as_uri(),
        )

    mlflow.set_experiment(experimento)

    print("\nMLflow configurado")
    print("Base de datos:", db)
    print("Artefactos:", artefactos)

    return cliente


# =========================================================
# 7. VALIDAR DATOS
# =========================================================

def cargar_validacion(args):
    datos = pd.read_csv(args.validation)

    requeridas = {"image_id", "label"}

    if not requeridas.issubset(datos.columns):
        raise ValueError(
            "El archivo de validación necesita image_id y label."
        )

    datos = datos[["image_id", "label"]].copy()

    if datos.isna().any().any():
        raise ValueError("Hay valores faltantes en validación.")

    datos = datos.astype({
        "image_id": int,
        "label": int,
    })

    if datos["image_id"].duplicated().any():
        raise ValueError("Existen image_id duplicados.")

    if not datos["label"].isin(range(10)).all():
        raise ValueError("Existen etiquetas fuera del rango 0-9.")

    # Comprobar que el conjunto de prueba no se utiliza.
    ruta_test = Path(args.test)

    if not ruta_test.exists():
        raise FileNotFoundError(
            "No se encontró el CSV de prueba final. "
            "No podemos verificar que validación y prueba "
            "sean independientes."
        )

    test_ids = set(
        pd.read_csv(ruta_test)["image_id"].astype(int)
    )

    val_ids = set(datos["image_id"])

    if val_ids.intersection(test_ids):
        raise ValueError(
            "Hay imágenes repetidas entre prueba y validación."
        )

    print("\nValidación:", len(datos))
    print("Prueba reservada:", len(test_ids))
    print("Solapamiento: 0")

    return datos


# =========================================================
# 8. GUARDADO Y REANUDACIÓN
# =========================================================

def crear_configuracion(args, datos, tratamiento):
    # Identificador del conjunto exacto de validación.
    registros = datos[["image_id", "label"]].to_numpy()
    dataset_hash = hashlib.sha256(
        registros.astype(np.int64).tobytes()
    ).hexdigest()

    prompt_hash = hashlib.sha256(
        PROMPT.encode("utf-8")
    ).hexdigest()

    config = {
        "model_id": args.model,
        "dataset_path": str(Path(args.dataset).resolve()),
        "validation_hash": dataset_hash,
        "seed": args.seed,
        "treatment": tratamiento,
        "sample_size": len(datos),
        "max_new_tokens": args.max_new_tokens,
        "prompt_hash": prompt_hash,
        "quantization": "NF4_4bit",
        "gaussian_kernel": 5,
        "gaussian_sigma": 1,
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "otsu_output": "masked_rgb",
        "pipeline_version": "v2_sqlite",
    }
    # Mantener intacta la configuración de los cinco experimentos anteriores
    # para poder reanudarlos sin invalidar sus CSV ni sus runs de MLflow.
    if tratamiento.endswith("_split_merge"):
        config.update({
            "pipeline_version": "v3_split_merge",
            "sm_tau": args.sm_tau,
            "sm_min_size": args.sm_min_size,
            "sm_fondo_k": args.sm_fondo_k,
            "sm_homogeneity": "std_union_le_tau",
            "sm_adjacency": 4,
            "sm_merge_order": "lowest_union_std_then_region_ids",
            "sm_foreground": "region_mean_gt_border_median_plus_k_max_1_1.4826mad",
            "sm_output": "masked_rgb",
        })
    return config


def guardar_o_validar_config(ruta, config):
    if ruta.exists():
        anterior = json.loads(
            ruta.read_text(encoding="utf-8")
        )

        if anterior != config:
            raise RuntimeError(
                f"Configuración distinta en {ruta}. "
                "No se mezclarán resultados. "
                "Utiliza otro --output para un nuevo experimento."
            )

    else:
        ruta.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


def cargar_completados(csv_path, datos, columnas=None):
    if not csv_path.exists():
        return set()

    anterior = pd.read_csv(csv_path)

    if not set(COLUMNAS if columnas is None else columnas).issubset(anterior.columns):
        raise ValueError(
            f"CSV incompleto o incompatible: {csv_path}"
        )

    if anterior["image_id"].duplicated().any():
        raise ValueError(
            f"Existen predicciones duplicadas en {csv_path}"
        )

    mapa = datos.set_index("image_id")["label"]

    for fila in anterior.itertuples(index=False):
        image_id = int(fila.image_id)

        if image_id not in mapa.index:
            raise ValueError(
                f"image_id inesperado: {image_id}"
            )

        if int(fila.label) != int(mapa.loc[image_id]):
            raise ValueError(
                f"Etiqueta inconsistente: {image_id}"
            )

    return set(anterior["image_id"].astype(int))


def obtener_run_id(cliente, experimento, carpeta, nombre):
    ruta = carpeta / "mlflow_run_id.txt"

    if ruta.exists():
        run_id = ruta.read_text(
            encoding="utf-8"
        ).strip()

        run = cliente.get_run(run_id)

        if run.info.run_id != run_id:
            raise RuntimeError("Run de MLflow inválido.")

        return run_id

    exp = cliente.get_experiment_by_name(experimento)

    run = cliente.create_run(
        experiment_id=exp.experiment_id,
        tags={"mlflow.runName": nombre},
    )

    ruta.write_text(
        run.info.run_id,
        encoding="utf-8"
    )

    return run.info.run_id


# =========================================================
# 9. EJECUTAR TRATAMIENTO
# =========================================================

def ejecutar_tratamiento(
    nombre,
    datos,
    imagenes,
    processor,
    model,
    args,
    cliente,
    salida,
):
    carpeta = salida / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    csv_path = carpeta / "predictions.csv"
    config_path = carpeta / "config.json"
    metrics_path = carpeta / "metrics.json"
    matrix_path = carpeta / "confusion_matrix.png"
    columnas = COLUMNAS + (COLUMNAS_SPLIT_MERGE if nombre.endswith("_split_merge") else [])

    config = crear_configuracion(args, datos, nombre)

    guardar_o_validar_config(
        config_path,
        config
    )

    completados = cargar_completados(
        csv_path,
        datos,
        columnas
    )

    pendientes = datos[
        ~datos["image_id"].isin(completados)
    ]

    print(f"\n{'=' * 55}")
    print("TRATAMIENTO:", nombre)
    print("Completados:", len(completados))
    print("Pendientes:", len(pendientes))

    run_id = obtener_run_id(
        cliente,
        args.experiment,
        carpeta,
        nombre
    )

    # Reutilizar el mismo run al reanudar.
    with mlflow.start_run(run_id=run_id):

        mlflow.log_params(config)

        crear_cabecera = (
            not csv_path.exists()
            or csv_path.stat().st_size == 0
        )

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as archivo:

            writer = csv.DictWriter(
                archivo,
                fieldnames=columnas
            )

            if crear_cabecera:
                writer.writeheader()

            for fila in tqdm(
                pendientes.itertuples(index=False),
                total=len(pendientes),
                desc=nombre
            ):
                image_id = int(fila.image_id)
                label = int(fila.label)

                rgb = np.asarray(
                    imagenes[image_id],
                    dtype=np.uint8
                )

                inicio_preprocesamiento = time.perf_counter()
                procesada, stats = aplicar_tratamiento(
                    rgb,
                    nombre,
                    sm_tau=args.sm_tau,
                    sm_min_size=args.sm_min_size,
                    sm_fondo_k=args.sm_fondo_k,
                )
                if nombre.endswith("_split_merge"):
                    stats["preprocessing_s"] = time.perf_counter() - inicio_preprocesamiento

                pred, texto, valido, tiempo_s = clasificar(
                    procesada,
                    processor,
                    model,
                    args.max_new_tokens
                )

                writer.writerow({
                    "image_id": image_id,
                    "label": label,
                    "prediction": pred,
                    "raw_response": texto,
                    "valid": valido,
                    "latency_s": tiempo_s,
                    **stats,
                })

                # Persistir cada predicción.
                archivo.flush()

        resultados = pd.read_csv(csv_path)

        esperados = set(datos["image_id"].astype(int))
        observados = set(
            resultados["image_id"].astype(int)
        )

        if (
            len(resultados) != len(datos)
            or observados != esperados
            or resultados["image_id"].duplicated().any()
        ):
            raise RuntimeError(
                f"Resultados incompletos en {nombre}"
            )

        metricas = calcular_metricas(resultados)

        if nombre.endswith("_otsu"):
            metricas.update({
                "mean_otsu_threshold": float(
                    resultados["otsu_threshold"].mean()
                ),
                "mean_segmented_pct": float(
                    resultados["segmented_pct"].mean()
                ),
                "mean_num_regions": float(
                    resultados["num_regions"].mean()
                ),
            })

        if nombre.endswith("_split_merge"):
            for columna in ["segmented_pct", "num_regions"] + COLUMNAS_SPLIT_MERGE:
                metricas[f"mean_{columna}"] = float(resultados[columna].mean())
            metricas["empty_masks"] = int((resultados["segmented_pct"] == 0).sum())
            metricas["full_masks"] = int((resultados["segmented_pct"] == 100).sum())

        guardar_matriz(
            resultados,
            matrix_path,
            f"Galaxy10 — {nombre}"
        )

        metrics_path.write_text(
            json.dumps(
                metricas,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        mlflow.log_metrics(metricas)

        mlflow.log_artifacts(
            str(carpeta),
            artifact_path="resultados"
        )

        mlflow.set_tags({
            "dataset": "Galaxy10_DECaLS",
            "split": "validation",
            "treatment": nombre,
            "model": args.model,
            "status": "complete",
        })

        print("\nRESULTADOS")
        print("Accuracy:", round(metricas["accuracy"], 4))
        print("F1 macro:", round(metricas["f1_macro"], 4))
        print(
            "Precision macro:",
            round(metricas["precision_macro"], 4)
        )
        print(
            "Recall macro:",
            round(metricas["recall_macro"], 4)
        )
        print(
            "Respuestas inválidas:",
            metricas["invalid_responses"]
        )

    return {
        "treatment": nombre,
        **metricas,
    }


# =========================================================
# 10. PIPELINE PRINCIPAL
# =========================================================

def consolidar_resultados(salida, args, datos):
    """Conserva en la comparación los tratamientos previos completos y compatibles."""
    filas = []
    for nombre in TRATAMIENTOS:
        carpeta = salida / nombre
        config_path = carpeta / "config.json"
        metrics_path = carpeta / "metrics.json"
        csv_path = carpeta / "predictions.csv"
        if not all(p.exists() for p in (config_path, metrics_path, csv_path)):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config != crear_configuracion(args, datos, nombre):
            print(f"Comparación: se omite {nombre}, su configuración es distinta.")
            continue
        columnas = COLUMNAS + (COLUMNAS_SPLIT_MERGE if nombre.endswith("_split_merge") else [])
        if len(cargar_completados(csv_path, datos, columnas)) != len(datos):
            continue
        metricas = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metricas.get("total_inferences") != len(datos):
            continue
        filas.append({"treatment": nombre, **metricas})
    return pd.DataFrame(filas)


def main():
    args = argumentos()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)

    datos = cargar_validacion(args)

    # SQLite en vez del backend de archivos obsoleto.
    cliente = configurar_mlflow(
        args.experiment
    )

    tratamientos = (
        TRATAMIENTOS
        if args.treatment == "all"
        else TRATAMIENTOS_SPLIT_MERGE if args.treatment == "split_merge"
        else [args.treatment]
    )

    # Solo cargar el modelo una vez.
    processor, model = cargar_qwen(args.model)

    with h5py.File(args.dataset, "r") as f:
        imagenes = f["images"]

        if (
            datos["image_id"].min() < 0
            or datos["image_id"].max() >= len(imagenes)
        ):
            raise ValueError(
                "Hay índices fuera del dataset."
            )

        for nombre in tratamientos:
            ejecutar_tratamiento(
                nombre,
                datos,
                imagenes,
                processor,
                model,
                args,
                cliente,
                salida
            )

            consolidar_resultados(salida, args, datos).to_csv(
                salida / "comparacion_mlflow.csv",
                index=False
            )

    tabla = consolidar_resultados(salida, args, datos)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETADO")
    print("=" * 60)

    columnas = [
        "treatment",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]

    print(tabla[columnas].to_string(index=False))

    mejor = tabla.loc[
        tabla["f1_macro"].idxmax()
    ]

    print("\nMejor transformación:", mejor["treatment"])
    print("Mejor F1 macro:", round(mejor["f1_macro"], 4))

    print("\nResultados:", salida.resolve())
    print("MLflow database:", Path("mlflow.db").resolve())


if __name__ == "__main__":
    main()
