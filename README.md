# Galaxy10: segmentación y clasificación morfológica de galaxias

Comparación de siete tratamientos de imagen con **Qwen3-VL-4B-Instruct**, usando Galaxy10 DECaLS y MLflow. Pregunta: ¿el preprocesamiento y la segmentación mejoran la clasificación morfológica?

## Alcance: experimento completo y prueba para el tutorial

| Material | Alcance | Estado |
|---|---|---|
| **Slides y resultados experimentales** | Las mismas **1,000 galaxias de validación**, 100 por clase, en siete tratamientos: **7,000 clasificaciones**. | Experimento terminado; resultados incluidos en la presentación. |
| **Código de prueba refinada: notebook del tutorial** | Las mismas **10 galaxias**, una por clase, en cada tratamiento: **70 imágenes procesadas**. | Probado localmente; falta probarlo en Google Colab. |
| **Código del repositorio** | Pipeline completo para preparar datos, segmentar, clasificar y registrar resultados. | Disponible; conserva la evaluación completa de 1,000 imágenes por tratamiento. |

El notebook ejecuta el procesamiento de los diez ejemplos y consulta las predicciones de Qwen ya calculadas. La inferencia nueva con Qwen y el registro de una nueva demo en MLflow son opcionales y están desactivados por defecto. Las métricas de la muestra se separan de las del experimento completo; **la muestra de diez no sustituye la validación de 1,000**.

## Entregables: qué tenemos y qué falta

### Construye 1: explicar el artículo y el método

| Entregable | Qué tenemos | Qué falta |
|---|---|---|
| **Presentación en PDF** | Slides en Canva y PDF, análisis ampliado del artículo principal, fundamentos, algoritmo, aplicaciones, ventajas, limitaciones y resultados del experimento completo. | Revisión final del equipo. |
| **Video de 10 minutos + demo de máximo 5 minutos** | Presentación y notebook para demostrar el método. | Grabar la exposición y la demo; ensayar los tiempos. |

### Construye 2: enseñar la implementación

| Entregable | Qué tenemos | Qué falta |
|---|---|---|
| **Google Colab y código** | Repositorio con el pipeline completo y notebook refinado de diez ejemplos por tratamiento, con explicación, resultados, limitaciones y referencias. | Probarlo en Colab y completar nombres y contribuciones reales de los integrantes. |
| **Video tutorial de 5 a 10 minutos** | Código y ejemplos listos para recorrer la implementación. | Grabar la ejecución y la explicación paso a paso. |
| **Presentación del artículo** | Se reutiliza la presentación de Construye 1, con el análisis ampliado del artículo. | Revisión final del equipo. |
| **ZIP final** | Código, notebook, presentación y un ZIP de exportaciones de la demo. | Integrar los materiales definitivos y el video tutorial. Los ZIP existentes no son la entrega final completa. |

**Pendientes comunes:** terminar los videos, verificar Colab, completar las contribuciones y preparar/subir las entregas finales. El guion anterior corresponde a 22 slides; requiere ajustarse a la presentación actual de 39.

## Ubicación de los materiales

Rutas relativas a la raíz del repositorio:

| Material | Ubicación |
|---|---|
| **Presentación actual, 39 slides** | [output/pdf/Presentacion_Galaxy10_Tec_Articulo_Ampliado.pdf](output/pdf/Presentacion_Galaxy10_Tec_Articulo_Ampliado.pdf) |
| **Presentación editable en Canva** | [Abrir editor](https://www.canva.com/d/XXyHnIu-gf-PTGw); enlaces y notas en [output/experimento/Canva_Presentacion_Tec.md](output/experimento/Canva_Presentacion_Tec.md) |
| **Notebook vigente: diez ejemplos por tratamiento** | [output/tutorial/Tutorial_Galaxy10_10_por_tratamiento.ipynb](output/tutorial/Tutorial_Galaxy10_10_por_tratamiento.ipynb) |
| Exportaciones de la demo | [output/tutorial/Demo_Galaxy10_10_por_tratamiento.zip](output/tutorial/Demo_Galaxy10_10_por_tratamiento.zip) |
| Análisis del artículo principal | [output/experimento/Analisis_Articulo_Horowitz_Pavlidis.md](output/experimento/Analisis_Articulo_Horowitz_Pavlidis.md) |
| Anexo de referencias, métricas y matrices | [output/pdf/Anexo_Referencias_y_Resultados_Galaxy10.pdf](output/pdf/Anexo_Referencias_y_Resultados_Galaxy10.pdf) |
| Resultados completos de clasificación | [resultados_qwen/comparacion_mlflow.csv](resultados_qwen/comparacion_mlflow.csv) y `resultados_qwen/<tratamiento>/` |
| Exportaciones completas de segmentación | `resultados_split_merge/gaussian/` y `resultados_split_merge/clahe/` |
| Seguimiento experimental local | `mlflow.db` y `mlartifacts/` |
| Código fuente y dependencias | Archivos `.py` de la raíz y [requirements.txt](requirements.txt) |

`output/parte1/Demo_Split_Merge_5_min.ipynb`, los otros PDF de presentación y `output/Parte1_Split_and_Merge.zip` son versiones anteriores. Para el tutorial usar el notebook de `output/tutorial/`; para las slides, el PDF ampliado indicado arriba.

## Datos y método

**Galaxy10 DECaLS:** 17,736 imágenes RGB de 256 × 256, diez clases. La evaluación reportada utiliza exclusivamente `data/validacion_1000.csv`: 1,000 imágenes, 100 por clase. Las 2,000 imágenes de prueba reservadas no se evaluaron y no sustentan estos resultados; se verificó que no se solapan con validación.

| ID | Clase | ID | Clase |
|---|---|---|---|
| 0 | Disturbed | 5 | Barred Spiral |
| 1 | Merging | 6 | Unbarred Tight Spiral |
| 2 | Round Smooth | 7 | Unbarred Loose Spiral |
| 3 | In-between Smooth | 8 | Edge-on without Bulge |
| 4 | Cigar-shaped Smooth | 9 | Edge-on with Bulge |

Flujo: **imagen → tratamiento → Qwen → clase 0–9 → métricas → MLflow**. Se mantuvieron las mismas imágenes, modelo, prompt y generación determinista (`do_sample=False`). Modelo cuantizado a 4 bits NF4, ejecutado localmente en una NVIDIA RTX 4060, sin fine-tuning.

| Tratamiento | Configuración |
|---|---|
| Original | RGB sin modificar. |
| Gaussian | Kernel 5 × 5, sigma 1. |
| CLAHE | Canal L de LAB; `clipLimit=2`, cuadrícula 8 × 8. |
| Gaussian + Otsu | Gaussian, umbral automático en grises y máscara aplicada al RGB procesado. |
| CLAHE + Otsu | CLAHE, umbral automático en grises y máscara aplicada al RGB procesado. |
| Gaussian + split and merge | Gaussian y segmentación con `tau=10`, `min_size=4`, `fondo_k=3`. |
| CLAHE + split and merge | CLAHE y la misma configuración de split and merge. |

**Split and merge:** `segmentacion.py` divide bloques en cuatro cuando su desviación estándar supera `tau`, hasta el tamaño mínimo. Fusiona regiones que comparten borde si la desviación estándar de la unión es ≤ `tau`, priorizando la unión más homogénea y actualizando adyacencias. Los bloques mínimos pueden conservar heterogeneidad. Para seleccionar primer plano se exige que la media de la región supere `mediana del perímetro + fondo_k × max(1, 1.4826 × MAD)`.

Es una variante inspirada en Horowitz y Pavlidis (1976), con criterio de desviación estándar y selección de fondo propios; el artículo original utiliza un criterio de rango de intensidades. Qwen recibe el **RGB enmascarado**. Los parámetros son la configuración evaluada, sin ajustes manuales por imagen ni demostración de optimalidad.

## Resultados completos: siete tratamientos

**7,000 clasificaciones, 1,000 por tratamiento; cero respuestas inválidas.** Accuracy mide la proporción de aciertos; F1 macro promedia el F1 de las diez clases con el mismo peso.

| Tratamiento | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Original | 0.2820 | 0.2417 | 0.2820 | 0.2202 |
| Gaussian | 0.2860 | 0.2857 | 0.2860 | 0.2224 |
| CLAHE | 0.3100 | 0.2515 | 0.3100 | 0.2387 |
| Gaussian + Otsu | 0.2300 | 0.2288 | 0.2300 | 0.1698 |
| **CLAHE + Otsu** | **0.3130** | **0.3439** | **0.3130** | **0.2472** |
| Gaussian + split and merge | 0.2180 | 0.2020 | 0.2180 | 0.1565 |
| CLAHE + split and merge | 0.1620 | 0.1595 | 0.1620 | 0.0926 |

### Cambio respecto a la imagen original

| Tratamiento | Cambio de accuracy (puntos porcentuales) | Cambio de F1 macro |
|---|---:|---:|
| Gaussian | +0.4 | +0.0022 |
| CLAHE | +2.8 | +0.0185 |
| Gaussian + Otsu | −5.2 | −0.0504 |
| **CLAHE + Otsu** | **+3.1** | **+0.0270** |
| Gaussian + split and merge | −6.4 | −0.0637 |
| CLAHE + split and merge | −12.0 | −0.1276 |

- **Mejor resultado:** CLAHE + Otsu, 31.3 % de accuracy y 0.2472 de F1 macro; mejora relativa de F1 ≈ 12.2 % frente al original. En la comparación emparejada corrigió 94 errores y perdió 63 aciertos: 31 aciertos netos; 219 imágenes fueron correctas en ambos y 624 incorrectas en ambos.
- **Agregar segmentación no siempre ayuda:** Gaussian + split and merge pierde 0.0660 de F1 frente a Gaussian; CLAHE + split and merge pierde 0.1461 frente a CLAHE y obtiene el menor desempeño global.
- **Errores por clase:** CLAHE + split and merge predice *Round Smooth* 604 veces, frente a 100 casos reales; original lo hace 524 veces y CLAHE + Otsu, 410. La clase 3 tiene F1 = 0 en los siete tratamientos; el máximo F1 de la clase 6 es apenas 0.0367.

Las métricas completas por clase, F1 weighted y predicciones están en `resultados_qwen/comparacion_mlflow.csv` y en cada carpeta de tratamiento. F1 weighted coincide con F1 macro porque todas las clases tienen 100 ejemplos.

### Estadísticas de segmentación

Medias sobre las 1,000 imágenes de cada tratamiento:

| Tratamiento | Píxeles conservados (%) | Componentes conectados | Umbral Otsu |
|---|---:|---:|---:|
| Gaussian + Otsu | 5.4582 | 9.236 | 86.321 |
| CLAHE + Otsu | 33.5811 | 2000.642 | 84.996 |
| Gaussian + split and merge | 7.9231 | 3.633 | — |
| CLAHE + split and merge | 2.9448 | 6.526 | — |

| Split and merge | Regiones tras dividir | Regiones tras fusionar | Regiones seleccionadas | Umbral de fondo | Máscaras vacías / completas |
|---|---:|---:|---:|---:|---:|
| Gaussian | 355.378 | 72.105 | 70.797 | 48.9704 | 5 / 1 |
| CLAHE | 4060.285 | 3970.012 | 60.885 | 164.0544 | 0 / 0 |

`num_regions` cuenta componentes conectados del primer plano con conectividad 8; no equivale al número de regiones tras la fusión. Los conteos de máscaras vacías/completas son totales.

### Tiempos

| Tratamiento | Generación media Qwen (s/imagen) | Preprocesamiento medio (s/imagen) | Duración aproximada registrada en la fase inicial |
|---|---:|---:|---|
| Original | 0.3274 | No medido | 11 min 51 s |
| Gaussian | 0.3257 | No medido | 11 min 53 s |
| CLAHE | 0.2961 | No medido | 11 min 30 s |
| Gaussian + Otsu | 0.3090 | No medido | 11 min 57 s |
| CLAHE + Otsu | 0.3080 | No medido | 11 min 41 s |
| Gaussian + split and merge | 0.2963 | 0.0250 | — |
| CLAHE + split and merge | 0.3002 | 0.0695 | — |

Las duraciones históricas de los **cinco tratamientos iniciales** suman **58 min 52 s**; no son el tiempo total de los siete. `latency_s` mide generación del modelo, no la ejecución completa. El preprocesamiento se midió por separado únicamente en los dos tratamientos de split and merge.

### Conclusión y límites

CLAHE + Otsu fue el mejor tratamiento en esta validación; ambas variantes de split and merge redujeron el desempeño. La selección de primer plano puede recortar estructuras tenues o conservar estrellas; atribuirle la caída de clasificación es una hipótesis, no una causa demostrada.

Accuracy/F1 evalúan **clasificación**. No hay máscaras de referencia para medir IoU/Dice. El modelo no recibió fine-tuning y no se realizó una prueba de significancia estadística; los resultados corresponden a esta validación, sin demostrar superioridad generalizable.

## MLflow y archivos de resultados

Los **siete runs** se encuentran en el mismo experimento: **`Galaxy10_Qwen3VL_Otsu`**. Cada uno registra parámetros, métricas generales y por clase, predicciones, matriz de confusión, tiempos y estadísticas de segmentación cuando corresponden.

Desde la raíz del proyecto:

```powershell
conda activate hitvision
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir [MLflow local](http://127.0.0.1:5000). La base está en `mlflow.db` y los artefactos en `mlartifacts/`. Las rutas pueden depender del equipo original; los resultados de `resultados_qwen/` se consultan sin descargar Qwen ni abrir MLflow.

Cada carpeta `resultados_qwen/<tratamiento>/` contiene `predictions.csv`, `metrics.json`, `config.json`, `mlflow_run_id.txt` y `confusion_matrix.png`. La comparación consolidada está en `resultados_qwen/comparacion_mlflow.csv`.

## Qué código ejecutar

| Archivo | Uso |
|---|---|
| `output/tutorial/Tutorial_Galaxy10_10_por_tratamiento.ipynb` | **Para el tutorial y la demo:** subir a Colab y ejecutar en orden. Incluye las diez galaxias; el recorrido predeterminado usa CPU y predicciones guardadas. |
| `segmentacion.py` | Implementación de split and merge, incluida en el notebook. Ejecutarlo solo define la función. |
| `evaluar_qwen.py` | Reproduce la evaluación completa y registra en MLflow. Ya está terminada; no hace falta repetirla en el video. |
| `experimentar_split_merge.py` | Exporta máscaras, RGB y mapas de regiones de las 1,000 imágenes por preprocesamiento. Las exportaciones completas ya existen. |
| `crear_notebook_tutorial.py` | Regenera el notebook desde datos y resultados locales; no es necesario para ejecutar la demo. |
| `descargar_dataset.py`, `explorar_dataset.py`, `preparar_muestra.py`, `preparar_validacion.py` | Descarga, inspección y preparación de datos. Conservar los CSV originales para reproducir las mismas particiones. |
| `experimentar_otsu.py`, `exportar_imagenes.py`, `probar_qwen.py` | Exploración y comprobaciones auxiliares; el notebook vigente integra el recorrido del tutorial. |

### Reproducir el experimento completo, si se necesita

Usar el entorno `hitvision` o instalar [requirements.txt](requirements.txt) en un entorno compatible con CUDA. El dataset debe estar en `data/Galaxy10_DECals.h5`. `evaluar_qwen.py` usa pesos locales (`local_files_only=True`) y caché `E:\huggingface`; ajustar `HF_HOME`, `HF_HUB_CACHE` y `HF_XET_CACHE` en otro equipo.

```powershell
conda activate hitvision
python evaluar_qwen.py --treatment all
```

Solo las dos variantes de split and merge:

```powershell
python evaluar_qwen.py --treatment split_merge
```

Se puede indicar un tratamiento individual, como `--treatment clahe_otsu` o `--treatment gaussian_split_merge`. El evaluador reanuda predicciones compatibles sin duplicarlas. Para cambiar parámetros, usar otra salida:

```powershell
python evaluar_qwen.py --treatment split_merge --sm-tau 8 --sm-min-size 2 --sm-fondo-k 2 --output resultados_qwen_sm_alternativo
```

Exportar segmentaciones sin cargar Qwen:

```powershell
python experimentar_split_merge.py --preprocesamiento gaussian
python experimentar_split_merge.py --preprocesamiento clahe
```

Cada comando procesa **1,000 imágenes** y guarda `clase_*/`, `estadisticas.csv`, `config.json` y `comparacion.png` bajo `resultados_split_merge/<preprocesamiento>/`. La comparación visual muestra una imagen por clase, sin limitar el procesamiento completo. `--por-clase 1` limita explícitamente a diez y requiere otro `--output`.

Comprobaciones de persistencia, reanudación y configuración, sin Qwen ni escrituras en MLflow:

```powershell
python -m unittest test_experimento_split_merge -v
```

El dataset, los pesos, las cachés y las exportaciones masivas individuales se excluyen de Git. El código del repositorio y los resultados portables permiten revisar el experimento sin incluir esos archivos pesados.

## Referencias

- **Artículo principal:** Horowitz, S. L. y Pavlidis, T. (1976). *Picture Segmentation by a Tree Traversal Algorithm*. JACM, 23(2), 368–388. [DOI](https://doi.org/10.1145/321941.321956).
- **Bibliografía complementaria:** Fu et al. (2013). *Remote Sensing*, 5(7), 3259–3279. [DOI](https://doi.org/10.3390/rs5073259).
- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. [DOI](https://doi.org/10.1109/TSMC.1979.4310076).
- Galaxy10 DECaLS: [dataset en Zenodo](https://zenodo.org/records/10845026), [documentación astroNN](https://astronn.readthedocs.io/en/stable/galaxy10.html) y [repositorio](https://github.com/henrysky/Galaxy10).
- Qwen: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- OpenCV: [suavizado](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html), [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) y [Otsu](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) y [métricas de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
