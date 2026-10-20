# Galaxy10 DECaLS: Clasificación morfológica de galaxias mediante Qwen3-VL y procesamiento de imágenes

Proyecto de visión computacional para evaluar el impacto del preprocesamiento digital de imágenes en la clasificación morfológica de galaxias utilizando **Qwen3-VL-4B-Instruct**, el conjunto de datos **Galaxy10 DECaLS** y **MLflow** para el seguimiento experimental.

## 1. Descripción del proyecto

La clasificación morfológica de galaxias consiste en identificar sus características estructurales, como su forma, orientación, presencia de brazos espirales, barras o señales de interacción con otras galaxias.

El reconocimiento de estas características puede verse afectado por las condiciones visuales de las imágenes, incluyendo el contraste, el ruido y la distribución de intensidades.

Este proyecto investiga si diferentes técnicas de procesamiento digital de imágenes pueden modificar el desempeño de un modelo multimodal de visión y lenguaje (VLM) en una tarea de clasificación astronómica.

Se comparan cinco condiciones experimentales utilizando imágenes originales, suavizado gaussiano, ecualización adaptativa CLAHE y segmentación mediante el método de Otsu.

### Pregunta de investigación

**¿En qué medida el suavizado gaussiano, la ecualización adaptativa CLAHE y la segmentación mediante Otsu afectan el desempeño de Qwen3-VL-4B en la clasificación morfológica de galaxias del conjunto Galaxy10 DECaLS?**

### Hipótesis

Las transformaciones de procesamiento digital de imágenes pueden facilitar la identificación de características morfológicas al mejorar el contraste y reducir información visual irrelevante.

Sin embargo, determinadas transformaciones también pueden eliminar detalles estructurales importantes y disminuir el desempeño del modelo.

Por ello, se espera que el efecto del preprocesamiento varíe según la técnica utilizada y la clase morfológica analizada.

## 2. Objetivos

### Objetivo general

Evaluar el impacto de diferentes técnicas de procesamiento digital de imágenes sobre el desempeño de Qwen3-VL-4B-Instruct en la clasificación morfológica de galaxias de Galaxy10 DECaLS.

### Objetivos específicos

1. Implementar transformaciones de suavizado gaussiano, CLAHE y segmentación mediante Otsu.
2. Evaluar el modelo Qwen3-VL-4B-Instruct bajo cinco condiciones visuales.
3. Comparar el desempeño mediante accuracy, precision, recall y F1.
4. Analizar los resultados por clase utilizando matrices de confusión.
5. Registrar los parámetros, métricas y resultados con MLflow.
6. Seleccionar una transformación candidata para una fase posterior de fine-tuning especializado.

## 3. Dataset: Galaxy10 DECaLS

Se utiliza el conjunto de datos **Galaxy10 DECaLS**, compuesto por 17,736 imágenes RGB de galaxias, de 256 × 256 píxeles, organizadas en diez categorías morfológicas.

**Fuentes:**

- [Galaxy10 DECaLS — Zenodo](https://zenodo.org/records/10845026)
- [Documentación de Galaxy10](https://astronn.readthedocs.io/en/stable/galaxy10.html)
- [Repositorio Galaxy10](https://github.com/henrysky/Galaxy10)

### Clases morfológicas

| ID | Clase |
|---|---|
| 0 | Disturbed |
| 1 | Merging |
| 2 | Round Smooth |
| 3 | In-between Smooth |
| 4 | Cigar-shaped Smooth |
| 5 | Barred Spiral |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

### División de los datos

| Conjunto | Imágenes | Uso |
|---|---:|---|
| Validación | 1,000 | Comparación de transformaciones |
| Prueba final | 2,000 | Evaluación final de modelos |
| Restantes | 14,736 | Disponibles para entrenamiento y otras particiones |
| **Total** | **17,736** | |

El conjunto de validación contiene 100 imágenes por clase y el conjunto de prueba reservado contiene 200 imágenes por clase.

Se verificó que no existiera solapamiento entre ambos conjuntos.

**Importante:** las imágenes de prueba final no se utilizaron para calcular los resultados reportados en este README.

## 4. Metodología experimental

Se utilizó el mismo conjunto de validación, modelo de lenguaje visual y prompt de clasificación en los cinco experimentos. La variable modificada fue el procesamiento aplicado a cada imagen.

### Transformaciones evaluadas

| ID | Tratamiento | Configuración |
|---|---|---|
| E0 | Original | Imagen RGB sin modificar |
| E1 | Gaussian | Kernel 5 × 5, sigma = 1 |
| E2 | CLAHE | Canal L en LAB, `clipLimit=2.0`, cuadrícula 8 × 8 |
| E3 | Gaussian + Otsu | Suavizado, escala de grises, Otsu y máscara sobre RGB |
| E4 | CLAHE + Otsu | Realce de contraste, escala de grises, Otsu y máscara sobre RGB |

En los tratamientos que utilizan Otsu, el umbral se calcula automáticamente para cada imagen. La máscara binaria resultante se utiliza para conservar los píxeles correspondientes al primer plano en la imagen RGB procesada, mientras que los demás píxeles se establecen en negro.

No se utilizaron máscaras de segmentación de referencia. Por tanto, las estadísticas de Otsu son descriptivas y no constituyen una evaluación de segmentación mediante IoU o Dice.

### Flujo experimental

```text
Galaxy10 DECaLS
       |
       v
Selección de 1,000 imágenes de validación
       |
       v
Cinco tratamientos de preprocesamiento
       |
       v
Qwen3-VL-4B-Instruct
       |
       v
Predicción de clase morfológica (0-9)
       |
       v
Comparación con etiquetas reales
       |
       v
Accuracy / Precision / Recall / F1
       |
       v
Registro y visualización en MLflow
```

## 5. Modelo y entorno de ejecución

| Elemento | Configuración |
|---|---|
| Modelo | Qwen3-VL-4B-Instruct |
| Arquitectura | Vision-Language Model (VLM) |
| Cuantización | 4 bits, NF4 |
| Hardware | NVIDIA GeForce RTX 4060 |
| Sistema operativo | Windows |
| Entorno Conda | `hitvision` |
| Lenguaje | Python |
| Procesamiento de imágenes | OpenCV |
| Inferencia | PyTorch y Transformers |
| Evaluación | scikit-learn |
| Seguimiento experimental | MLflow 3.17.1 |
| Backend MLflow | SQLite |
| Fine-tuning en esta fase | No |

El modelo se ejecutó mediante inferencia local, sin realizar fine-tuning.

Para cada imagen, se solicitó al modelo seleccionar una de las diez categorías morfológicas y devolver únicamente un número entero del 0 al 9.

Se utilizó generación determinista (`do_sample=False`) y la misma instrucción de clasificación para todos los tratamientos.

## 6. Resultados experimentales

Se completaron **5,000 inferencias**, distribuidas entre los cinco tratamientos.

Cada experimento evaluó las mismas 1,000 imágenes de validación.

### Resultados generales

| Tratamiento | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Original | 0.2820 | 0.2417 | 0.2820 | 0.2202 |
| Gaussian | 0.2860 | 0.2857 | 0.2860 | 0.2224 |
| CLAHE | 0.3100 | 0.2515 | 0.3100 | 0.2387 |
| Gaussian + Otsu | 0.2300 | 0.2288 | 0.2300 | 0.1698 |
| **CLAHE + Otsu** | **0.3130** | **0.3439** | **0.3130** | **0.2472** |

**Mejor tratamiento según F1 macro: CLAHE + Otsu.**

El tratamiento obtuvo:

- Accuracy: **31.3 %**
- Precision macro: **0.3439**
- Recall macro: **0.3130**
- F1 macro: **0.2472**
- Respuestas inválidas: **0**

### Comparación respecto a la imagen original

| Tratamiento | Diferencia de accuracy | Diferencia de F1 macro |
|---|---:|---:|
| Gaussian | +0.4 puntos porcentuales | +0.0022 |
| CLAHE | +2.8 puntos porcentuales | +0.0185 |
| Gaussian + Otsu | −5.2 puntos porcentuales | −0.0504 |
| **CLAHE + Otsu** | **+3.1 puntos porcentuales** | **+0.0270** |

CLAHE + Otsu incrementó el F1 macro aproximadamente **12.2 % en términos relativos** respecto a las imágenes originales.

Todos los experimentos produjeron respuestas válidas dentro del rango de clases 0–9.

### Tiempo de ejecución

| Tratamiento | Tiempo aproximado |
|---|---|
| Original | 11 min 51 s |
| Gaussian | 11 min 53 s |
| CLAHE | 11 min 30 s |
| Gaussian + Otsu | 11 min 57 s |
| CLAHE + Otsu | 11 min 41 s |
| **Total de inferencias** | **58 min 52 s** |

El tiempo total indicado corresponde a la suma de las ejecuciones de clasificación y no incluye necesariamente toda la inicialización del programa.

## 7. Análisis de resultados

### 7.1. Impacto del preprocesamiento

Las métricas muestran diferencias en el desempeño del modelo dependiendo de la transformación utilizada.

Esto respalda, a nivel descriptivo, que el preprocesamiento puede modificar la información visual que utiliza Qwen3-VL-4B para realizar la clasificación.

### 7.2. Efecto del suavizado gaussiano

El suavizado gaussiano produjo una mejora pequeña respecto a las imágenes originales:

- Accuracy: de 0.2820 a 0.2860.
- F1 macro: de 0.2202 a 0.2224.

La diferencia observada es reducida, por lo que no se puede afirmar que el suavizado aporte una mejora generalizable sin análisis adicionales.

### 7.3. Efecto de CLAHE

CLAHE produjo una mejora más considerable:

- Accuracy: de 0.2820 a 0.3100.
- F1 macro: de 0.2202 a 0.2387.

Estos resultados son compatibles con la hipótesis de que el realce de contraste local puede facilitar el reconocimiento de ciertas características morfológicas.

### 7.4. Efecto de Gaussian + Otsu

Gaussian + Otsu obtuvo el menor desempeño:

- Accuracy: 0.2300.
- F1 macro: 0.1698.

Una posible explicación es que el suavizado, seguido de una umbralización global, descarte información visual importante, especialmente estructuras tenues.

Esta interpretación constituye una hipótesis que debe contrastarse mediante análisis visual y métricas por clase.

### 7.5. Efecto de CLAHE + Otsu

CLAHE + Otsu presentó el mejor desempeño general:

- Accuracy: 0.3130.
- Precision macro: 0.3439.
- F1 macro: 0.2472.

El resultado sugiere que la combinación de realce de contraste y segmentación puede beneficiar la clasificación en determinadas condiciones.

Sin embargo, una mejora en la métrica agregada no garantiza mejoras uniformes para las diez clases.

### 7.6. Limitaciones

La exactitud máxima de 31.3 % indica que el desempeño del modelo base sigue siendo limitado para esta tarea especializada.

Adicionalmente:

- No se ha aplicado fine-tuning especializado.
- No se ha realizado una prueba de significancia estadística.
- La evaluación corresponde a un conjunto de validación de 1,000 imágenes.
- Las máscaras producidas por Otsu no se contrastaron con segmentaciones de referencia.
- Es necesario revisar matrices de confusión y F1 por clase para identificar errores específicos.

Los resultados se interpretan como evidencia experimental preliminar y no como demostración definitiva de superioridad de una transformación.

## 8. MLflow: seguimiento de experimentos

Se utilizó MLflow para registrar y comparar automáticamente los resultados de los cinco tratamientos.

**Nombre del experimento:** `Galaxy10_Qwen3VL_Otsu`

Cada ejecución registra:

- Identificador y configuración del tratamiento.
- Parámetros del modelo y del preprocesamiento.
- Accuracy, precision, recall y F1.
- Métricas individuales por clase.
- Tiempo promedio de inferencia.
- Respuestas inválidas.
- Predicciones individuales en CSV.
- Matrices de confusión.
- Estadísticas descriptivas de Otsu, cuando corresponde.

### Abrir MLflow

Desde la carpeta raíz del proyecto:

```powershell
conda activate hitvision
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador:

http://127.0.0.1:5000

La interfaz permite visualizar las métricas, comparar las ejecuciones y revisar los artefactos disponibles.

### Almacenamiento de resultados

| Archivo o carpeta | Descripción |
|---|---|
| `mlflow.db` | Base de datos SQLite con experimentos y métricas |
| `mlartifacts/` | Artefactos registrados en MLflow |
| `resultados_qwen/` | Archivos de resultados por tratamiento |
| `comparacion_mlflow.csv` | Comparación consolidada de métricas |

Las visualizaciones interactivas de MLflow se generan a partir de las métricas almacenadas; no todas existen como archivos PNG independientes.

## 9. Estructura del proyecto

```text
Proyecto final/
|
├── data/
│   ├── Galaxy10_DECals.h5
│   ├── validacion_1000.csv
│   └── muestra_2000/
│       └── etiquetas.csv
│
├── descargar_dataset.py
├── explorar_dataset.py
├── preparar_muestra.py
├── preparar_validacion.py
├── experimentar_otsu.py
├── exportar_imagenes.py
├── probar_qwen.py
├── evaluar_qwen.py
│
├── comparacion_otsu.png
│
├── resultados_qwen/
│   ├── original/
│   ├── gaussian/
│   ├── clahe/
│   ├── gaussian_otsu/
│   ├── clahe_otsu/
│   └── comparacion_mlflow.csv
│
├── mlflow.db
├── mlartifacts/
├── requirements.txt
└── README.md
```

Cada carpeta de tratamiento contiene:

```text
tratamiento/
├── predictions.csv
├── metrics.json
├── config.json
├── mlflow_run_id.txt
└── confusion_matrix.png
```

## 10. Instalación y reproducción

### 10.1. Clonar el repositorio

```powershell
git clone URL_DEL_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO
```

Sustituir los valores anteriores por la dirección y el nombre reales del repositorio.

### 10.2. Crear un entorno virtual

```powershell
conda create -n galaxy10 python=3.12 -y
conda activate galaxy10
```

### 10.3. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

**Nota:** el archivo `requirements.txt` se genera a partir del entorno utilizado durante el desarrollo. Puede requerir adaptaciones si se ejecuta en un sistema operativo diferente o en un equipo sin GPU NVIDIA compatible con CUDA.

### 10.4. Descargar el dataset

Fuente oficial:

https://zenodo.org/records/10845026

El archivo requerido es:

`Galaxy10_DECals.h5`

Debe colocarse en:

```text
data/Galaxy10_DECals.h5
```

El dataset no se incluye en el repositorio debido a su tamaño.

### 10.5. Preparar los conjuntos de datos

El proyecto incluye los scripts:

```powershell
python preparar_muestra.py
python preparar_validacion.py
```

Para reproducir exactamente los experimentos originales, se deben conservar los mismos identificadores de imágenes y etiquetas utilizados en los archivos CSV de validación y prueba.

### 10.6. Configurar Hugging Face

El modelo utilizado es:

https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

El script de evaluación fue configurado originalmente para utilizar la caché ubicada en `E:\huggingface`.

Si se utiliza otra computadora, es necesario ajustar al inicio de `evaluar_qwen.py` las variables `HF_HOME`, `HF_HUB_CACHE` y `HF_XET_CACHE` para que apunten a una ubicación válida.

El modelo debe descargarse previamente. El script de evaluación utiliza `local_files_only=True`.

### 10.7. Ejecutar la evaluación completa

```powershell
python evaluar_qwen.py
```

Este comando evalúa las cinco condiciones experimentales.

Para ejecutar únicamente un tratamiento:

```powershell
python evaluar_qwen.py --treatment clahe_otsu
```

El script permite reanudar una ejecución interrumpida, utilizando las predicciones ya guardadas, siempre que se conserve la misma configuración.

### 10.8. Consultar los resultados sin ejecutar Qwen

Los compañeros que únicamente necesiten revisar las conclusiones pueden consultar:

```text
resultados_qwen/comparacion_mlflow.csv
```

También pueden visualizar las matrices de confusión y los archivos `metrics.json` de cada tratamiento.

No es necesario descargar los pesos de Qwen para consultar estos archivos.

## 11. Colaboración y GitHub

Se recomienda versionar:

- Código fuente en Python.
- `README.md`.
- `requirements.txt`.
- Archivos CSV de partición de datos.
- Resultados experimentales consolidados.
- Matrices de confusión y otras visualizaciones relevantes.

No se recomienda incluir:

- Dataset completo Galaxy10 en formato HDF5.
- Pesos del modelo Qwen.
- Cachés de pip, Hugging Face o Conda.
- Entornos virtuales.
- Archivos temporales.

### Consideraciones para MLflow

La base de datos SQLite y sus artefactos pueden conservarse como respaldo del experimento, pero no constituyen automáticamente un servidor compartido.

Las rutas de artefactos pueden estar vinculadas al equipo donde se generaron. Por lo tanto, para colaborar entre distintas computadoras, los archivos de `resultados_qwen/` son la principal fuente de resultados portables.

Para compartir una instancia interactiva de MLflow entre varios integrantes se requeriría configurar un servidor de seguimiento accesible para el equipo.

## 12. Próximas etapas

### Etapa 1. Análisis detallado por clase

- Analizar los valores de F1 de cada una de las diez clases.
- Identificar los errores más frecuentes mediante matrices de confusión.
- Evaluar qué morfologías se benefician o perjudican con cada transformación.
- Inspeccionar ejemplos correctamente clasificados y errores representativos.

### Etapa 2. Fine-tuning especializado

Implementar un procedimiento de fine-tuning eficiente mediante LoRA o QLoRA, sujeto a la memoria disponible de la GPU.

La configuración candidata inicial será **CLAHE + Otsu**, seleccionada por su F1 macro en validación.

Se deberán separar apropiadamente los conjuntos de entrenamiento y validación para evitar fuga de información.

### Etapa 3. Evaluación final

Comparar el desempeño del modelo original y el modelo ajustado sobre las 2,000 imágenes reservadas para prueba.

El conjunto de prueba final no deberá utilizarse para elegir transformaciones, hiperparámetros o configuraciones de entrenamiento.

### Etapa 4. Análisis estadístico

Evaluar la incertidumbre de las diferencias observadas y aplicar pruebas estadísticas apropiadas para predicciones emparejadas cuando corresponda.

## 13. Conclusiones preliminares

Los experimentos muestran que el procesamiento digital de imágenes modifica el desempeño de Qwen3-VL-4B en la clasificación morfológica de Galaxy10 DECaLS.

**CLAHE + Otsu presentó el mejor resultado global**, alcanzando un F1 macro de 0.2472 y una exactitud de 31.3 %, frente a un F1 macro de 0.2202 y una exactitud de 28.2 % con imágenes originales.

En contraste, Gaussian + Otsu redujo el desempeño, lo que evidencia que la segmentación automática no necesariamente beneficia a un modelo multimodal.

Estos resultados permiten seleccionar una estrategia candidata de preprocesamiento, pero también muestran la necesidad de mejorar el reconocimiento morfológico mediante adaptación especializada y análisis de errores.

**La contribución principal de esta etapa es una evaluación reproducible de cinco estrategias de procesamiento digital de imágenes sobre un mismo modelo multimodal, con métricas comparables y trazabilidad experimental mediante MLflow.**

## 14. Referencias

1. **Galaxy10 DECaLS.** Conjunto de datos disponible en Zenodo. https://zenodo.org/records/10845026
2. **astroNN — Galaxy10 Documentation.** https://astronn.readthedocs.io/en/stable/galaxy10.html
3. **Qwen3-VL-4B-Instruct.** Hugging Face. https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
4. **OpenCV — Smoothing Images.** https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
5. **OpenCV — Histograms: Histogram Equalization (CLAHE).** https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
6. **OpenCV — Image Thresholding (Otsu).** https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
7. **Otsu, N. (1979).** A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics. https://doi.org/10.1109/TSMC.1979.4310076
8. **MLflow Documentation.** https://mlflow.org/docs/latest/ml/tracking/
9. **scikit-learn — Classification Metrics.** https://scikit-learn.org/stable/modules/model_evaluation.html