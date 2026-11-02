"""Segmentación split-and-merge con NumPy y OpenCV.

Los umbrales usan intensidades de 0 a 255. Los valores predeterminados son
puntos de partida que deben ajustarse con imágenes de validación.
"""

import heapq

import cv2
import numpy as np


def segmentar_split_merge(rgb, tau=10.0, min_size=4, fondo_k=3.0):
    """Devuelve (rgb_segmentada, mascara, etiquetas, estadisticas).

    Divide bloques heterogéneos en cuatro y fusiona regiones que comparten
    borde cuando la desviación estándar de su unión es <= tau. La fusión
    prioriza la unión con menor desviación y actualiza las adyacencias.

    La máscara conserva regiones cuya intensidad media supera la mediana
    del borde + fondo_k * sigma robusta del borde. Esta regla supone fondo
    oscuro en el perímetro; no identifica semánticamente la galaxia central.

    Args:
        rgb: Imagen uint8 de forma (alto, ancho, 3), en orden RGB.
        tau: Desviación estándar máxima para considerar homogénea una región.
        min_size: Tamaño mínimo de los hijos al dividir un bloque.
        fondo_k: Distancia al fondo, en unidades de sigma robusta.

    Returns:
        rgb_segmentada: RGB de entrada con fondo negro.
        mascara: uint8, con valores 0 y 255.
        etiquetas: int32, etiquetas consecutivas desde cero, incluido el fondo.
        estadisticas: Conteos y parámetros descriptivos de la segmentación.
    """
    rgb = np.asarray(rgb)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or min(rgb.shape[:2]) == 0):
        raise ValueError("rgb debe ser una imagen uint8 no vacía de forma (H, W, 3).")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau debe ser finito y >= 0.")
    if not isinstance(min_size, (int, np.integer)) or min_size < 1:
        raise ValueError("min_size debe ser un entero >= 1.")
    if not np.isfinite(fondo_k) or fondo_k < 0:
        raise ValueError("fondo_k debe ser finito y >= 0.")

    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alto, ancho = gris.shape
    etiquetas = np.empty((alto, ancho), dtype=np.int32)
    cantidades, sumas, cuadrados = [], [], []

    def dividir(y0, y1, x0, x1):
        bloque = gris[y0:y1, x0:x1].astype(np.float64)
        puede_dividir = (y1 - y0 >= 2 * min_size
                         and x1 - x0 >= 2 * min_size)
        if puede_dividir and bloque.std() > tau:
            ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
            dividir(y0, ym, x0, xm)
            dividir(y0, ym, xm, x1)
            dividir(ym, y1, x0, xm)
            dividir(ym, y1, xm, x1)
        else:
            etiqueta = len(cantidades)
            etiquetas[y0:y1, x0:x1] = etiqueta
            cantidades.append(bloque.size)
            sumas.append(float(bloque.sum()))
            cuadrados.append(float(np.square(bloque).sum()))

    dividir(0, alto, 0, ancho)
    n_split = len(cantidades)
    vecinos = [set() for _ in range(n_split)]

    # Adyacencia de cuatro vecinos, también entre hojas de distinto tamaño.
    for a, b in ((etiquetas[:, :-1], etiquetas[:, 1:]),
                 (etiquetas[:-1, :], etiquetas[1:, :])):
        diferentes = a != b
        pares = np.column_stack((a[diferentes], b[diferentes]))
        pares.sort(axis=1)
        for i, j in np.unique(pares, axis=0):
            vecinos[i].add(int(j))
            vecinos[j].add(int(i))

    padres = np.arange(n_split)
    versiones = np.zeros(n_split, dtype=np.int64)
    pendientes = []

    def desviacion_union(i, j):
        n = cantidades[i] + cantidades[j]
        media = (sumas[i] + sumas[j]) / n
        varianza = (cuadrados[i] + cuadrados[j]) / n - media * media
        return float(np.sqrt(max(0.0, varianza)))

    def proponer(i, j):
        i, j = min(i, j), max(i, j)
        desviacion = desviacion_union(i, j)
        if desviacion <= tau:
            heapq.heappush(pendientes, (desviacion, i, j,
                                       int(versiones[i]), int(versiones[j])))

    for i in range(n_split):
        for j in sorted(vecinos[i]):
            if i < j:
                proponer(i, j)

    while pendientes:
        _, i, j, vi, vj = heapq.heappop(pendientes)
        if (padres[i] != i or padres[j] != j
                or versiones[i] != vi or versiones[j] != vj
                or j not in vecinos[i]):
            continue

        # i absorbe j; las estadísticas permiten calcular la varianza exacta
        # de la unión, sin recorrer otra vez todos sus píxeles.
        padres[j] = i
        cantidades[i] += cantidades[j]
        sumas[i] += sumas[j]
        cuadrados[i] += cuadrados[j]
        versiones[i] += 1
        adyacentes = (vecinos[i] | vecinos[j]) - {i, j}
        for k in sorted(adyacentes):
            vecinos[k].discard(j)
            vecinos[k].add(i)
        vecinos[i] = adyacentes
        vecinos[j].clear()
        for k in sorted(adyacentes):
            proponer(i, k)

    def raiz(i):
        while padres[i] != i:
            padres[i] = padres[padres[i]]
            i = int(padres[i])
        return i

    raices = np.array([raiz(i) for i in range(n_split)])
    regiones, indices = np.unique(raices, return_inverse=True)
    etiquetas = indices[etiquetas].astype(np.int32)
    medias = np.array([sumas[i] / cantidades[i] for i in regiones])

    # Cada píxel del perímetro se cuenta una sola vez, incluso en imágenes
    # de una sola fila o columna.
    borde = np.zeros(gris.shape, dtype=bool)
    borde[[0, -1], :] = True
    borde[:, [0, -1]] = True
    valores_borde = gris[borde].astype(np.float64)
    fondo = float(np.median(valores_borde))
    mad = float(np.median(np.abs(valores_borde - fondo)))
    sigma_fondo = max(1.0, 1.4826 * mad)
    umbral_fondo = fondo + fondo_k * sigma_fondo
    mascara = (medias[etiquetas] > umbral_fondo).astype(np.uint8) * 255
    segmentada = cv2.bitwise_and(rgb, rgb, mask=mascara)
    n_componentes, _ = cv2.connectedComponents(mascara, connectivity=8)

    stats = {
        "otsu_threshold": np.nan,
        "segmented_pct": float(100 * np.count_nonzero(mascara) / mascara.size),
        "num_regions": int(n_componentes - 1),
        "split_regions": n_split,
        "merged_regions": int(len(regiones)),
        "foreground_regions": int(np.count_nonzero(medias > umbral_fondo)),
        "background_threshold": umbral_fondo,
    }
    return segmentada, mascara, etiquetas, stats
