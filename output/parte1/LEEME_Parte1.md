# Parte 1: presentación comparativa de segmentación

## Material preparado

- `Presentacion_Split_and_Merge_Parte1.pdf`: 16 diapositivas. Las primeras 14 forman la exposición de 10 minutos; las dos últimas contienen las referencias.
- `Guion_Presentacion_y_Demo.pdf`: texto de exposición con tiempos, plan de demo de 4:50 y preguntas para ensayar.
- `Guion_Presentacion_y_Demo.md`: versión editable del guion.
- `Demo_Split_Merge_5_min.ipynb`: notebook autocontenido, con tres galaxias de validación embebidas y resultados completos de los siete tratamientos. Se ejecutaron localmente las celdas de Python de la demo y se guardaron sus salidas. La instalación inicial y la apertura en Google Colab se realizan al subir el notebook.
- `resultados_7_tratamientos.csv`: métricas exportadas del experimento completo.

El ZIP también contiene `segmentacion.py`, los CSV de las particiones y la evidencia de los siete tratamientos. No incluye el HDF5 ni los pesos de Qwen.

## Uso de la demo

1. Abrir https://colab.research.google.com/.
2. Elegir Archivo > Subir notebook y seleccionar el `.ipynb`.
3. Ejecutar todas las celdas una vez antes de grabar. La instalación inicial depende de la conexión y queda fuera de la demostración cronometrada.
4. Seguir el guion de 4:50. La segmentación usa CPU y las predicciones de Qwen se cargan como resultados previamente calculados.

Los tres ejemplos muestran contrastes y no sustituyen las 1,000 imágenes por tratamiento. Son los IDs 14633 (perfil, clasificación correcta), 8101 (espiral barrada) y 5893 (primera máscara vacía por ID en Gaussian + split and merge).

## Fuentes y alcance

Artículo principal: Horowitz, S. L., & Pavlidis, T. (1976). *Picture Segmentation by a Tree Traversal Algorithm*. Journal of the ACM, 23(2), 368-388. https://doi.org/10.1145/321941.321956

La presentación usa la motivación y formulación accesibles del trabajo como antecedente. El texto accesible también puede consultarse en https://paperity.org/p/376822885/picture-segmentation-by-a-tree-traversal-algorithm. No se atribuyen al artículo cifras de desempeño obtenidas en Galaxy10 ni se presenta el código como una reproducción literal de su recorrido de árbol.

Bibliografía complementaria:

- Fu, G., Zhao, H., Li, C., & Shi, L. (2013). *Segmentation for High-Resolution Optical Remote Sensing Imagery Using Improved Quadtree and Region Adjacency Graph Technique*. Remote Sensing, 5(7), 3259-3279. https://doi.org/10.3390/rs5073259
- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076
- Leung, W. H., & Bovy, J. (2024). *Galaxy10 DECaLS* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10845026
- OpenCV. *Image Thresholding*. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

La variante del proyecto usa desviación estándar, una cola de prioridad para fusionar regiones vecinas y una regla propia de primer plano basada en el perímetro. El desempeño corresponde a la combinación de estas decisiones. La selección bibliográfica ofrece un antecedente y una aplicación complementaria, sin constituir una revisión exhaustiva del estado del arte.

## Antes de entregar

Asignar las secciones a todos los integrantes reales y ensayar con cronómetro. Grabar la presentación de 10 minutos y la demo de máximo 5 minutos con voz y participación del equipo. El paquete no contiene un video grabado. Entregar el PDF y el video según las indicaciones de Canvas.
