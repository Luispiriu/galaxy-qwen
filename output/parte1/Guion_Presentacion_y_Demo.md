# Guion de grabación: Parte 1

Duración objetivo de presentación: **10:00**. Demo adicional: **4:50**, con 10 segundos de margen.

Las diapositivas 1-14 forman la exposición. Las 15-16 contienen referencias de consulta. Los intervalos incluyen pausas para señalar imágenes y transiciones. Ensayar con cronómetro: el texto por sí solo no garantiza una duración exacta.

Repartir los bloques entre **todos los integrantes del equipo**, anotando sus nombres antes de grabar. Cada integrante debe poder explicar el algoritmo y responder preguntas.

## 01. Split and merge (00:00-00:20)

Presentamos split and merge, un método de segmentación por regiones. Explicaremos su fundamento, nuestra implementación y una aplicación a galaxias. La pregunta del experimento es si conservar ciertas regiones de la imagen ayuda a Qwen a reconocer su morfología. Después de la presentación mostraremos una demo ejecutable en Google Colab.

## 02. Artículo principal y motivación (00:20-01:05)

El artículo principal es Picture Segmentation by a Tree Traversal Algorithm, de Steven Horowitz y Theodosios Pavlidis, publicado en Journal of the ACM en 1976. Combina división y fusión para segmentar imágenes con una estructura jerárquica. Su motivación incluye las dificultades de detectar objetos con ruido, bajo contraste o fronteras difusas. El trabajo busca eficiencia de procesamiento y memoria. Lo usamos como antecedente fundacional, sin atribuirle resultados de nuestro experimento ni afirmar que nuestra variante reproduzca todos sus procedimientos. La aplicación astronómica y la selección de primer plano son decisiones del proyecto.

## 03. Criterio original y variante implementada (01:05-01:50)

La formulación original usa una tolerancia de error respecto a una constante dentro de cada región. Nuestra variante define homogeneidad mediante desviación estándar en escala de grises. Son criterios diferentes: una desviación baja puede admitir algunos píxeles muy alejados de la media. Fijamos tau igual a diez en intensidades de cero a doscientos cincuenta y cinco. La elección de la característica y su tolerancia determina qué diferencias conserva el algoritmo. Por ello, presentamos una implementación de la familia split and merge, con sus propias decisiones y limitaciones, y no una reproducción literal del artículo.

## 04. Split: división por homogeneidad (01:50-02:35)

Aquí mostramos una galaxia de validación y el mapa real de bloques que genera la división. Empezamos con toda la imagen. Si la desviación supera diez y podemos crear cuatro hijos de al menos cuatro por cuatro píxeles, subdividimos. Repetimos la operación en cada hijo. El fondo uniforme suele producir bloques grandes y las zonas con variaciones generan bloques pequeños. Los colores representan identificadores de regiones, sin significado de clase. El tamaño mínimo detiene la división incluso cuando un bloque sigue siendo heterogéneo. Este límite evita una fragmentación hasta píxeles individuales.

## 05. Merge: fusión de vecinos (02:35-03:30)

Después de dividir, buscamos regiones que comparten borde. La adyacencia usa cuatro vecinos, de modo que tocarse solamente por una esquina no basta. También consideramos vecinos que provienen de ramas distintas del quadtree. Proponemos fusionar una pareja cuando la desviación de su unión es menor o igual a diez. La cola de prioridad selecciona la unión con menor desviación y resuelve empates con los identificadores. Al fusionar actualizamos vecinos, conteos, sumas y sumas de cuadrados. Esas estadísticas permiten calcular la desviación exacta de la unión sin recorrer de nuevo sus píxeles. El proceso termina cuando ya no hay una fusión permitida. El orden de fusión forma parte de la configuración del método.

## 06. Selección de primer plano (03:30-04:20)

El mapa de regiones cubre toda la imagen. Para producir una máscara de galaxia añadimos una regla de primer plano. Estimamos el fondo usando la mediana de los píxeles del perímetro y su desviación absoluta mediana, o MAD. Conservamos una región si su intensidad media supera ese fondo más tres veces una estimación robusta de dispersión. La dispersión tiene un mínimo de uno. Finalmente, aplicamos la máscara sobre el RGB procesado y dejamos el resto en negro. Esta regla supone un borde representativo del fondo. Puede eliminar brazos débiles y conservar estrellas brillantes externas. Es una decisión de nuestra implementación que también influye en el resultado final.

## 07. Código de la implementación (04:20-05:05)

Este fragmento corresponde a la interfaz que ejecutaremos en la demo. Primero aplicamos Gaussian con kernel de cinco por cinco y sigma uno. La función recibe la imagen RGB y los tres parámetros de segmentación. Devuelve la imagen segmentada, la máscara, las etiquetas de las regiones y estadísticas descriptivas. En el notebook incluimos la función completa, documentada, para que la ejecución sea autocontenida. La variante CLAHE realiza el realce sobre luminancia antes de usar la misma función. Las versiones del experimento mantienen la misma configuración durante toda la validación. Cualquier cambio posterior se registra como un experimento distinto.

## 08. Aplicaciones documentadas (05:05-05:45)

Como bibliografía complementaria, Fu y colaboradores publicaron en 2013 una aplicación a imágenes de teledetección de alta resolución. Usan quadtree y un grafo de adyacencia de regiones. Su propuesta incorpora imágenes integrales, indexación espacial y características espectrales y de textura. Esa aplicación ilustra cómo adaptar la familia a otro dominio. Nuestro código usa intensidad en grises y estadísticas de regiones, sin implementar todas esas extensiones. La lección práctica es elegir características y criterios que correspondan a la estructura de interés. En galaxias, una frontera de intensidad puede separar o recortar estructuras astronómicas que pertenecen al mismo objeto.

## 09. Ventajas y limitaciones (05:45-06:25)

Nuestra variante es interpretable: podemos inspeccionar qué bloques se dividieron, qué regiones se unieron y qué umbral produjo la máscara. No requiere entrenamiento ni ejemplos etiquetados para segmentar. A cambio, depende de tau, del tamaño mínimo y de la definición de primer plano. El ruido puede aumentar la fragmentación y los bloques imponen fronteras rectangulares iniciales. La fusión recupera regiones más amplias, pero la homogeneidad estadística no garantiza identidad semántica. Un parámetro fijo también puede actuar de forma distinta después de Gaussian o CLAHE, porque cada transformación cambia la distribución de intensidades.

## 10. Diseño del experimento Galaxy10 (06:25-07:15)

El experimento utiliza mil imágenes de validación de Galaxy10 DECaLS, cien por cada una de las diez clases. Se comparan siete tratamientos y se conservan los mismos identificadores en todos. Qwen3-VL de cuatro mil millones de parámetros trabaja cuantizado a cuatro bits, con el mismo prompt y generación determinista. En total tenemos siete mil clasificaciones. Las dos mil imágenes reservadas para prueba final permanecen independientes. MLflow guarda parámetros, predicciones, métricas y matrices de confusión. Evaluamos accuracy y F1 macro de clasificación. Como no tenemos máscaras de referencia, el porcentaje conservado describe la segmentación, pero no permite calcular su precisión mediante IoU o Dice.

## 11. Resultados de clasificación (07:15-08:00)

El mejor tratamiento de esta validación fue CLAHE más Otsu, con F1 macro de cero punto dos cuatro siete dos y accuracy de treinta y uno punto tres por ciento. Gaussian más split and merge obtuvo F1 de cero punto uno cinco seis cinco y accuracy de veintiuno punto ocho por ciento. CLAHE más split and merge obtuvo cero punto cero nueve dos seis y dieciséis punto dos por ciento. Ambos quedaron por debajo de la imagen original. Esto describe nuestra configuración evaluada y el modelo utilizado. Los resultados negativos también aportan evidencia: una segmentación visualmente sencilla no garantiza que el clasificador reciba la información que necesita.

## 12. Ejemplos y fallos de la máscara (08:00-08:55)

Estos tres ejemplos proceden de validación. El primero es una galaxia de perfil que Qwen clasificó correctamente después de Gaussian más split and merge. El segundo permite revisar una espiral barrada con estructuras tenues. El tercero corresponde a una máscara vacía. Los elegimos para mostrar contrastes, no como una muestra aleatoria representativa. Debemos observar la forma conservada y también lo que desaparece. Una predicción correcta tampoco demuestra que la máscara sea correcta, porque carecemos de referencia de segmentación. En el conjunto completo, Gaussian produjo cinco máscaras vacías y una completamente llena. CLAHE no produjo máscaras vacías, pero tuvo peor clasificación. Por tanto, evitar máscaras vacías es una comprobación útil y claramente insuficiente.

## 13. Interpretación de los resultados (08:55-09:30)

CLAHE más split and merge conservó en promedio el dos punto nueve cuatro por ciento de los píxeles. Además, Qwen asignó la clase Round Smooth a seiscientas cuatro de las mil imágenes, aunque cien pertenecen realmente a ella. Esto muestra un sesgo marcado de predicción. La eliminación de estructura es una hipótesis compatible con la inspección visual, pero estos datos no demuestran una causa. Evaluamos la combinación del particionado y nuestra regla de primer plano. Para separarlas necesitaríamos experimentos adicionales con configuraciones registradas.

## 14. Conclusiones y demo (09:30-10:00)

Split and merge permite construir regiones mediante criterios explícitos de división y fusión. En nuestro experimento, la configuración evaluada redujo el desempeño de Qwen y CLAHE más Otsu mantuvo el mejor F1 macro. La conclusión queda limitada a estas imágenes, parámetros y modelo. Ahora mostraremos la demo en CPU: ejecutaremos la segmentación de tres galaxias y consultaremos los resultados completos. Las referencias y los archivos de configuración permiten rastrear cada afirmación.

## Demo de Google Colab: 4:50

Abrir https://colab.research.google.com/ y subir `Demo_Split_Merge_5_min.ipynb` desde Archivo > Subir notebook. Ejecutar todas las celdas una vez antes de grabar. La instalación inicial depende de internet y debe completarse antes del tiempo de demo.

### 00:00-00:30 · Contexto y datos

Mostrar el título y las tres galaxias embebidas. Explicar: son ejemplos de validación para la demo; las métricas agregadas incluyen las 1,000 imágenes de cada tratamiento.

### 00:30-01:10 · Criterio y código

Mostrar la firma de segmentar_split_merge y los parámetros. Señalar división por desviación, adyacencia por borde y fusión sobre la unión. La función completa está incluida.

### 01:10-02:30 · Ejecución

Ejecutar la celda de los tres casos. Señalar original, regiones, máscara y RGB segmentada. Leer porcentaje conservado y conteos split/merge. Mostrar que se calcula en CPU y no necesita Qwen.

### 02:30-03:20 · Fortalezas y fallo

Contrastar la galaxia de perfil con la espiral y la máscara vacía. Explicar que la regla de fondo puede eliminar estructura útil. Evitar presentar estos casos seleccionados como una evaluación cuantitativa representativa.

### 03:20-04:15 · Resultados completos

Mostrar la tabla de siete tratamientos y el F1 macro. Aclarar que los resultados de Qwen ya estaban calculados y que la celda solo los carga. Identificar CLAHE + Otsu como mejor tratamiento de esta validación.

### 04:15-04:50 · Cierre y referencias

Mostrar las referencias. Explicar que accuracy/F1 miden clasificación, no IoU de segmentación. Cerrar: esta configuración no benefició al modelo y cualquier ajuste requiere otra ejecución registrada.

## Preguntas para ensayar

### ¿Qué distingue split and merge de Otsu?

Otsu calcula un umbral global a partir del histograma. Nuestra variante construye regiones mediante homogeneidad y adyacencia, y después aplica una regla de primer plano.

### ¿La implementación reproduce exactamente el artículo de 1976?

No. Es una variante de la familia. Usa desviación estándar, fusión con una cola de prioridad y una regla propia de fondo. La formulación histórica parte de un criterio de error acotado y un recorrido de árbol.

### ¿Por qué los parámetros son iguales en Gaussian y CLAHE?

Para comparar esa configuración explícita. No demostramos que esos valores sean óptimos para ambos. El realce modifica la distribución de intensidades y puede cambiar fuertemente el resultado.

### ¿Por qué usar F1 macro?

Promedia el F1 de las diez clases con el mismo peso. Permite observar el desempeño desigual entre morfologías, además de accuracy.

### ¿Por qué accuracy coincide con recall macro en estos resultados?

El conjunto tiene el mismo soporte de 100 imágenes por clase. El promedio de recalls coincide con la fracción global de aciertos en este caso.

### ¿Por qué no reportar IoU o Dice?

Tenemos etiquetas de clase, pero no máscaras de referencia. El porcentaje de píxeles conservados no mide exactitud de segmentación.

### ¿Las cinco máscaras vacías explican toda la caída?

No. Representan el 0.5 % de Gaussian + split and merge. Son un fallo verificable, pero la pérdida global requiere analizar también las otras imágenes y la conducta del clasificador.

### ¿La pérdida de estructura causó el menor F1?

Es una hipótesis visual. El experimento no aísla particionado y selección de primer plano, ni demuestra causalidad entre píxeles eliminados y errores.

### ¿Por qué tres imágenes en la demo?

Para explicar casos contrastantes en menos de cinco minutos. Las métricas del experimento provienen de las mismas 1,000 imágenes por tratamiento, no de esos tres ejemplos.

### ¿Dónde se registró la comparación?

En el experimento Galaxy10_Qwen3VL_Otsu de MLflow y en los archivos exportados. Los siete runs terminaron, con 1,000 predicciones cada uno y cero respuestas inválidas.

### ¿Se utilizó prueba final para elegir parámetros?

No. Las 2,000 imágenes de prueba permanecen reservadas. Cualquier ajuste posterior se evalúa con una configuración y salida distintas.

### ¿El artículo sigue siendo relevante aunque sea antiguo?

Es el antecedente fundacional elegido. La bibliografía complementaria muestra una extensión aplicada a teledetección y la implementación actual permite estudiar sus decisiones. No presentamos esta selección como una revisión exhaustiva del estado del arte.

## Bibliografía

- [1] Artículo principal. Horowitz, S. L., & Pavlidis, T. (1976). Picture Segmentation by a Tree Traversal Algorithm. Journal of the ACM, 23(2), 368-388. https://doi.org/10.1145/321941.321956

- [2] Aplicación complementaria. Fu, G., Zhao, H., Li, C., & Shi, L. (2013). Segmentation for High-Resolution Optical Remote Sensing Imagery Using Improved Quadtree and Region Adjacency Graph Technique. Remote Sensing, 5(7), 3259-3279. https://doi.org/10.3390/rs5073259

- [3] Método comparativo. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076

- [4] Leung, W. H., & Bovy, J. (2024). Galaxy10 DECaLS [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10845026

- [5] OpenCV. Image Thresholding. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

## Grabación y entrega

1. Asignar diapositivas a todos los integrantes y ensayar transiciones.
2. Grabar la exposición de 10 minutos con la presentación en pantalla.
3. Grabar la demo de máximo 5 minutos, con Colab preparado.
4. Comprobar voz, legibilidad, duración y participación antes de exportar a MP4.
5. Entregar el PDF y el video según Canvas. Los archivos de esta carpeta preparan la grabación; el video con las voces y participación del equipo todavía debe grabarse.
