# Guion del experimento Galaxy10

Exposición objetivo: **10:00**, con 22 diapositivas principales. El anexo contiene referencias, métricas por clase, prompt completo y matrices de consulta. Ensayar con cronómetro y repartir la exposición entre todos los integrantes. El texto y los tiempos son una guía, no una duración garantizada.

Alcance de esta presentación: **1,000 imágenes de validación, 100 por clase, siete tratamientos y 7,000 clasificaciones**.

## 01. Título del proyecto (00:00-00:15)

Presentamos el experimento de clasificación morfológica de galaxias. Comparamos imágenes originales, transformaciones y dos métodos de segmentación. Explicaremos el procesamiento, el uso de Qwen y los resultados sobre mil imágenes de validación.

## 02. Problema y pregunta (00:15-00:40)

Las galaxias contienen estructuras débiles, ruido y objetos cercanos. El procesamiento puede facilitar su observación, pero también alterar las señales que utiliza un clasificador. Nuestra pregunta es cómo cambia el desempeño de Qwen al transformar y segmentar estas imágenes. La galaxia central es el objeto de clasificación, mientras que el resto del campo puede aportar información o introducir distractores.

## 03. Objetivo e hipótesis (00:40-01:00)

El objetivo es comparar siete tratamientos bajo condiciones comunes. La hipótesis contempla dos posibilidades: mejorar contraste o eliminar fondo podría ayudar a distinguir morfologías, y recortar o suavizar estructuras podría perjudicarlas. Por eso medimos clasificación y revisamos también ejemplos visuales.

## 04. Dataset Galaxy10 (01:00-01:25)

Usamos Galaxy10 DECaLS. Las imágenes son RGB de doscientos cincuenta y seis por doscientos cincuenta y seis píxeles. Las diez clases describen galaxias perturbadas, en interacción, suaves, espirales y de perfil. El mosaico muestra un ejemplo de validación de cada clase. Las etiquetas numéricas que aparecen aquí también identifican las filas y columnas de las matrices de confusión.

## 05. Validación (01:25-01:45)

La comparación se concentra en mil imágenes únicas de validación, con cien ejemplos de cada clase. Cada tratamiento recibe exactamente los mismos identificadores y etiquetas. Así podemos comparar los resultados sobre un conjunto equilibrado sin confundir el número de imágenes únicas con el número de ejecuciones.

## 06. Diseño experimental (01:45-02:15)

El flujo empieza con una imagen, aplica un tratamiento, consulta Qwen y compara la predicción con la etiqueta real. Tenemos siete condiciones: original, Gaussian, CLAHE, las dos combinaciones con Otsu y las dos combinaciones con split and merge. Guardamos las predicciones y métricas en MLflow. Cada condición procesa mil imágenes, así que el experimento completo comprende siete mil clasificaciones.

## 07. Referencia original (02:15-02:25)

La imagen original conserva color, forma, estructura, fondo y objetos cercanos. Su desempeño establece la referencia: veintiocho punto dos por ciento de accuracy y F1 macro de cero punto dos dos cero dos.

## 08. Gaussian (02:25-02:50)

El suavizado gaussiano combina intensidades vecinas y reduce variaciones rápidas. Usamos un kernel de cinco por cinco y sigma uno. Al comparar las dos imágenes se aprecia una reducción del ruido, junto con suavizado de detalles. Esto puede facilitar algunas señales globales, pero también disminuir la definición de brazos o estructuras finas. El experimento determina el efecto neto sobre Qwen.

## 09. CLAHE (02:50-03:15)

CLAHE modifica el contraste local con un límite de amplificación. En nuestro código convertimos a LAB, aplicamos CLAHE sobre el canal de luminancia y regresamos a RGB. Usamos clipLimit dos y una cuadrícula de ocho por ocho. El realce hace visibles diferencias locales, aunque también puede aumentar el ruido. Conservamos el mismo procedimiento para la variante con Otsu y con split and merge.

## 10. Otsu (03:15-03:50)

Otsu elige automáticamente un umbral global a partir del histograma. Su criterio maximiza la varianza entre las dos clases de intensidades. En la figura, la línea naranja indica el umbral calculado para una galaxia después del suavizado. El eje vertical usa escala logarítmica para mostrar también intensidades poco frecuentes. La máscara separa píxeles por intensidad, sin reconocer semánticamente la galaxia. En campos con objetos brillantes y estructura tenue, esa distinción puede diferir del objeto astronómico que queremos conservar.

## 11. Variantes con Otsu (03:50-04:20)

Aplicamos Otsu después de Gaussian y después de CLAHE. Aquí mostramos, para la misma galaxia, la imagen procesada, la máscara y el RGB segmentado. Qwen recibe la última columna, con colores conservados y fondo negro. Esto es importante: la imagen binaria sirve para construir la máscara, mientras que la clasificación utiliza RGB. El preprocesamiento cambia el histograma y por ello también puede cambiar el umbral y la extensión conservada.

## 12. Fundamento split and merge (04:20-04:50)

El antecedente principal es el artículo de Horowitz y Pavlidis de mil novecientos setenta y seis. Combina división y fusión para segmentar por regiones con una estructura jerárquica. Nuestra implementación pertenece a esa familia y usa desviación estándar, vecindad por borde y una regla propia de primer plano. La bibliografía complementaria incluye una aplicación de quadtree y grafos de adyacencia a teledetección. Distinguimos esas referencias de las decisiones y resultados de nuestro proyecto.

## 13. Split (04:50-05:20)

La división comienza con toda la imagen. Si la desviación estándar supera diez y podemos crear cuatro hijos de al menos cuatro por cuatro píxeles, dividimos. Repetimos el procedimiento en cada hijo. Los colores identifican bloques, no clases de galaxias. La estructura genera bloques grandes en zonas uniformes y pequeños donde cambia la intensidad. El tamaño mínimo detiene el proceso incluso si queda heterogeneidad.

## 14. Merge (05:20-05:50)

Después revisamos regiones que comparten borde, incluso si provienen de distintas ramas del árbol. Fusionamos una pareja si la desviación de la unión no supera diez. Una cola prioriza la unión con menor desviación. Después de cada fusión actualizamos vecinos, sumas y conteos. El proceso continúa hasta que no quedan fusiones permitidas. En el mapa final desaparecen varias fronteras rectangulares introducidas durante la división.

## 15. Máscara split and merge (05:50-06:25)

El mapa de regiones todavía cubre todo el campo. Para seleccionar primer plano usamos la intensidad media de cada región y un umbral estimado con el perímetro de la imagen. Ese umbral combina la mediana del borde y tres veces una dispersión robusta basada en MAD, con mínimo de uno. Conservamos tau diez, tamaño mínimo cuatro y fondo k tres. La comparación muestra que Gaussian y CLAHE pueden producir máscaras distintas con esos mismos parámetros. La regla de fondo es una decisión adicional de nuestro código.

## 16. Qwen (06:25-06:55)

El clasificador es Qwen3-VL-4B-Instruct, cuantizado a cuatro bits con NF4 y ejecutado en una RTX cuatro mil sesenta. El prompt pide clasificar la galaxia principal cerca del centro y devolver únicamente un dígito del cero al nueve. Las categorías se mantienen fijas. La imagen que recibe el modelo cambia según el tratamiento. Aquí mostramos el extracto del prompt y una respuesta guardada correcta para una galaxia de perfil.

## 17. Control y MLflow (06:55-07:25)

Mantenemos imágenes, modelo, prompt y generación determinista. MLflow registra parámetros, predicciones, métricas y matrices para las siete ejecuciones. Todas terminaron con mil predicciones y cero respuestas inválidas. Accuracy mide la fracción de aciertos. F1 macro promedia el F1 de las diez clases con el mismo peso. El balance de la validación permite comparar las morfologías bajo igual soporte.

## 18. Resultados generales (07:25-08:00)

La tabla reúne los siete tratamientos. CLAHE más Otsu obtuvo el mejor resultado: treinta y uno punto tres por ciento de accuracy y F1 macro de cero punto dos cuatro siete dos. CLAHE sin segmentación quedó cerca. Gaussian más split and merge obtuvo veintiuno punto ocho por ciento de accuracy y CLAHE más split and merge dieciséis punto dos. Ambas variantes quedaron por debajo de la referencia original. Estos son resultados descriptivos de nuestra configuración y de esta validación.

## 19. Efecto de segmentar (08:00-08:30)

Separar las familias de preprocesamiento permite ver el efecto de agregar segmentación. En Gaussian, tanto Otsu como split and merge redujeron el F1 respecto al suavizado solo. En CLAHE, Otsu produjo una mejora pequeña y split and merge una caída marcada. El F1 pasó de cero punto dos tres ocho siete a cero punto cero nueve dos seis. La segmentación puede ayudar o perjudicar según la combinación y los parámetros.

## 20. Clases y errores (08:30-09:05)

Las matrices muestran aciertos en la diagonal y errores fuera de ella. Cada fila contiene cien ejemplos de una clase real. Comparamos CLAHE más Otsu con CLAHE más split and merge. La clase dos, Round Smooth, concentra cuatrocientas diez predicciones en el primer tratamiento y seiscientas cuatro en el segundo, aunque solo cien imágenes pertenecen realmente a ella. Este sesgo ayuda a explicar por qué un promedio global debe acompañarse de análisis por clase. El anexo contiene las siete matrices y todos los F1 por clase.

## 21. Limitaciones (09:05-09:35)

Los ejemplos muestran estructuras recortadas y objetos externos conservados. Gaussian más split and merge produjo cinco máscaras vacías y una completamente llena. Esos casos son verificables, pero no explican por sí solos toda la caída. Sin máscaras de referencia no calculamos IoU ni Dice. Tampoco realizamos una prueba de significancia. La pérdida de estructura es una hipótesis visual: el experimento evalúa conjuntamente particionado y selección de primer plano.

## 22. Conclusiones y demo (09:35-10:00)

Concluimos que CLAHE más Otsu obtuvo el mejor desempeño de esta validación. Las variantes evaluadas de split and merge redujeron el desempeño respecto a sus transformaciones base. El efecto depende de la información que conserva cada combinación. Ahora pasamos a la demo en Colab: segmentaremos tres galaxias representativas y consultaremos los resultados completos, calculados sobre mil imágenes por tratamiento.

## Demo adicional: 4:50

Abrir `Demo_Split_Merge_5_min.ipynb` en Google Colab y ejecutar la instalación antes de iniciar la grabación.

00:00-00:30: contextualizar los tres ejemplos de validación.
00:30-01:10: mostrar la función, el criterio de división y el de fusión.
01:10-02:30: ejecutar la segmentación y señalar regiones, máscara y RGB.
02:30-03:20: contrastar un caso conservado y los fallos.
03:20-04:15: mostrar los resultados completos de los siete tratamientos.
04:15-04:50: explicar límites y cerrar con las conclusiones.

Las predicciones de Qwen están calculadas previamente. La demo ejecuta segmentación en CPU; los tres ejemplos no sustituyen la validación completa.
