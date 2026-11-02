# Análisis del artículo principal

**Referencia:** Horowitz, S. L., & Pavlidis, T. (1976). *Picture Segmentation by a Tree Traversal Algorithm*. Journal of the ACM, 23(2), 368-388. https://doi.org/10.1145/321941.321956

**Texto completo consultado:** copia académica alojada en Duke, https://courses.cs.duke.edu/spring06/cps296.1/handouts/Horowitz%20Pavlidis%201976.pdf. Se revisaron las 21 páginas y se comprobó visualmente la Tabla I, porque su extracción automática mezcla columnas.

## Problema y contribución

El trabajo propone organizar imágenes mediante regiones antes de introducir interpretación semántica. Combina movimientos de división y fusión en un árbol para reducir trabajo de procesamiento frente a partir exclusivamente de una imagen completa o de regiones mínimas. Su estructura permite reutilizar el procedimiento entre diferentes tipos de imágenes y variar unas pocas tolerancias. Esta motivación no constituye una promesa de mayor accuracy para cualquier clasificador posterior.

## Formulación: sección 2, páginas 370-372

El predicado P(S) es verdadero cuando existe una constante a tal que todos los valores de intensidad de la región difieren de a a lo sumo epsilon. Equivale a un rango máximo-mínimo no mayor que dos veces epsilon.

La partición debe cubrir la imagen, tener regiones disjuntas, satisfacer el predicado en cada región y evitar parejas adyacentes cuya unión siga satisfaciéndolo. El número de regiones no es necesariamente único ni el mínimo global. Se debe distinguir esta formulación de las etapas posteriores que introducen criterios por medias.

## Árbol, cutset e inicialización: secciones 3-4, páginas 372-374

Cada nodo representa un bloque cuadrado. El cutset es la selección activa de nodos que describe la partición: cada camino de la raíz a un píxel cruza un nodo seleccionado. La implementación conserva cinco arreglos con coordenadas, tamaño, intensidad máxima y mínima de los bloques activos, en vez de almacenar todos los nodos del árbol.

El procedimiento comienza en un nivel intermedio l0, elegido según el tamaño de regiones esperado. Fusiona grupos compatibles de cuatro hermanos y después divide bloques que exceden la tolerancia. Las operaciones sobre hermanos no eliminan todas las fronteras impuestas por el árbol, por lo que el artículo añade agrupamiento lateral.

## Agrupamiento y limpieza: sección 5, páginas 375-377

El agrupamiento examina bloques vecinos de diferentes ramas o niveles y actualiza sus máximos y mínimos. El artículo reconoce dependencia del orden de incorporación.

Después trata regiones pequeñas, ruido y zonas de transición con reglas basadas en tamaño y diferencia de medias. Esta limpieza relaja el criterio minimax inicial: no debe explicarse como si todas sus fusiones respetaran necesariamente la tolerancia original de rango.

## Contornos: sección 6, página 377

El etiquetado de regiones permite obtener contornos cerrados, localizar puntos donde se encuentran varias regiones y compactar tramos rectos conservando sus extremos. El resultado se propone como representación reducida para análisis posterior.

## Experimentos: sección 7 y Tabla I, página 378

Los autores trabajaron con imágenes de 64 × 64 y 7 bits en un IBM 360/91, usando compiladores PL/I y Fortran H. Presentaron rostros, radiografías de tórax, paisajes y cromosomas.

| Caso | Regiones después del agrupamiento | Regiones después de la limpieza | Tiempo reportado (s) |
|---|---:|---:|---:|
| Rostro | 148 | 13 | 4.59 |
| Radiografía | 49 | 11 | 1.45 |
| Paisaje | 186 | 13 | 3.62 |
| Cromosoma | 115 | 6 | 3.12 |

La nota de la tabla indica tiempos dominados por entrada/salida, con precisión de ±10 %. No deben compararse directamente con nuestra RTX 4060 ni presentarse como accuracy, F1 o validación de Galaxy10. El artículo también informa operaciones de split/merge, bloques, puntos de contorno y vértices.

## Conclusiones y lectura crítica: sección 8, páginas 378 y 387

La contribución es la reducción y organización de datos para un análisis semántico posterior. El trabajo reconoce la necesidad de semántica para interpretar una segmentación. Discute procesar imágenes mayores por bloques y agrupar sus resultados en otra pasada.

La evidencia presentada es principalmente visual y de procesamiento, sobre ejemplos de baja resolución. La Tabla I permite observar compactación, pero no prueba desempeño universal ni equivale a una evaluación moderna de clasificación. Esta lectura justifica estudiar la familia y evaluar explícitamente su adaptación a galaxias.

## Diferencias con nuestro código

| Aspecto | Artículo | Proyecto |
|---|---|---|
| Homogeneidad | Rango de intensidades | Desviación estándar |
| Inicio | Nivel intermedio del árbol | Imagen completa |
| Fusión | Hermanos y agrupamiento lateral | Regiones vecinas con cola de prioridad |
| Limpieza por tamaño/medias | Etapa posterior | No implementada como en el artículo |
| Extracción y compactación de contornos | Etapa explícita | No implementada |
| Máscara de primer plano | No usa nuestra regla astronómica | Mediana y MAD del perímetro |

Nuestro experimento evalúa una variante de la familia y su regla adicional de primer plano. Sus resultados no refutan el artículo original ni demuestran que todas las variantes de split and merge reduzcan el desempeño.

## Ubicación en la presentación

Se incorporaron siete diapositivas después de la introducción del artículo. El bloque nuevo ocupa las páginas 14-20 de `Presentacion_Galaxy10_Tec_Articulo_Ampliado.pdf`. La presentación ahora tiene 39 páginas, incluidas las referencias. El experimento de validación y sus resultados se conservan.
