# Galaxy10: presentación en Canva

- Versión actual de 39 diapositivas, con análisis ampliado del artículo: https://www.canva.com/d/TDyXy5CJMVsJykX
- Editor actual: https://www.canva.com/d/XXyHnIu-gf-PTGw
- ID del diseño actual: DAHZux_grJ8
- PDF actual: `output/pdf/Presentacion_Galaxy10_Tec_Articulo_Ampliado.pdf`.
- Versión anterior de 32 diapositivas: https://www.canva.com/d/2Vwhx93N5knz0D4

El bloque nuevo ocupa las diapositivas 14-20: formulación, representación del árbol, procedimiento original, agrupamiento y limpieza, resultados del artículo, lectura crítica y diferencias con el proyecto. Su análisis detallado está en `Analisis_Articulo_Horowitz_Pavlidis.md`. Las páginas posteriores se desplazaron siete posiciones.

La versión contiene portada con logo institucional, separadores Método / Resultado / Conclusiones, los 22 contenidos anteriores y cinco páginas adicionales de análisis de MLflow, además de dos páginas de referencias.

## Nuevas páginas de resultados

- 21: accuracy, precisión macro, recall macro y F1 macro.
- 23: comparación emparejada original frente a CLAHE + Otsu: 94 errores corregidos, 63 aciertos perdidos y 31 aciertos netos. Se mantienen 219 imágenes correctas y 624 incorrectas en ambos tratamientos.
- 25: F1 por clase en los siete tratamientos, con los errores persistentes de las clases 3 y 6.
- 26: porcentaje de píxeles conservados, componentes conectados de las máscaras y regiones antes/después de la fusión. Los números son medias sobre 1,000 imágenes, con redondeo.
- 27: tiempo medio de generación de Qwen y preprocesamiento. Las mediciones de preprocesamiento solo existen para los dos tratamientos nuevos. El tiempo de generación no representa el tiempo total de ejecución.

Los valores se comprobaron contra `latest_metrics` en `mlflow.db` y los archivos de `resultados_qwen/`. Los conteos emparejados proceden de las predicciones individuales; no constituyen una prueba de significancia.

Alcance: 1,000 imágenes únicas de validación, 100 por clase, siete tratamientos, 7,000 clasificaciones. Esta versión no incluye la partición de prueba final.

## Fuentes de identidad visual

- Logo descargado del sitio oficial: https://tec.mx/sites/default/files/repositorio/Marketing/logotipo-institucional-tec-de-monterrey.jpg
- Lineamientos y azul institucional: https://bamboo.tec.mx/latest/foundations/lineamientos-de-marca-tec-l7luyCtI

Para la exposición de diez minutos, utilizar las páginas 31-32 como referencias de consulta y ensayar la selección de énfasis. El guion anterior de 22 diapositivas pertenece a la versión anterior y no coincide con la numeración de esta versión.
