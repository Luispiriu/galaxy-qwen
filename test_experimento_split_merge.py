"""Pruebas del protocolo y la reanudación; no ejecutan Qwen ni escriben en MLflow."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluar_qwen as experimento


class ExperimentoSplitMergeTests(unittest.TestCase):
    def test_interrupcion_reanudacion_y_comparacion(self):
        datos = pd.DataFrame({"image_id": range(10), "label": range(10)})
        imagenes = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        imagenes[:, 8:24, 8:24] = 200
        args = SimpleNamespace(
            dataset="dataset_sintetico.h5", model="modelo_sintetico", seed=42,
            max_new_tokens=16, sm_tau=10., sm_min_size=4, sm_fondo_k=3.,
            experiment="prueba",
        )
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent, prefix=".test_split_merge_"
        ) as temporal, ExitStack() as stack:
            salida = Path(temporal)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(experimento, "obtener_run_id", return_value="fake"))
            stack.enter_context(patch.object(experimento.mlflow, "start_run", side_effect=lambda **kw: nullcontext()))
            for funcion in ("log_params", "log_metrics", "log_artifacts", "set_tags"):
                stack.enter_context(patch.object(experimento.mlflow, funcion))

            def ejecutar(nombre):
                return experimento.ejecutar_tratamiento(
                    nombre, datos, imagenes, None, None, args, None, salida
                )

            nombre = "gaussian_split_merge"
            # Interrupción después de tres predicciones ya persistidas.
            respuestas = [(i, str(i), True, .1) for i in range(3)]
            with patch.object(experimento, "clasificar", side_effect=respuestas + [RuntimeError("interrupción")]):
                with self.assertRaisesRegex(RuntimeError, "interrupción"):
                    ejecutar(nombre)
            ruta = salida / nombre / "predictions.csv"
            self.assertEqual(len(pd.read_csv(ruta)), 3)
            with patch.object(experimento, "clasificar", side_effect=[
                (i, str(i), True, .1) for i in range(3, 10)
            ]) as clasificar:
                resultado = ejecutar(nombre)
                self.assertEqual(clasificar.call_count, 7)
            self.assertEqual(resultado["accuracy"], 1.)
            df = pd.read_csv(ruta)
            self.assertEqual(len(df), 10)
            self.assertFalse(df.image_id.duplicated().any())
            self.assertTrue(set(experimento.COLUMNAS_SPLIT_MERGE).issubset(df.columns))
            self.assertTrue((salida / nombre / "confusion_matrix.png").exists())
            with patch.object(experimento, "clasificar") as clasificar:
                ejecutar(nombre)
                clasificar.assert_not_called()

            # Completar el segundo tratamiento y conservar ambos en la tabla.
            with patch.object(experimento, "clasificar", side_effect=[
                (i, str(i), True, .1) for i in range(10)
            ]):
                ejecutar("clahe_split_merge")
            tabla = experimento.consolidar_resultados(salida, args, datos)
            self.assertEqual(set(tabla.treatment), set(experimento.TRATAMIENTOS_SPLIT_MERGE))

            # Cambiar parámetros no puede agregar predicciones a un CSV previo.
            args.sm_tau = 8.
            with self.assertRaisesRegex(RuntimeError, "Configuración distinta"):
                ejecutar(nombre)
            self.assertEqual(len(pd.read_csv(ruta)), 10)
            config = json.loads((salida / nombre / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["sm_tau"], 10.)


if __name__ == "__main__":
    unittest.main()
