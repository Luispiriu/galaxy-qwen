
import time
from pathlib import Path

import requests

URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1"

DESTINO = Path("data/Galaxy10_DECals.h5")
TEMPORAL = Path("data/Galaxy10_DECals.h5.part")

DESTINO.parent.mkdir(parents=True, exist_ok=True)

def descargar():
    for intento in range(1, 11):
        descargados = TEMPORAL.stat().st_size if TEMPORAL.exists() else 0

        headers = {"Accept-Encoding": "identity"}
        if descargados:
            headers["Range"] = f"bytes={descargados}-"

        try:
            with requests.get(
                URL,
                headers=headers,
                stream=True,
                timeout=(20, 120),
            ) as respuesta:
                respuesta.raise_for_status()

                # 206 indica que el servidor aceptó reanudar.
                if descargados and respuesta.status_code != 206:
                    print("El servidor no permite reanudar. Reiniciando descarga.")
                    descargados = 0

                modo = "ab" if descargados else "wb"

                content_range = respuesta.headers.get("Content-Range", "")
                if descargados and not content_range.startswith(
                    f"bytes {descargados}-"
                ):
                    raise RuntimeError("El servidor devolvió un rango inesperado.")

                total = (
                    descargados
                    + int(respuesta.headers.get("Content-Length", 0))
                )

                with open(TEMPORAL, modo) as archivo:
                    for bloque in respuesta.iter_content(1024 * 1024):
                        if bloque:
                            archivo.write(bloque)

                if total and TEMPORAL.stat().st_size != total:
                    raise IOError("Descarga incompleta.")

                TEMPORAL.replace(DESTINO)
                print(f"Descarga completa: {DESTINO}")
                return

        except (requests.RequestException, IOError) as error:
            print(f"Intento {intento}/10: {error}")
            if intento < 10:
                time.sleep(5)

    raise RuntimeError("No se pudo completar la descarga.")

if __name__ == "__main__":
    if DESTINO.exists():
        print(f"Ya existe: {DESTINO}")
    else:
        descargar()
