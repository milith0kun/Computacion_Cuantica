"""
Exporta presentacion.html a un PDF 16:9 (widescreen estandar).

Reutiliza las capturas 1920x1080 generadas para el PPTX y las combina en un
PDF con img2pdf (no recomprime las imagenes, mantiene calidad maxima).

Si no hay capturas, regenera primero llamando a exportar_pptx.py.

Uso:
    python exportar_pdf.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS_DIR = HERE / "_pptx_shots"
OUT_PDF = HERE / "presentacion.pdf"
EXPORTER = HERE / "exportar_pptx.py"


def ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"  -Instalando {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])


def main() -> int:
    # 1. Asegurar que tenemos capturas frescas
    shots = sorted(SHOTS_DIR.glob("slide_*.png")) if SHOTS_DIR.exists() else []
    if not shots:
        print("[*] No hay capturas previas. Generandolas con exportar_pptx.py...")
        result = subprocess.call([sys.executable, str(EXPORTER)])
        if result != 0:
            print("[X] Fallo al generar capturas.")
            return result
        shots = sorted(SHOTS_DIR.glob("slide_*.png"))

    if not shots:
        print("[X] No se encontraron capturas para procesar.")
        return 1

    print(f"[*] {len(shots)} capturas detectadas en {SHOTS_DIR.name}/")

    # 2. Combinar en PDF con img2pdf (lossless)
    ensure_pkg("img2pdf")
    import img2pdf

    # Pagina en pulgadas: 13.333 x 7.5 (PowerPoint widescreen 16:9 estandar)
    page_w_pt = img2pdf.in_to_pt(13.333)
    page_h_pt = img2pdf.in_to_pt(7.5)
    layout = img2pdf.get_layout_fun(pagesize=(page_w_pt, page_h_pt))

    print(f"[*] Ensamblando PDF 13.33\" x 7.5\" (16:9)...")
    with open(OUT_PDF, "wb") as f:
        f.write(img2pdf.convert([str(s) for s in shots], layout_fun=layout))

    size_mb = OUT_PDF.stat().st_size / 1024 / 1024
    print()
    print(f"[OK] PDF generado: {OUT_PDF.name}")
    print(f"  - {len(shots)} paginas")
    print(f"  - {size_mb:.2f} MB")
    print(f"  - Resolucion: 1920x1080 nativo (sin recompresion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
