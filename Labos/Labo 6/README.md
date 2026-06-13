# Labo 6 - Google Colab

Base para desarrollar las actividades del laboratorio en un notebook compatible con Google Colab.

## Entorno recomendado

- Google Colab: runtime administrado por Colab.
- Local: Python 3.11.9 con `.venv` y `pip`.

En esta maquina no esta instalado `uv`, por eso el fallback local sera:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## Estructura

- `Labo_6_Colab.ipynb`: notebook principal.
- `requirements.txt`: dependencias para Colab o ejecucion local.
- `data/raw/`: datos originales.
- `data/processed/`: datos procesados.
- `reports/`: graficos, tablas o resultados exportados.
- `lib/`: funciones reutilizables.
- `scripts/`: scripts auxiliares.
