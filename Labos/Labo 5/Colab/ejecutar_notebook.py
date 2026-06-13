import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

print("Cargando el notebook...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print("Inicializando el procesador de ejecución (esto ejecutará todas las celdas)...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

# Ejecutar el notebook en la carpeta Colab para que los paths relativos funcionen
try:
    ep.preprocess(nb, {'metadata': {'path': r'd:\Proyectos\Cuantica\Labos\Labo 5\Colab'}})
    print("Notebook ejecutado con éxito sin errores.")
except Exception as e:
    print("Ocurrió un error durante la ejecución del notebook:", e)

print("Guardando el notebook ejecutado...")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook guardado con éxito con todas las salidas e imágenes actualizadas.")
