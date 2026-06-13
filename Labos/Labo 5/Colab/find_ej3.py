import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if "Ejercicio 3" in source or "ejercicio 3" in source or "ej3" in source or "Ej3" in source:
        print(f"Celda {idx} ({cell['cell_type']}):")
        print(source[:300])
        print("="*40)
