import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells_to_inspect = [11, 15, 21, 23, 28, 29, 33]

for idx in cells_to_inspect:
    cell = nb['cells'][idx]
    print(f"=== CELDA {idx} ===")
    source = "".join(cell.get('source', []))
    print(source[:500]) # Mostrar primeros 500 caracteres
    print("=" * 20)
