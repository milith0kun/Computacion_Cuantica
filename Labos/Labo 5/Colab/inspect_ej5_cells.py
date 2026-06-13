import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c_idx in [23, 28, 29]:
    cell = nb['cells'][c_idx]
    print(f"=== CELDA {c_idx} ===")
    print("".join(cell.get('source', [])))
    print("=" * 30)
