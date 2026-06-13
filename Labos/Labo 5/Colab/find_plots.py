import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if "savefig" in source or "plt.figure" in source or "plt.subplots" in source:
        print(f"Celda {idx} tiene graficos:")
        for line in source.split('\n'):
            if "savefig" in line or "plt.figure" in line or "plt.subplots" in line:
                print(f"  {line}")
