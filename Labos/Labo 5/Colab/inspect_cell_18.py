import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][18]
source = "".join(cell.get('source', []))
print("--- CODIGO ---")
print(source)
print("--- OUTPUTS ---")
for out in cell.get('outputs', []):
    if 'text' in out:
        print("".join(out['text']))
    elif 'data' in out:
        print("Mime types:", list(out['data'].keys()))
