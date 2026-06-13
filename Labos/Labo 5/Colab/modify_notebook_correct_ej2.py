import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Buscar la celda con el código del Ejercicio 2
# Es la celda 14
cell = nb['cells'][14]
source = "".join(cell.get('source', []))

print("Original source of cell 14:")
print(source)

# Reemplazar theta_1_c = 2 * np.arcsin(np.sqrt(0.8)) con theta_1_c = 2 * np.arcsin(np.sqrt(0.2))
new_source = source.replace("theta_1_c = 2 * np.arcsin(np.sqrt(0.8)) # Fila 1 transicion a 1 es 0.8", 
                            "theta_1_c = 2 * np.arcsin(np.sqrt(0.2)) # Fila 1 transicion a 0 es 0.2")

cell['source'] = [line + '\n' for line in new_source.split('\n')]
# Quitar el caracter de salto de línea extra al final si lo hay
if cell['source'][-1] == '\n':
    cell['source'] = cell['source'][:-1]

print("Modified source of cell 14:")
print("".join(cell['source']))

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook modificado con éxito.")
