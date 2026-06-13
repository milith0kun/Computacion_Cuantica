import json

path = r"d:\Proyectos\Cuantica\Proyecto Segunda Parte\Colab\Proyecto_Segunda_Parte.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'][0] = "# IMPLEMENTACIÓN DE UN MODELO DE APRENDIZAJE AUTOMÁTICO CUÁNTICO UTILIZANDO QISKIT MACHINE LEARNING\n"
nb['cells'][0]['source'].insert(1, "## Proyecto Segunda Parte: Experimentación en Hardware Cuántico Real de IBM\n")

new_path = r"d:\Proyectos\Cuantica\GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_CODIGO.ipynb"
with open(new_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
