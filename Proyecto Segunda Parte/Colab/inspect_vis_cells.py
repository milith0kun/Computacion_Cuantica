import json
import io

p = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Colab/GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_CODIGO.ipynb'

with io.open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [10, 11, 13, 14, 15, 16]:
    print(f"=== Cell {idx} ===")
    src = "".join(nb['cells'][idx].get('source', []))
    print(src)
    print("\n" + "="*40 + "\n")
