import json
import io

p = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Colab/GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_CODIGO.ipynb'

with io.open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 19 contains the VQC corregido
cell = nb['cells'][19]
print("--- Cell 19 Outputs ---")
for o in cell.get('outputs', []):
    if 'text' in o:
        print("".join(o['text']))
    elif 'data' in o and 'text/plain' in o['data']:
        print("".join(o['data']['text/plain']))
