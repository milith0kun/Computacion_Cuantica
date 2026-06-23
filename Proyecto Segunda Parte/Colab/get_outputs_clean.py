import json
import io

p = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Colab/GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_CODIGO.ipynb'

with io.open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("--- Cell 41 Outputs ---")
for o in nb['cells'][41].get('outputs', []):
    if 'text' in o:
        print("".join(o['text']))

print("--- Cell 42 Outputs ---")
for o in nb['cells'][42].get('outputs', []):
    if 'text' in o:
        print("".join(o['text']))
