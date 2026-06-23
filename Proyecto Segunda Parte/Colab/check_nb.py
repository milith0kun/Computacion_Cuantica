import json
filepath = r'd:\Proyectos\Cuantica\Proyecto Segunda Parte\Colab\GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_CODIGO.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total de celdas: {len(nb.get('cells', []))}")

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if source and 'CELDA VIS-3: ZZFeatureMap' in source[0]:
            print('\n--- CONTENIDO DE CELDA VIS-3 ---')
            for line in source:
                print(repr(line))
        
        if source and 'CELDA VIS-5: COMO LOS DATOS REALES SE CODIFICAN EN EL CIRCUITO' in source[0]:
            print('\n--- CONTENIDO DE CELDA VIS-5 ---')
            for line in source:
                print(repr(line))
