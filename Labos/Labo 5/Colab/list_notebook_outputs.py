import json
import os
import base64

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"
output_dir = r"d:\Proyectos\Cuantica\Labos\Labo 5\Informe\imagenes"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total de celdas: {len(nb['cells'])}")

img_count = 0
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type', '')
    if cell_type == 'code':
        outputs = cell.get('outputs', [])
        for out_idx, out in enumerate(outputs):
            if 'data' in out:
                data = out['data']
                for mime, content in data.items():
                    if mime.startswith('image/'):
                        img_count += 1
                        print(f"Celda {idx}, Output {out_idx}: Tipo {mime}, Tamaño {len(content)}")
                        # Guardar imagen para ver qué es
                        img_data = base64.b64decode(content)
                        filename = f"extracted_cell_{idx}_out_{out_idx}.png"
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'wb') as img_f:
                            img_f.write(img_data)
                        print(f"  Guardado en: {filepath}")
