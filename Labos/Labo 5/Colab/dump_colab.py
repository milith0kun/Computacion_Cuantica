import json

notebook_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\Examen_Matrices_Estocasticas_Qiskit.ipynb"
output_text_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\colab_content.txt"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(output_text_path, 'w', encoding='utf-8') as out_f:
    for idx, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type', '')
        out_f.write(f"=== CELDA {idx} ({cell_type.upper()}) ===\n")
        source = "".join(cell.get('source', []))
        out_f.write(source)
        out_f.write("\n")
        outputs = cell.get('outputs', [])
        if outputs:
            out_f.write("--- SALIDAS ---\n")
            for out in outputs:
                if 'text' in out:
                    out_f.write("".join(out['text']))
                elif 'data' in out:
                    out_f.write(f"[MIME TYPES: {list(out['data'].keys())}]\n")
                    if 'text/plain' in out['data']:
                        out_f.write("".join(out['data']['text/plain']))
        out_f.write("\n\n" + "="*50 + "\n\n")

print(f"Notebook volcado a {output_text_path}")
