import re

latex_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Informe\Informe.tex"

keywords = ["TODO", "??", "placeholder", "pendiente", "completar", "ejercicio", "grafico", "figura"]

with open(latex_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE):
            print(f"Línea {idx+1}: {line.strip()} (Keyword: {kw})")
            break
